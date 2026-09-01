"""
Fine-tune Whisper (Sunbird SALT) on combined Kinyarwanda + Kidaw'ida.

Run from the repo root:
  python training/whisper/train.py --config config/whisper_config.yaml --curriculum
      # WER-rank at each stage with current model → 20/50/70/100% easiest
  python training/whisper/train.py --config config/whisper_config.yaml --eval_only
  python training/whisper/train.py --config config/whisper_config.yaml --curriculum --resume
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import (
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum

AUDIO_COLS = ("audio_path", "audio_filepath", "path", "filename", "file", "audio")
TEXT_COLS = ("transcript", "text", "sentence", "transcription")
DURATION_COLS = ("duration_sec", "duration")
MAX_LABEL_LEN = 448      # Whisper decoder context limit
MAX_AUDIO_SEC = 30.0     # Whisper encoder window; longer clips would silently truncate


def load_manifest(
    path: str,
    audio_dir: str | None = None,
    drop_long: bool = True,
    require_text: bool = True,
) -> pd.DataFrame:
    """Read a TSV/CSV/JSON/JSONL manifest and normalize to columns: audio, text."""
    p = Path(path)
    if p.suffix in (".tsv", ".csv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:
        df = pd.read_json(p, lines=(p.suffix == ".jsonl"))
    cols = {c.lower(): c for c in df.columns}
    audio_col = next(cols[c] for c in AUDIO_COLS if c in cols)
    # Prefer the text-like column with the most real (non-null, non-empty) values.
    text_candidates = [cols[c] for c in TEXT_COLS if c in cols]
    if not text_candidates:
        if require_text:
            raise ValueError(f"{p}: no transcript column among {TEXT_COLS}")
        text_col = None
        n_ok = 0
    else:
        def _text_score(col: str) -> int:
            s = df[col]
            as_str = s.map(lambda x: "" if pd.isna(x) else str(x).strip())
            # Treat literal "nan" from bad exports as empty
            return int((~as_str.isin(["", "nan", "None"])).sum())

        text_col = max(text_candidates, key=_text_score)
        n_ok = _text_score(text_col)

    if n_ok == 0:
        msg = f"{p}: no non-empty transcripts; columns={list(df.columns)}"
        if require_text:
            raise ValueError(msg)
        print(f"WARNING: {msg} — empty references (hyp-only)", flush=True)
    else:
        print(
            f"{p.name}: using audio={audio_col!r} text={text_col!r} ({n_ok}/{len(df)} non-null)",
            flush=True,
        )

    dur_col = next((cols[c] for c in DURATION_COLS if c in cols), None)
    if drop_long and dur_col:
        too_long = df[dur_col].astype(float) > MAX_AUDIO_SEC
        if too_long.any():
            print(f"{p.name}: dropping {int(too_long.sum())} clips longer than {MAX_AUDIO_SEC}s")
            df = df[~too_long]
    elif (not drop_long) and dur_col:
        n_long = int((df[dur_col].astype(float) > MAX_AUDIO_SEC).sum())
        if n_long:
            print(
                f"{p.name}: keeping {n_long} clips longer than {MAX_AUDIO_SEC}s "
                f"(chunked decode at export)",
                flush=True,
            )

    audio = df[audio_col].astype(str)
    if audio_dir:
        audio = audio.map(lambda a: str(Path(audio_dir) / a))
    if text_col is None:
        text = pd.Series([""] * len(df), index=df.index)
    else:
        text = df[text_col].where(df[text_col].notna(), "").astype(str)
        text = text.replace({"nan": "", "None": ""})
    out = {"audio": audio, "text": text}
    if dur_col:
        out["duration_sec"] = df[dur_col].astype(float)
    return pd.DataFrame(out)


def build_language_datasets(
    cfg: dict,
    drop_long: bool = True,
    require_text: bool = True,
    eval_only: bool = False,
) -> dict:
    """For each configured language: train/eval Datasets + token id + oversample factor."""
    out = {}
    for name, lc in cfg["languages"].items():
        entry = {"token_id": int(lc["lang_token_id"]), "oversample": int(lc.get("oversample", 1))}
        for split in ("train", "eval"):
            if eval_only and split == "train":
                entry[split] = Dataset.from_dict(
                    {"audio": [], "text": [], "lang_token_id": []}
                )
                continue
            # Hyp-only: allow empty text on eval (e.g. Kin test with no refs)
            need_text = require_text if split == "eval" else True
            if eval_only and split == "eval":
                need_text = require_text
            df = load_manifest(
                lc[f"{split}_manifest"],
                lc.get("audio_dir"),
                drop_long=drop_long,
                require_text=need_text,
            )
            df["lang_token_id"] = entry["token_id"]
            entry[split] = Dataset.from_pandas(df, preserve_index=False)
        out[name] = entry
    return out


def combine(datasets: list[Dataset], repeats: list[int]) -> Dataset:
    parts = [ds for ds, n in zip(datasets, repeats) for _ in range(n)]
    return concatenate_datasets(parts).shuffle(seed=42)


class WerSampleCallback(TrainerCallback):
    """NeMo-style mid-training prints: WER reference / WER predicted on fixed eval clips."""

    def __init__(self, processor, samples: list[dict], every_n_steps: int = 50):
        self.processor = processor
        self.samples = samples  # [{audio, text, lang_token_id}, ...]
        self.every_n_steps = every_n_steps

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        for ex in self.samples:
            language = self.processor.tokenizer.decode([ex["lang_token_id"]])
            feats = self.processor.feature_extractor(
                curriculum.load_audio(ex["audio"]), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=model.dtype)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                ids = model.generate(feats, language=language, task="transcribe")
            hyp = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            print(f"WER reference:{ex['text']}")
            print(f"WER predicted:{hyp}")
        if was_training:
            model.train()


def pick_wer_samples(langs: dict, n_per_lang: int = 2) -> list[dict]:
    """Fixed first-N eval clips per language so ref/pred are comparable across steps."""
    samples = []
    for name, lang in langs.items():
        n = min(n_per_lang, len(lang["eval"]))
        for i in range(n):
            row = lang["eval"][i]
            samples.append({
                "audio": row["audio"],
                "text": row["text"],
                "lang_token_id": row["lang_token_id"],
                "language": name,
            })
    return samples


class EpochProgressCallback(TrainerCallback):
    """Print clear epoch markers in the log (HF's float epoch is easy to miss)."""

    def on_train_begin(self, args, state, control, **kwargs):
        total = args.num_train_epochs
        print(f"\n=== Training start: target {total:g} epoch(s) "
              f"(max_steps={state.max_steps}) ===\n", flush=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch is the epoch index about to run (0-based float at boundaries)
        cur = int(state.epoch) + 1
        total = int(args.num_train_epochs)
        print(f"\n=== Epoch {cur}/{total} starting "
              f"(global_step={state.global_step}) ===\n", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        cur = int(state.epoch)
        total = int(args.num_train_epochs)
        print(f"\n=== Epoch {cur}/{total} finished "
              f"(global_step={state.global_step}) ===\n", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        ep = logs.get("epoch", state.epoch)
        step = state.global_step
        total_steps = state.max_steps
        print(f"[progress] step {step}/{total_steps} | epoch {ep:.4f}/{args.num_train_epochs:g}",
              flush=True)


def make_collator(processor):
    """Batch raw rows into (input_features, labels); labels get the per-row language token."""
    tok = processor.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe = tok.convert_tokens_to_ids("<|transcribe|>")
    notimestamps = tok.convert_tokens_to_ids("<|notimestamps|>")

    def collate(batch):
        feats = processor.feature_extractor(
            [curriculum.load_audio(ex["audio"]) for ex in batch],
            sampling_rate=16000, return_tensors="pt",
        ).input_features
        labels = []
        for ex in batch:
            ids = [sot, ex["lang_token_id"], transcribe, notimestamps]
            ids += tok(ex["text"], add_special_tokens=False).input_ids
            ids = ids[: MAX_LABEL_LEN - 1] + [tok.eos_token_id]
            labels.append(ids)
        pad = max(len(l) for l in labels)
        labels = torch.tensor([l + [-100] * (pad - len(l)) for l in labels])
        return {"input_features": feats, "labels": labels}

    return collate


def latest_valid_checkpoint(output_dir: str) -> str | None:
    """Newest checkpoint-* that has trainer_state.json (skip half-written TIME_LIMIT saves)."""
    root = Path(output_dir)
    candidates = []
    for p in root.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        if not (p / "trainer_state.json").is_file():
            print(f"Skipping incomplete checkpoint (no trainer_state.json): {p}")
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def report_per_language_wer(model, processor, langs: dict, cfg: dict) -> None:
    """Decode each language's eval set and print corpus WER (same as --eval_only)."""
    cc = cfg.get("curriculum") or {}
    score_bs = int(cc.get("score_batch_size", 32))
    num_workers = int(cc.get("score_num_workers", 16))
    max_new = int(cc.get("score_max_new_tokens", 128))
    print("\n=== Final per-language corpus WER (dev) ===", flush=True)
    for name, lang in langs.items():
        print(f"Evaluating {name} ({len(lang['eval'])} clips)...", flush=True)
        _, corpus_wer = curriculum.score_wer(
            model, processor, lang["eval"], lang["token_id"],
            batch_size=score_bs, num_workers=num_workers, max_new_tokens=max_new,
        )
        print(f"{name}: corpus WER {corpus_wer:.4f} over {len(lang['eval'])} samples", flush=True)
    print("=== End WER report ===\n", flush=True)


def build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir,
                  wer_samples=None, **overrides):
    tc = dict(cfg["training"])
    # null / 0 / missing → no early stopping (run full num_train_epochs)
    patience = tc.pop("early_stopping_patience", None)
    log_every = int(tc.get("logging_steps", 50))
    use_early_stop = patience is not None and int(patience) > 0
    # Defaults; anything under training: in YAML overrides (e.g. save_strategy: epoch).
    merged = {
        "output_dir": output_dir,
        "bf16": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": use_early_stop,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "remove_unused_columns": False,  # collator needs the raw audio/text columns
        **tc,
        **overrides,
    }
    args = Seq2SeqTrainingArguments(**merged)
    callbacks = [EpochProgressCallback()]
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))
    if wer_samples:
        callbacks.append(WerSampleCallback(processor, wer_samples, every_n_steps=log_every))
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(processor),
        callbacks=callbacks,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to whisper_config.yaml")
    parser.add_argument("--curriculum", action="store_true", help="Staged easiest-first training")
    parser.add_argument("--eval_only", action="store_true", help="Report per-language WER and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint under output_dir (for multi-job runs)")
    args = parser.parse_args()

    load_dotenv()  # HF_TOKEN for the gated Sunbird checkpoint
    cfg = yaml.safe_load(Path(args.config).read_text())

    processor = WhisperProcessor.from_pretrained(cfg["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["checkpoint"])
    if not getattr(model.generation_config, "lang_to_id", None):
        # outdated generation config (e.g. akera checkpoints): borrow the vanilla
        # large-v3 one so generate(language=...) knows the language-token map
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    model.generation_config.forced_decoder_ids = None
    if torch.cuda.is_available():
        model.to("cuda")

    langs = build_language_datasets(cfg)
    output_dir = cfg.get("output_dir", "./whisper_experiments")
    score_bs = cfg.get("curriculum", {}).get("score_batch_size", 32)

    if args.eval_only:
        report_per_language_wer(model, processor, langs, cfg)
        return

    eval_ds = concatenate_datasets([l["eval"] for l in langs.values()])
    wer_samples = pick_wer_samples(langs, n_per_lang=2)  # 2 kin + 2 dav, printed every logging_steps

    if args.curriculum:
        cc = cfg["curriculum"]
        schedule = cc["schedule"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Rescore with the current model at the start of every stage (easiest-first).
        # Per-stage .npy files let --resume skip re-scoring a stage already ranked.
        for stage, fraction in enumerate(schedule, start=1):
            print(f"\n=== Curriculum stage {stage}/{len(schedule)} (fraction={fraction}) ===")
            parts, repeats = [], []
            for name, lang in langs.items():
                score_path = Path(output_dir) / f"wer_difficulty_{name}_stage{stage}.npy"
                if score_path.is_file():
                    scores = np.load(score_path).tolist()
                    print(f"  {name}: loaded WER scores from {score_path} ({len(scores)} clips)")
                else:
                    print(f"  {name}: WER ranking ({len(lang['train'])} clips) with current model...")
                    scores, corpus_wer = curriculum.score_wer(
                        model, processor, lang["train"], lang["token_id"],
                        batch_size=score_bs,
                        num_workers=int(cc.get("score_num_workers", 16)),
                        max_new_tokens=int(cc.get("score_max_new_tokens", 128)),
                    )
                    print(f"  {name}: corpus WER {corpus_wer:.4f}")
                    np.save(score_path, np.asarray(scores, dtype=np.float32))
                    print(f"  {name}: saved {score_path}")
                ranked = sorted(range(len(scores)), key=lambda i: scores[i])  # lowest WER = easiest
                n = max(1, int(len(ranked) * fraction))
                keep = ranked[:n]
                print(f"  {name}: keeping {len(keep)}/{len(lang['train'])}")
                parts.append(lang["train"].select(keep))
                repeats.append(lang["oversample"])
            stage_dir = f"{output_dir}/stage_{stage}"
            trainer = build_trainer(
                model, processor, combine(parts, repeats), eval_ds, cfg,
                stage_dir, wer_samples=wer_samples,
                num_train_epochs=float(cc["epochs_per_stage"][stage - 1]))
            ckpt = latest_valid_checkpoint(stage_dir) if args.resume else None
            if args.resume and ckpt:
                print(f"Resuming from {ckpt}")
            elif args.resume:
                print(f"--resume set but no valid checkpoint under {stage_dir}; starting fresh.")
            trainer.train(resume_from_checkpoint=ckpt)
    else:
        train_ds = combine([l["train"] for l in langs.values()],
                           [l["oversample"] for l in langs.values()])
        trainer = build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir,
                                wer_samples=wer_samples)
        ckpt = latest_valid_checkpoint(output_dir) if args.resume else None
        if args.resume and ckpt:
            print(f"Resuming from {ckpt}")
        elif args.resume:
            print(f"--resume set but no valid checkpoint under {output_dir}; starting fresh.")
        trainer.train(resume_from_checkpoint=ckpt)

    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"Saved final model to {output_dir}/final")

    # Always report corpus WER on every language in the config after saving final
    report_per_language_wer(model, processor, langs, cfg)


if __name__ == "__main__":
    main()
