"""
Fine-tune a Whisper checkpoint (Sunbird SALT) on combined Kinyarwanda + Kidaw'ida.

Run from the repo root:
  python training/whisper/train.py --config config/whisper_config.yaml               # standard
  python training/whisper/train.py --config config/whisper_config.yaml --curriculum  # staged
  python training/whisper/train.py --config config/whisper_config.yaml --eval_only   # WER baseline
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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


def load_manifest(path: str, audio_dir: str | None = None) -> pd.DataFrame:
    """Read a TSV/CSV/JSON/JSONL manifest and normalize to columns: audio, text."""
    p = Path(path)
    if p.suffix in (".tsv", ".csv"):
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:
        df = pd.read_json(p, lines=(p.suffix == ".jsonl"))
    cols = {c.lower(): c for c in df.columns}
    audio_col = next(cols[c] for c in AUDIO_COLS if c in cols)
    text_col = next(cols[c] for c in TEXT_COLS if c in cols)
    dur_col = next((cols[c] for c in DURATION_COLS if c in cols), None)
    if dur_col:  # drop clips beyond Whisper's window: audio would truncate but labels wouldn't
        too_long = df[dur_col].astype(float) > MAX_AUDIO_SEC
        if too_long.any():
            print(f"{p.name}: dropping {int(too_long.sum())} clips longer than {MAX_AUDIO_SEC}s")
            df = df[~too_long]
    audio = df[audio_col].astype(str)
    if audio_dir:
        audio = audio.map(lambda a: str(Path(audio_dir) / a))
    return pd.DataFrame({"audio": audio, "text": df[text_col].astype(str)})


def build_language_datasets(cfg: dict) -> dict:
    """For each configured language: train/eval Datasets + token id + oversample factor."""
    out = {}
    for name, lc in cfg["languages"].items():
        entry = {"token_id": int(lc["lang_token_id"]), "oversample": int(lc.get("oversample", 1))}
        for split in ("train", "eval"):
            df = load_manifest(lc[f"{split}_manifest"], lc.get("audio_dir"))
            df["lang_token_id"] = entry["token_id"]
            # "audio" stays a path string; WAVs are read with soundfile at batch time
            entry[split] = Dataset.from_pandas(df, preserve_index=False)
        out[name] = entry
    return out


def combine(datasets: list[Dataset], repeats: list[int]) -> Dataset:
    parts = [ds for ds, n in zip(datasets, repeats) for _ in range(n)]
    return concatenate_datasets(parts).shuffle(seed=42)


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


def build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir, **overrides):
    tc = dict(cfg["training"])
    patience = tc.pop("early_stopping_patience", 4)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,  # collator needs the raw audio/text columns
        **{**tc, **overrides},
    )
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to whisper_config.yaml")
    parser.add_argument("--curriculum", action="store_true", help="Staged easiest-first training")
    parser.add_argument("--eval_only", action="store_true", help="Report per-language WER and exit")
    args = parser.parse_args()

    load_dotenv()  # HF_TOKEN for the gated Sunbird checkpoint
    cfg = yaml.safe_load(Path(args.config).read_text())

    processor = WhisperProcessor.from_pretrained(cfg["checkpoint"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["checkpoint"])
    model.generation_config.forced_decoder_ids = None
    if torch.cuda.is_available():
        model.to("cuda")

    langs = build_language_datasets(cfg)
    output_dir = cfg.get("output_dir", "./whisper_experiments")
    score_bs = cfg.get("curriculum", {}).get("score_batch_size", 32)

    if args.eval_only:
        for name, lang in langs.items():
            _, corpus_wer = curriculum.score_wer(
                model, processor, lang["eval"], lang["token_id"], batch_size=score_bs)
            print(f"{name}: corpus WER {corpus_wer:.4f} over {len(lang['eval'])} samples")
        return

    eval_ds = concatenate_datasets([l["eval"] for l in langs.values()])

    if args.curriculum:
        cc = cfg["curriculum"]
        for stage, fraction in enumerate(cc["schedule"], start=1):
            print(f"\n=== Curriculum stage {stage}/{len(cc['schedule'])} (fraction={fraction}) ===")
            parts, repeats = [], []
            for name, lang in langs.items():  # rank within each language so every stage is bilingual
                wers, corpus_wer = curriculum.score_wer(
                    model, processor, lang["train"], lang["token_id"], batch_size=score_bs)
                keep = curriculum.easiest_fraction(wers, fraction)
                print(f"  {name}: train WER {corpus_wer:.4f}, keeping {len(keep)}/{len(wers)}")
                parts.append(lang["train"].select(keep))
                repeats.append(lang["oversample"])
            trainer = build_trainer(
                model, processor, combine(parts, repeats), eval_ds, cfg,
                f"{output_dir}/stage_{stage}",
                num_train_epochs=float(cc["epochs_per_stage"][stage - 1]))
            trainer.train()
    else:
        train_ds = combine([l["train"] for l in langs.values()],
                           [l["oversample"] for l in langs.values()])
        trainer = build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir)
        trainer.train()

    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"Saved final model to {output_dir}/final")


if __name__ == "__main__":
    main()
