"""
QAT + short finetune for any Hugging Face Whisper checkpoint (torchao).

Model-agnostic: pass --model_path (any final/checkpoint dir). Data/languages
come from a whisper YAML (same shape as training configs).

Flow (matches teammate NeMo compression script):
  1) baseline WER + size
  2) QAT prepare (fake quant on Linear)
  3) short Seq2SeqTrainer finetune
  4) persist → fresh model + real Int8/Float8 weight-only PTQ
  5) re-eval WER + size → results.json

Example:
  python training/whisper/qat_finetune.py \\
    --config config/whisper_qat.yaml \\
    --model_path /path/to/any/whisper/final \\
    --quant int8_weight_qat \\
    --output_dir /project/.../compression/run1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import concatenate_datasets
from dotenv import load_dotenv
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum  # noqa: E402
import train as whisper_train  # noqa: E402
from compression.eval_metrics import count_parameters, measure_model_size_bytes  # noqa: E402
from compression.quantize import persist_after_finetune, prepare_qat  # noqa: E402


def load_json(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_whisper(model_path: str, device: torch.device):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if not getattr(model.generation_config, "lang_to_id", None):
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    return model, processor


def evaluate_languages(model, processor, langs: dict, cfg: dict) -> dict:
    """Per-language corpus WER on each language's eval set."""
    qat_cfg = cfg.get("qat") or {}
    score_bs = int(qat_cfg.get("score_batch_size", (cfg.get("curriculum") or {}).get("score_batch_size", 32)))
    num_workers = int(qat_cfg.get("score_num_workers", (cfg.get("curriculum") or {}).get("score_num_workers", 16)))
    max_new = int(qat_cfg.get("score_max_new_tokens", (cfg.get("curriculum") or {}).get("score_max_new_tokens", 128)))

    out = {}
    model.eval()
    for name, lang in langs.items():
        print(f"  Evaluating {name} ({len(lang['eval'])} clips)...", flush=True)
        _, wer = curriculum.score_wer(
            model, processor, lang["eval"], lang["token_id"],
            batch_size=score_bs, num_workers=num_workers, max_new_tokens=max_new,
        )
        out[name] = {"wer": float(wer), "n": int(len(lang["eval"]))}
        print(f"  {name}: WER {wer:.4f}", flush=True)
    return out


def short_finetune(model, processor, langs: dict, cfg: dict, output_dir: str) -> None:
    """One (or few) epoch(s) of bilingual/monolingual FT with current model (QAT-prepared)."""
    train_ds = whisper_train.combine(
        [l["train"] for l in langs.values()],
        [l["oversample"] for l in langs.values()],
    )
    eval_ds = concatenate_datasets([l["eval"] for l in langs.values()])
    wer_samples = whisper_train.pick_wer_samples(langs, n_per_lang=1)

    # Merge training knobs: base training: then qat.finetune: overrides (short run)
    ft_cfg = dict(cfg)
    training = dict(cfg.get("training") or {})
    training.update(cfg.get("qat", {}).get("finetune") or {})
    # Sensible short-FT defaults if not specified
    training.setdefault("num_train_epochs", 1)
    training.setdefault("eval_steps", 500)
    training.setdefault("save_steps", 500)
    training.setdefault("logging_steps", 50)
    training.setdefault("report_to", "none")
    training["early_stopping_patience"] = None
    ft_cfg["training"] = training

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer = whisper_train.build_trainer(
        model, processor, train_ds, eval_ds, ft_cfg, output_dir,
        wer_samples=wer_samples,
    )
    trainer.train()


def run_one_model(
    model_path: str,
    cfg: dict,
    schemes: list[str],
    output_dir: Path,
    skip_baseline: bool,
    existing: dict,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = Path(model_path).name
    if model_id in ("final", "checkpoint"):
        model_id = Path(model_path).parent.name
    print(f"\n=== Model: {model_id}  path={model_path} ===", flush=True)

    entry = existing.setdefault(model_id, {})
    langs = whisper_train.build_language_datasets(cfg)

    def fresh_loader():
        m, _ = load_whisper(model_path, device)
        return m

    # --- baseline ---
    if not skip_baseline and "baseline" not in entry:
        print("\n-- baseline --", flush=True)
        model, processor = load_whisper(model_path, device)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        entry["baseline"] = {
            "model_path": model_path,
            "params": count_parameters(model),
            "size": measure_model_size_bytes(model),
            "languages": lang_results,
        }
        save_json(existing, output_dir / "results.json")
        del model
        torch.cuda.empty_cache()
    elif "baseline" in entry:
        print("Skipping baseline (already in results.json)", flush=True)

    baseline_size = entry.get("baseline", {}).get("size")

    # --- QAT schemes ---
    quant_results = entry.setdefault("quantization", {})
    for scheme in schemes:
        print(f"\n-- QAT scheme: {scheme} --", flush=True)
        q_entry = quant_results.setdefault(scheme, {})
        if "finetuned" in q_entry:
            print(f"  {scheme}: finetuned already in results — skip", flush=True)
            continue

        model, processor = load_whisper(model_path, device)
        prepare_qat(model, scheme)
        print("  QAT prepare done; starting short finetune...", flush=True)

        ft_dir = str(output_dir / model_id / scheme / "finetune")
        short_finetune(model, processor, langs, cfg, ft_dir)

        print("  Persisting real weight-only quantization...", flush=True)
        q_model = persist_after_finetune(model, fresh_loader, scheme)
        q_model.to(device)
        del model
        torch.cuda.empty_cache()

        print("  Evaluating quantized model...", flush=True)
        lang_results = evaluate_languages(q_model, processor, langs, cfg)
        q_size = measure_model_size_bytes(q_model)
        q_entry["finetuned"] = {
            "params": count_parameters(q_model),
            "size": q_size,
            "size_ratio_vs_baseline": (q_size / baseline_size) if baseline_size else None,
            "languages": lang_results,
        }

        # Save quantized weights for later use
        save_dir = output_dir / model_id / scheme / "quantized"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q_model.state_dict(), save_dir / "quantized_state_dict.pt")
        processor.save_pretrained(save_dir)
        # Keep a pointer to the original HF config/tokenizer for reload experiments
        (save_dir / "SOURCE_MODEL.txt").write_text(model_path + "\n")
        print(f"  Saved quantized state_dict → {save_dir}", flush=True)

        save_json(existing, output_dir / "results.json")
        del q_model
        torch.cuda.empty_cache()

    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Whisper YAML (languages + training/qat knobs)")
    parser.add_argument(
        "--model_path", action="append", dest="model_paths", default=None,
        help="HF Whisper dir (final/checkpoint). Repeatable. Overrides config checkpoint if set.",
    )
    parser.add_argument(
        "--quant", action="append", dest="schemes", default=None,
        help="QAT scheme: int8_weight_qat and/or float8_weight_qat (repeatable)",
    )
    parser.add_argument("--output_dir", required=True, help="Where to write results.json + artifacts")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline if already computed")
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = args.model_paths or []
    if not model_paths:
        if cfg.get("checkpoint"):
            model_paths = [cfg["checkpoint"]]
        else:
            raise SystemExit("Provide --model_path or set checkpoint: in the YAML")

    schemes = args.schemes or (cfg.get("qat") or {}).get("schemes") or ["int8_weight_qat"]

    results_path = output_dir / "results.json"
    results = load_json(results_path)

    for mp in model_paths:
        if not Path(mp).exists():
            raise FileNotFoundError(f"model_path not found: {mp}")
        results = run_one_model(
            mp, cfg, schemes, output_dir,
            skip_baseline=args.skip_baseline,
            existing=results,
        )
        save_json(results, results_path)

    print(f"\nDone. Results → {results_path}", flush=True)


if __name__ == "__main__":
    main()
