"""
Run int8 QAT Whisper inference on a language split → CSV + WER/CER.

Default config uses Kinyarwanda **test** + curriculum int8 quantized weights.

Example::

  python training/whisper/infer_int8_csv.py \\
    --config config/whisper_int8_kin_test.yaml \\
    --output_dir /project/community/rmwisene/pipeline_outputs/whisper_predictions \\
    --keep_long_audio
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum  # noqa: E402
import train as whisper_train  # noqa: E402
from compression.eval_metrics import ASREvaluator  # noqa: E402
from compression.quantize import get_qat_scheme, quantize_model  # noqa: E402


def load_int8_qat(base_model: str, quantized_dir: Path, scheme: str, device: torch.device):
    sd_path = quantized_dir / "quantized_state_dict.pt"
    if not sd_path.is_file():
        raise SystemExit(f"Missing {sd_path}")

    # Prefer processor saved with quantized dump; fall back to base HF final
    proc_src = str(quantized_dir) if (quantized_dir / "tokenizer_config.json").is_file() else base_model
    processor = WhisperProcessor.from_pretrained(proc_src)

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    if not getattr(model.generation_config, "lang_to_id", None):
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    model.generation_config.forced_decoder_ids = None

    quantize_model(model, config=get_qat_scheme(scheme)["base"])
    try:
        state = torch.load(sd_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(sd_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded int8 state_dict (missing={len(missing)}, unexpected={len(unexpected)})",
        flush=True,
    )
    model.to(device)
    model.eval()
    return model, processor


def write_csv(path: Path, utterance_ids, references, hyps, pred_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utterance_id", "reference", pred_col]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for uid, ref, hyp in zip(utterance_ids, references, hyps):
            w.writerow({"utterance_id": uid, "reference": ref, pred_col: hyp})
    print(f"Wrote {path} ({len(utterance_ids)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--language", default="kinyarwanda",
        help="Language key in config (default: kinyarwanda)",
    )
    parser.add_argument(
        "--keep_long_audio", action="store_true",
        help="Keep clips >30s and chunk-decode (full test row count)",
    )
    parser.add_argument(
        "--csv_name", default=None,
        help="Output CSV filename (default: kinyarwanda_test_int8_qat_predictions.csv)",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_name = args.language
    if lang_name not in cfg.get("languages", {}):
        raise SystemExit(f"Language {lang_name!r} not in config")

    base_model = cfg.get("base_model") or cfg.get("checkpoint")
    quantized_dir = Path(cfg["quantized_dir"])
    scheme = cfg.get("qat_scheme", "int8_weight_qat")
    pred_key = cfg.get("prediction_key", "int8-qat")
    pred_col = f"prediction_{pred_key}"

    if not base_model or not Path(base_model).exists():
        raise SystemExit(f"base_model missing: {base_model}")

    # Only load the requested language
    cfg_use = {**cfg, "languages": {lang_name: cfg["languages"][lang_name]}}
    langs = whisper_train.build_language_datasets(
        cfg_use,
        drop_long=not args.keep_long_audio,
        require_text=False,
        eval_only=True,
    )
    lang = langs[lang_name]

    ev_cfg = cfg.get("eval") or {}
    score_bs = int(ev_cfg.get("score_batch_size", 32))
    num_workers = int(ev_cfg.get("score_num_workers", 16))
    max_new = int(ev_cfg.get("score_max_new_tokens", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device}  language={lang_name}  clips={len(lang['eval'])}  "
        f"keep_long_audio={args.keep_long_audio}",
        flush=True,
    )
    print(f"base_model={base_model}", flush=True)
    print(f"quantized_dir={quantized_dir}", flush=True)

    model, processor = load_int8_qat(base_model, quantized_dir, scheme, device)
    paths, refs, hyps = curriculum.transcribe_dataset_for_export(
        model,
        processor,
        lang["eval"],
        lang["token_id"],
        batch_size=score_bs,
        num_workers=num_workers,
        max_new_tokens=max_new,
    )

    # Metrics only if we have real references
    n_refs = sum(1 for r in refs if str(r).strip() and str(r).strip().lower() != "nan")
    metrics = {
        "language": lang_name,
        "split": "eval_manifest",
        "n": len(refs),
        "n_with_reference": n_refs,
        "model": pred_key,
        "quantized_dir": str(quantized_dir),
        "base_model": base_model,
        "keep_long_audio": args.keep_long_audio,
    }
    if n_refs > 0:
        refs_n = [" ".join(curriculum._norm(r)) for r in refs]
        hyps_n = [" ".join(curriculum._norm(h)) for h in hyps]
        ev = ASREvaluator()
        ev.compute_wer(refs_n, hyps_n)
        ev.compute_cer(refs_n, hyps_n)
        metrics.update({
            "wer": ev.wer,
            "cer": ev.cer,
            "combined_error": ev.to_dict()["combined_error"],
        })
        print(
            f"WER={ev.wer:.4f}  CER={ev.cer:.4f}  combined_error={metrics['combined_error']:.4f}",
            flush=True,
        )
    else:
        metrics.update({"wer": None, "cer": None, "combined_error": None})
        print("No references in manifest — skipping WER/CER (hyp-only)", flush=True)

    csv_name = args.csv_name or f"{lang_name}_test_int8_qat_predictions.csv"
    write_csv(output_dir / csv_name, paths, refs, hyps, pred_col)
    metrics_path = output_dir / csv_name.replace(".csv", "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {metrics_path}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
