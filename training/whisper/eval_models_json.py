"""
Eval HF Whisper checkpoints → mate-style baseline JSON (WER/CER/CES/size/params/MACs).

Output shape (one file, many models)::

  {
    "kin-only-sunbird-e10": {
      "baseline": {
        "params": ...,
        "macs": ...,
        "size": ...,
        "languages": {
          "kinyarwanda": {"wer", "cer", "ces": null, "combined_error"},
          "kidawida": {...}
        }
      }
    },
    ...
  }

Example::

  python training/whisper/eval_models_json.py \\
    --config config/whisper_eval_models.yaml \\
    --output_dir /path/to/out
"""
from __future__ import annotations

import argparse
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
from compression.eval_metrics import (  # noqa: E402
    ASREvaluator,
    count_parameters,
    estimate_macs_whisper,
    measure_model_size_bytes,
)


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


def model_key_from_path(model_path: str, override: str | None = None) -> str:
    if override:
        return override
    name = Path(model_path).name
    if name in ("final", "checkpoint"):
        return Path(model_path).parent.name
    return name


def evaluate_languages(model, processor, langs: dict, cfg: dict) -> dict:
    ev_cfg = cfg.get("eval") or cfg.get("qat") or {}
    cc = cfg.get("curriculum") or {}
    score_bs = int(ev_cfg.get("score_batch_size", cc.get("score_batch_size", 32)))
    num_workers = int(ev_cfg.get("score_num_workers", cc.get("score_num_workers", 16)))
    max_new = int(ev_cfg.get("score_max_new_tokens", cc.get("score_max_new_tokens", 128)))

    out = {}
    model.eval()
    for name, lang in langs.items():
        print(f"  Evaluating {name} ({len(lang['eval'])} clips)...", flush=True)
        refs, hyps = curriculum.transcribe_dataset(
            model, processor, lang["eval"], lang["token_id"],
            batch_size=score_bs, num_workers=num_workers, max_new_tokens=max_new,
        )
        ev = ASREvaluator()
        ev.compute_wer(refs, hyps)
        ev.compute_cer(refs, hyps)
        row = ev.to_dict()
        row["ces"] = None  # mate: baseline CES is always null
        out[name] = {
            "wer": row["wer"],
            "cer": row["cer"],
            "ces": None,
            "combined_error": row["combined_error"],
        }
        print(
            f"  {name}: WER {ev.wer:.4f}  CER {ev.cer:.4f}  "
            f"combined_error {row['combined_error']:.4f}",
            flush=True,
        )
    return out


def eval_one(model_path: str, key: str, cfg: dict, langs: dict, device: torch.device) -> dict:
    print(f"\n=== {key}  path={model_path} ===", flush=True)
    model, processor = load_whisper(model_path, device)
    languages = evaluate_languages(model, processor, langs, cfg)
    baseline = {
        "params": count_parameters(model),
        "macs": estimate_macs_whisper(model),
        "size": measure_model_size_bytes(model),
        "languages": languages,
    }
    del model
    torch.cuda.empty_cache()
    return {"baseline": baseline}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_path", action="append", dest="model_paths", default=None,
        help="HF Whisper final dir (repeatable). Overrides config models: list if set.",
    )
    parser.add_argument(
        "--model_key", action="append", dest="model_keys", default=None,
        help="Optional JSON key per --model_path (same order).",
    )
    parser.add_argument("--output_dir", required=True, help="Writes results.json here")
    parser.add_argument(
        "--languages", default=None,
        help="Comma list to eval subset, e.g. kinyarwanda (default: all in config)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-eval even if model key already in results.json",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results = load_json(results_path)

    # models: from CLI or YAML list [{path, key?}, ...] or single checkpoint
    pairs: list[tuple[str, str]] = []
    if args.model_paths:
        keys = args.model_keys or []
        for i, mp in enumerate(args.model_paths):
            key = keys[i] if i < len(keys) else model_key_from_path(mp)
            pairs.append((mp, key))
    else:
        for item in cfg.get("models") or []:
            if isinstance(item, str):
                pairs.append((item, model_key_from_path(item)))
            else:
                mp = item["path"]
                pairs.append((mp, item.get("key") or model_key_from_path(mp)))
        if not pairs and cfg.get("checkpoint"):
            mp = cfg["checkpoint"]
            pairs.append((mp, model_key_from_path(mp)))

    if not pairs:
        raise SystemExit("Provide --model_path or config models: / checkpoint:")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    langs = whisper_train.build_language_datasets(cfg)
    if args.languages:
        keep = {x.strip() for x in args.languages.split(",") if x.strip()}
        langs = {k: v for k, v in langs.items() if k in keep}
        if not langs:
            raise SystemExit(f"No languages left after filter {keep}")

    for mp, key in pairs:
        if not Path(mp).exists():
            raise FileNotFoundError(f"model_path not found: {mp}")
        if key in results and "baseline" in results[key] and not args.force:
            print(f"Skipping {key} (already in results.json; use --force to redo)", flush=True)
            continue
        results[key] = eval_one(mp, key, cfg, langs, device)
        save_json(results, results_path)
        print(f"  Wrote {key} → {results_path}", flush=True)

    print(f"\nDone. Results → {results_path}", flush=True)


if __name__ == "__main__":
    main()
