"""
Export mate-style prediction CSVs for Whisper FT checkpoints.

Writes three files under --output_dir:

  kinyarwanda_kin_only_dev_predictions.csv
    prediction_kin-only-sunbird-e10
    prediction_kin-only-curriculum-e10

  kinyarwanda_combined_dev_predictions.csv
    prediction_kin-dav-balanced-27h-curriculum-e15
    prediction_kin-dav-balanced-27h-e15-nocurriculum

  kidawida_combined_dev_predictions.csv
    (same two combined model columns)

Columns: utterance_id (Orchard audio path), reference (original text), prediction_*.

Example::

  python training/whisper/export_predictions_csv.py \\
    --config config/whisper_eval_models.yaml \\
    --output_dir /project/community/rmwisene/pipeline_outputs/whisper_predictions
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curriculum  # noqa: E402
import train as whisper_train  # noqa: E402

# (output filename, language key in config, model keys)
BUNDLES: list[tuple[str, str, list[str]]] = [
    (
        "kinyarwanda_kin_only_dev_predictions.csv",
        "kinyarwanda",
        ["kin-only-sunbird-e10", "kin-only-curriculum-e10"],
    ),
    (
        "kinyarwanda_combined_dev_predictions.csv",
        "kinyarwanda",
        [
            "kin-dav-balanced-27h-curriculum-e15",
            "kin-dav-balanced-27h-e15-nocurriculum",
        ],
    ),
    (
        "kidawida_combined_dev_predictions.csv",
        "kidawida",
        [
            "kin-dav-balanced-27h-curriculum-e15",
            "kin-dav-balanced-27h-e15-nocurriculum",
        ],
    ),
]


def load_whisper(model_path: str, device: torch.device):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if not getattr(model.generation_config, "lang_to_id", None):
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    model.eval()
    return model, processor


def models_by_key(cfg: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in cfg.get("models") or []:
        if isinstance(item, str):
            key = Path(item).parent.name if Path(item).name == "final" else Path(item).name
            out[key] = item
        else:
            key = item.get("key") or Path(item["path"]).parent.name
            out[key] = item["path"]
    return out


def write_csv(path: Path, utterance_ids: list[str], references: list[str], pred_cols: dict[str, list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["utterance_id", "reference"] + [f"prediction_{k}" for k in pred_cols]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, uid in enumerate(utterance_ids):
            row = {"utterance_id": uid, "reference": references[i]}
            for key, hyps in pred_cols.items():
                row[f"prediction_{key}"] = hyps[i]
            w.writerow(row)
    print(f"Wrote {path} ({len(utterance_ids)} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML with languages: + models:")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--bundles",
        default=None,
        help="Comma list of CSV basenames to run (default: all three). "
        "Example: kinyarwanda_kin_only_dev_predictions.csv",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key_to_path = models_by_key(cfg)
    langs = whisper_train.build_language_datasets(cfg)

    ev_cfg = cfg.get("eval") or {}
    cc = cfg.get("curriculum") or {}
    score_bs = int(ev_cfg.get("score_batch_size", cc.get("score_batch_size", 32)))
    num_workers = int(ev_cfg.get("score_num_workers", cc.get("score_num_workers", 16)))
    max_new = int(ev_cfg.get("score_max_new_tokens", cc.get("score_max_new_tokens", 128)))

    want = None
    if args.bundles:
        want = {b.strip() for b in args.bundles.split(",") if b.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Cache decode results per (model_key, language) so combined models run once per lang
    cache: dict[tuple[str, str], tuple[list[str], list[str], list[str]]] = {}

    def decode(model_key: str, lang_name: str):
        ck = (model_key, lang_name)
        if ck in cache:
            return cache[ck]
        if model_key not in key_to_path:
            raise SystemExit(f"Model key {model_key!r} not in config models:")
        path = key_to_path[model_key]
        if not (Path(path) / "config.json").is_file():
            raise SystemExit(f"Missing checkpoint: {path}")
        if lang_name not in langs:
            raise SystemExit(f"Language {lang_name!r} not in config")

        lang = langs[lang_name]
        print(f"\n=== {model_key} on {lang_name} ({len(lang['eval'])} clips) ===", flush=True)
        print(f"  path={path}", flush=True)
        model, processor = load_whisper(path, device)
        paths, refs, hyps = curriculum.transcribe_dataset_for_export(
            model,
            processor,
            lang["eval"],
            lang["token_id"],
            batch_size=score_bs,
            num_workers=num_workers,
            max_new_tokens=max_new,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cache[ck] = (paths, refs, hyps)
        return cache[ck]

    for csv_name, lang_name, model_keys in BUNDLES:
        if want is not None and csv_name not in want:
            print(f"Skipping {csv_name}", flush=True)
            continue

        pred_cols: dict[str, list[str]] = {}
        utterance_ids = references = None
        for mk in model_keys:
            paths, refs, hyps = decode(mk, lang_name)
            if utterance_ids is None:
                utterance_ids, references = paths, refs
            elif paths != utterance_ids:
                raise SystemExit(
                    f"utterance_id order mismatch for {mk} on {lang_name} "
                    f"(expected same eval set order)"
                )
            pred_cols[mk] = hyps

        assert utterance_ids is not None and references is not None
        write_csv(output_dir / csv_name, utterance_ids, references, pred_cols)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
