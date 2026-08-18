"""
Prune → QAT combination for Whisper (mate-compatible nested results.json).

For each prune level (10% / 20% / 50%) and each QAT scheme (int4 / int6 / int8):
  load pruned+FT checkpoint → QAT prepare → eval → short FT → persist PTQ → eval

JSON shape (matches teammate NeMo compression report)::

  {
    "<model_id>": {
      "baseline": { params, macs, size, languages },
      "pruning": {
        "10percent": {
          "languages" | "finetuned" | ...   # copied from prune run if seeded
          "quantization": {
            "int8_weight_qat": {
              "languages": {...},   # QAT-prepare
              "size": ...,
              "finetuned": { languages, size, params, macs }
            },
            "int4_weight_qat": {...},
            "int6_weight_qat": {...}
          }
        },
        "20percent": {...},
        "50percent": {...}
      }
    }
  }

Pruned checkpoints are expected at::

  {prune_root}/{model_id}/ffn_{10,20,50}/final
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
from compression.eval_metrics import (  # noqa: E402
    ASREvaluator,
    attach_ces_from_size,
    count_parameters,
    estimate_macs_whisper,
    measure_model_size_bytes,
)
from compression.quantize import (  # noqa: E402
    get_qat_scheme,
    list_schemes,
    persist_after_finetune,
    prepare_qat,
)


PERCENT_TO_FFN = {10: "ffn_10", 20: "ffn_20", 50: "ffn_50"}
FFN_TO_PERCENT = {v: k for k, v in PERCENT_TO_FFN.items()}


def load_json(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def percent_key(p: int) -> str:
    return f"{int(p)}percent"


def load_whisper(model_path: str, device: torch.device):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if not getattr(model.generation_config, "lang_to_id", None):
        model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    model.generation_config.forced_decoder_ids = None
    model.to(device)
    return model, processor


def score_knobs(cfg: dict) -> tuple[int, int, int]:
    block = cfg.get("prune_qat") or cfg.get("qat") or cfg.get("prune") or {}
    cc = cfg.get("curriculum") or {}
    return (
        int(block.get("score_batch_size", cc.get("score_batch_size", 32))),
        int(block.get("score_num_workers", cc.get("score_num_workers", 16))),
        int(block.get("score_max_new_tokens", cc.get("score_max_new_tokens", 128))),
    )


def evaluate_languages(model, processor, langs: dict, cfg: dict) -> dict:
    bs, nw, mx = score_knobs(cfg)
    out = {}
    model.eval()
    for name, lang in langs.items():
        print(f"  Evaluating {name} ({len(lang['eval'])} clips)...", flush=True)
        refs, hyps = curriculum.transcribe_dataset(
            model, processor, lang["eval"], lang["token_id"],
            batch_size=bs, num_workers=nw, max_new_tokens=mx,
        )
        ev = ASREvaluator()
        ev.compute_wer(refs, hyps)
        ev.compute_cer(refs, hyps)
        row = ev.to_dict()
        row["n"] = int(len(lang["eval"]))
        out[name] = row
        print(
            f"  {name}: WER {ev.wer:.4f}  CER {ev.cer:.4f}  "
            f"combined_error {row['combined_error']:.4f}",
            flush=True,
        )
    return out


def short_finetune(model, processor, langs: dict, cfg: dict, output_dir: str) -> None:
    train_ds = whisper_train.combine(
        [l["train"] for l in langs.values()],
        [l["oversample"] for l in langs.values()],
    )
    eval_ds = concatenate_datasets([l["eval"] for l in langs.values()])
    wer_samples = whisper_train.pick_wer_samples(langs, n_per_lang=1)

    ft_cfg = dict(cfg)
    training = dict(cfg.get("training") or {})
    training.update((cfg.get("prune_qat") or cfg.get("qat") or {}).get("finetune") or {})
    training.setdefault("num_train_epochs", 1)
    training.setdefault("eval_steps", 500)
    training.setdefault("save_steps", 500)
    training.setdefault("logging_steps", 50)
    training.setdefault("report_to", "none")
    training["save_total_limit"] = max(2, int(training.get("save_total_limit") or 2))
    training["early_stopping_patience"] = None
    ft_cfg["training"] = training

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer = whisper_train.build_trainer(
        model, processor, train_ds, eval_ds, ft_cfg, output_dir,
        wer_samples=wer_samples,
    )
    ckpt = whisper_train.latest_valid_checkpoint(output_dir)
    if ckpt:
        print(f"  Resuming QAT finetune from {ckpt}", flush=True)
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"  Saved QAT-finetuned weights → {output_dir}/final", flush=True)


def finetune_weights_dir(ft_dir: Path) -> Path | None:
    final = ft_dir / "final"
    if (final / "config.json").is_file() and (
        (final / "model.safetensors").is_file() or (final / "pytorch_model.bin").is_file()
    ):
        return final
    ckpt = whisper_train.latest_valid_checkpoint(str(ft_dir))
    if ckpt is None:
        return None
    try:
        state = json.loads((Path(ckpt) / "trainer_state.json").read_text())
        if float(state.get("epoch") or 0) >= 0.999:
            return Path(ckpt)
    except Exception:
        pass
    return None


def resolve_model_id(model_path: str) -> str:
    model_id = Path(model_path).name
    if model_id in ("final", "checkpoint"):
        model_id = Path(model_path).parent.name
    return model_id


def seed_from_prune_results(existing: dict, seed_path: Path, model_id: str) -> dict:
    """
    Copy baseline + prune metrics from a prior prune results.json into mate keys
    (ffn_10 → 10percent, etc.). Does not overwrite non-empty quantization blocks.
    """
    seed = load_json(seed_path)
    src = seed.get(model_id) or {}
    dst = existing.setdefault(model_id, {})

    if "baseline" not in dst and "baseline" in src:
        dst["baseline"] = src["baseline"]
        print(f"  Seeded baseline from {seed_path}", flush=True)

    src_pruning = src.get("pruning") or {}
    dst_pruning = dst.setdefault("pruning", {})
    for key, block in src_pruning.items():
        # accept both ffn_10 and 10percent
        if key in FFN_TO_PERCENT:
            pkey = percent_key(FFN_TO_PERCENT[key])
        elif key.endswith("percent"):
            pkey = key
        else:
            continue
        dst_block = dst_pruning.setdefault(pkey, {})
        for field in ("params", "macs", "size", "languages", "meta"):
            if field in block and field not in dst_block:
                dst_block[field] = block[field]
        if "finetuned" in block and "finetuned" not in dst_block:
            dst_block["finetuned"] = block["finetuned"]
        dst_block.setdefault("quantization", {})
        print(f"  Seeded prune block {pkey} from {key}", flush=True)
    return existing


def find_pruned_checkpoint(prune_root: Path, model_id: str, percent: int) -> Path:
    ffn = PERCENT_TO_FFN[int(percent)]
    candidates = [
        prune_root / model_id / ffn / "final",
        prune_root / model_id / ffn / "finetune" / "final",
        prune_root / ffn / "final",
    ]
    for c in candidates:
        if (c / "config.json").is_file():
            return c
    raise FileNotFoundError(
        f"No pruned checkpoint for {percent}% under {prune_root} "
        f"(tried {[str(c) for c in candidates]})"
    )


def run_grid(
    baseline_path: str,
    cfg: dict,
    percents: list[int],
    schemes: list[str],
    prune_root: Path,
    output_dir: Path,
    skip_finetune: bool,
    existing: dict,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = resolve_model_id(baseline_path)
    # Prefer prune-run folder name if finals live under curriculum id
    print(f"\n=== prune×QAT grid  model_id={model_id} ===", flush=True)

    entry = existing.setdefault(model_id, {})
    langs = whisper_train.build_language_datasets(cfg)

    # Baseline (mate: ces=null)
    if "baseline" not in entry:
        print("\n-- baseline --", flush=True)
        model, processor = load_whisper(baseline_path, device)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        for v in lang_results.values():
            v["ces"] = None
        entry["baseline"] = {
            "model_path": baseline_path,
            "params": count_parameters(model),
            "macs": estimate_macs_whisper(model),
            "size": measure_model_size_bytes(model),
            "languages": lang_results,
        }
        save_json(existing, output_dir / "results.json")
        del model
        torch.cuda.empty_cache()
    else:
        print("Skipping baseline (already in results.json)", flush=True)

    baseline = entry["baseline"]
    baseline_size = baseline.get("size")
    baseline_langs = baseline.get("languages") or {}

    pruning = entry.setdefault("pruning", {})

    for percent in percents:
        pkey = percent_key(percent)
        pruned_path = find_pruned_checkpoint(prune_root, model_id, percent)
        print(f"\n==== {pkey}  checkpoint={pruned_path} ====", flush=True)
        p_entry = pruning.setdefault(pkey, {})
        p_entry.setdefault("quantization", {})

        # Ensure prune finetuned metrics exist (optional quick size snapshot from ckpt)
        if "finetuned" not in p_entry and (pruned_path / "config.json").is_file():
            print(f"  Measuring pruned+FT checkpoint stats for {pkey}...", flush=True)
            model, processor = load_whisper(str(pruned_path), device)
            # If languages missing entirely, eval once (expensive but mate-complete)
            if "languages" not in p_entry and "languages" not in (p_entry.get("finetuned") or {}):
                print("  Evaluating pruned+FT model (seed metrics missing)...", flush=True)
                ft_langs = evaluate_languages(model, processor, langs, cfg)
                ft_size = measure_model_size_bytes(model)
                attach_ces_from_size(ft_langs, baseline_langs, baseline_size, ft_size)
                p_entry["finetuned"] = {
                    "params": count_parameters(model),
                    "macs": estimate_macs_whisper(model),
                    "size": ft_size,
                    "languages": ft_langs,
                }
                p_entry["size"] = ft_size
            else:
                p_entry.setdefault("finetuned", {})
                p_entry["finetuned"].setdefault("params", count_parameters(model))
                p_entry["finetuned"].setdefault("size", measure_model_size_bytes(model))
                p_entry["finetuned"].setdefault("macs", estimate_macs_whisper(model))
            save_json(existing, output_dir / "results.json")
            del model
            torch.cuda.empty_cache()

        for scheme in schemes:
            print(f"\n-- {pkey} × {scheme} --", flush=True)
            q_entry = p_entry["quantization"].setdefault(scheme, {})
            if "finetuned" in q_entry and not skip_finetune:
                print(f"  already done — skip", flush=True)
                continue

            def fresh_pruned():
                m, _ = load_whisper(str(pruned_path), device)
                return m

            ft_dir = output_dir / model_id / pkey / scheme / "finetune"
            reused = finetune_weights_dir(ft_dir)

            if reused is not None and not skip_finetune:
                print(f"  Reusing QAT finetune from {reused}", flush=True)
                model, processor = load_whisper(str(reused), device)
            else:
                model, processor = load_whisper(str(pruned_path), device)
                prepare_qat(model, scheme)
                print("  QAT prepare done.", flush=True)

                if "languages" not in q_entry:
                    print("  Evaluating QAT-prepared (fake quant)...", flush=True)
                    prep_langs = evaluate_languages(model, processor, langs, cfg)
                    prep_size = measure_model_size_bytes(model)
                    attach_ces_from_size(prep_langs, baseline_langs, baseline_size, prep_size)
                    q_entry["languages"] = prep_langs
                    q_entry["size"] = prep_size
                    save_json(existing, output_dir / "results.json")

                if skip_finetune:
                    del model
                    torch.cuda.empty_cache()
                    continue

                print("  Starting QAT short finetune...", flush=True)
                short_finetune(model, processor, langs, cfg, str(ft_dir))

            print("  Persisting real weight-only quantization...", flush=True)
            meta = get_qat_scheme(scheme)
            if meta.get("base_desc"):
                print(f"  PTQ backend: {meta['base_desc']}", flush=True)
            # IMPORTANT: fresh model must be pruned architecture, not baseline
            q_model = persist_after_finetune(model, fresh_pruned, scheme)
            q_model.to(device)
            del model
            torch.cuda.empty_cache()

            print("  Evaluating persisted quantized model...", flush=True)
            lang_results = evaluate_languages(q_model, processor, langs, cfg)
            q_size = measure_model_size_bytes(q_model)
            q_params = count_parameters(q_model)
            q_macs = estimate_macs_whisper(q_model)
            attach_ces_from_size(lang_results, baseline_langs, baseline_size, q_size)

            q_entry["size"] = q_size
            q_entry["finetuned"] = {
                "params": q_params,
                "macs": q_macs,
                "size": q_size,
                "size_ratio_vs_baseline": (q_size / baseline_size) if baseline_size else None,
                "languages": lang_results,
            }

            save_dir = output_dir / model_id / pkey / scheme / "quantized"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(q_model.state_dict(), save_dir / "quantized_state_dict.pt")
            processor.save_pretrained(save_dir)
            (save_dir / "SOURCE_PRUNED.txt").write_text(str(pruned_path) + "\n")
            print(f"  Saved → {save_dir}", flush=True)

            save_json(existing, output_dir / "results.json")
            del q_model
            torch.cuda.empty_cache()

    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_path", default=None,
        help="Unpruned baseline HF Whisper dir (for baseline metrics / model_id)",
    )
    parser.add_argument(
        "--prune_root", default=None,
        help="Root with {model_id}/ffn_{10,20,50}/final pruned+FT checkpoints",
    )
    parser.add_argument(
        "--seed_results", default=None,
        help="Optional prior prune results.json to copy baseline/pruning metrics",
    )
    parser.add_argument(
        "--percent", action="append", dest="percents", type=int, default=None,
        help="Prune percents (repeatable). Default: 10 20 50",
    )
    parser.add_argument(
        "--quant", action="append", dest="schemes", default=None,
        help="QAT schemes (repeatable). Default: int4 int6 int8",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_finetune", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    pq = cfg.get("prune_qat") or {}

    baseline_path = args.model_path or cfg.get("checkpoint")
    if not baseline_path:
        raise SystemExit("Provide --model_path or checkpoint: in YAML")
    prune_root = Path(args.prune_root or pq.get("prune_root") or "")
    if not prune_root.is_dir():
        raise SystemExit(f"prune_root not found: {prune_root}")

    percents = args.percents or pq.get("percents") or [10, 20, 50]
    schemes = args.schemes or pq.get("schemes") or [
        "int4_weight_qat", "int6_weight_qat", "int8_weight_qat",
    ]
    for s in schemes:
        if s not in list_schemes() and s not in (
            "int4_weight_qat", "int6_weight_qat", "int8_weight_qat", "float8_weight_qat",
        ):
            raise SystemExit(f"Unknown scheme {s}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results = load_json(results_path)

    model_id = resolve_model_id(baseline_path)
    seed = args.seed_results or pq.get("seed_results")
    if seed:
        results = seed_from_prune_results(results, Path(seed), model_id)
        save_json(results, results_path)

    if not Path(baseline_path).exists():
        raise FileNotFoundError(f"model_path not found: {baseline_path}")

    results = run_grid(
        baseline_path=baseline_path,
        cfg=cfg,
        percents=[int(p) for p in percents],
        schemes=list(schemes),
        prune_root=prune_root,
        output_dir=output_dir,
        skip_finetune=args.skip_finetune,
        existing=results,
    )
    save_json(results, results_path)
    print(f"\nDone. Results → {results_path}", flush=True)


if __name__ == "__main__":
    main()
