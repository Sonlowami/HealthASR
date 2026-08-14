"""
Structural FFN pruning + short recovery finetune for HF Whisper.

Ratios (mate-style): 10% / 20% / 50% of FFN intermediate width
  --ratio 0.1 --ratio 0.2 --ratio 0.5

Flow:
  1) baseline WER/CER/size/params/MACs
  2) prune FFN → eval
  3) short Seq2SeqTrainer recovery FT
  4) re-eval + CES → results.json
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
    attach_ces_from_size,
    count_parameters,
    estimate_macs_whisper,
    measure_model_size_bytes,
)
from compression.prune import (  # noqa: E402
    list_default_ratios,
    prune_by_ratio,
    ratio_tag,
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


def score_cfg(cfg: dict) -> tuple[int, int, int]:
    block = cfg.get("prune") or cfg.get("qat") or {}
    cc = cfg.get("curriculum") or {}
    bs = int(block.get("score_batch_size", cc.get("score_batch_size", 32)))
    nw = int(block.get("score_num_workers", cc.get("score_num_workers", 16)))
    mx = int(block.get("score_max_new_tokens", cc.get("score_max_new_tokens", 128)))
    return bs, nw, mx


def evaluate_languages(model, processor, langs: dict, cfg: dict) -> dict:
    from compression.eval_metrics import ASREvaluator

    score_bs, num_workers, max_new = score_cfg(cfg)
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
    training.update((cfg.get("prune") or {}).get("finetune") or {})
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
        print(f"  Resuming recovery finetune from {ckpt}", flush=True)
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"  Saved pruned+finetuned weights → {output_dir}/final", flush=True)


def run_one_model(
    model_path: str,
    cfg: dict,
    ratios: list[float],
    output_dir: Path,
    scope: str,
    skip_baseline: bool,
    skip_finetune: bool,
    existing: dict,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = Path(model_path).name
    if model_id in ("final", "checkpoint"):
        model_id = Path(model_path).parent.name
    print(f"\n=== Model: {model_id}  path={model_path} ===", flush=True)

    entry = existing.setdefault(model_id, {})
    langs = whisper_train.build_language_datasets(cfg)

    if not skip_baseline and "baseline" not in entry:
        print("\n-- baseline --", flush=True)
        model, processor = load_whisper(model_path, device)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        for v in lang_results.values():
            v["ces"] = None
        entry["baseline"] = {
            "model_path": model_path,
            "params": count_parameters(model),
            "macs": estimate_macs_whisper(model),
            "size": measure_model_size_bytes(model),
            "languages": lang_results,
        }
        save_json(existing, output_dir / "results.json")
        del model
        torch.cuda.empty_cache()
    elif "baseline" in entry:
        print("Skipping baseline (already in results.json)", flush=True)

    baseline = entry.get("baseline") or {}
    baseline_size = baseline.get("size")
    baseline_langs = baseline.get("languages") or {}

    prune_results = entry.setdefault("pruning", {})
    for ratio in ratios:
        tag = ratio_tag(ratio)
        print(f"\n-- prune {tag} (ratio={ratio}, scope={scope}) --", flush=True)
        p_entry = prune_results.setdefault(tag, {})
        if "finetuned" in p_entry and not skip_finetune:
            print(f"  {tag}: finetuned already in results — skip", flush=True)
            continue

        model, processor = load_whisper(model_path, device)
        model, meta = prune_by_ratio(model, ratio=ratio, scope=scope, verbose=True)
        p_entry["meta"] = meta

        # Post-prune (no FT yet)
        if "languages" not in p_entry:
            print("  Evaluating pruned model (before recovery FT)...", flush=True)
            prep_langs = evaluate_languages(model, processor, langs, cfg)
            prep_size = measure_model_size_bytes(model)
            attach_ces_from_size(prep_langs, baseline_langs, baseline_size, prep_size)
            p_entry["languages"] = prep_langs
            p_entry["size"] = prep_size
            p_entry["params"] = count_parameters(model)
            p_entry["macs"] = estimate_macs_whisper(model)
            save_json(existing, output_dir / "results.json")

        # Save pruned-only weights
        pruned_dir = output_dir / model_id / tag / "pruned"
        pruned_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(pruned_dir)
        processor.save_pretrained(pruned_dir)
        (pruned_dir / "PRUNE_META.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  Saved pruned model → {pruned_dir}", flush=True)

        if skip_finetune:
            del model
            torch.cuda.empty_cache()
            continue

        print("  Starting recovery finetune...", flush=True)
        ft_dir = str(output_dir / model_id / tag / "finetune")
        short_finetune(model, processor, langs, cfg, ft_dir)

        print("  Evaluating after recovery FT...", flush=True)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        q_size = measure_model_size_bytes(model)
        q_params = count_parameters(model)
        q_macs = estimate_macs_whisper(model)
        attach_ces_from_size(lang_results, baseline_langs, baseline_size, q_size)

        p_entry["size"] = q_size
        p_entry["finetuned"] = {
            "params": q_params,
            "macs": q_macs,
            "size": q_size,
            "size_ratio_vs_baseline": (q_size / baseline_size) if baseline_size else None,
            "languages": lang_results,
            "meta": meta,
        }

        final_dir = output_dir / model_id / tag / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"  Saved pruned+FT model → {final_dir}", flush=True)

        save_json(existing, output_dir / "results.json")
        del model
        torch.cuda.empty_cache()

    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", action="append", dest="model_paths", default=None)
    parser.add_argument(
        "--ratio", action="append", dest="ratios", type=float, default=None,
        help="FFN prune ratio (repeatable). Default: 0.1 0.2 0.5",
    )
    parser.add_argument(
        "--scope", choices=("encoder", "decoder", "both"), default=None,
        help="Which FFN stacks to prune (default: config prune.scope or encoder)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument(
        "--skip_finetune", action="store_true",
        help="Prune + eval only (no recovery FT)",
    )
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

    prune_cfg = cfg.get("prune") or {}
    ratios = args.ratios or prune_cfg.get("ratios") or list_default_ratios()
    scope = args.scope or prune_cfg.get("scope") or "encoder"

    results_path = output_dir / "results.json"
    results = load_json(results_path)

    for mp in model_paths:
        if not Path(mp).exists():
            raise FileNotFoundError(f"model_path not found: {mp}")
        results = run_one_model(
            mp, cfg, [float(r) for r in ratios], output_dir,
            scope=scope,
            skip_baseline=args.skip_baseline,
            skip_finetune=args.skip_finetune,
            existing=results,
        )
        save_json(results, results_path)

    print(f"\nDone. Results → {results_path}", flush=True)


if __name__ == "__main__":
    main()
