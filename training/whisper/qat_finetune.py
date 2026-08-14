"""
QAT + short finetune for any Hugging Face Whisper checkpoint (torchao).

Model-agnostic: pass --model_path (any final/checkpoint dir). Data/languages
come from a whisper YAML (same shape as training configs).

Flow (matches teammate NeMo compression script):
  1) baseline WER/CER/size/params/MACs
  2) QAT prepare (fake quant on Linear) → eval
  3) short Seq2SeqTrainer finetune
  4) persist → fresh model + real Int8/Int4/Int6/Float8 weight-only PTQ
  5) re-eval + CES → results.json

Schemes: --quant int8_weight_qat | int4_weight_qat | int6_weight_qat | float8_weight_qat
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
    quantize_model,
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


def evaluate_languages(model, processor, langs: dict, cfg: dict) -> dict:
    """Per-language WER/CER/combined_error (ces filled later via attach_ces_from_size)."""
    qat_cfg = cfg.get("qat") or {}
    score_bs = int(qat_cfg.get("score_batch_size", (cfg.get("curriculum") or {}).get("score_batch_size", 32)))
    num_workers = int(qat_cfg.get("score_num_workers", (cfg.get("curriculum") or {}).get("score_num_workers", 16)))
    max_new = int(qat_cfg.get("score_max_new_tokens", (cfg.get("curriculum") or {}).get("score_max_new_tokens", 128)))

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
    training.update(cfg.get("qat", {}).get("finetune") or {})
    training.setdefault("num_train_epochs", 1)
    training.setdefault("eval_steps", 500)
    training.setdefault("save_steps", 500)
    training.setdefault("logging_steps", 50)
    training.setdefault("report_to", "none")
    # Keep mid + end checkpoints so persist failures don't lose FT weights
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
        print(f"  Resuming short finetune from {ckpt}", flush=True)
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()
    # Persist a stable "final" so persist-only restarts don't depend on checkpoint-*
    model.save_pretrained(f"{output_dir}/final")
    processor.save_pretrained(f"{output_dir}/final")
    print(f"  Saved QAT-finetuned weights → {output_dir}/final", flush=True)


def finetune_weights_dir(ft_dir: Path) -> Path | None:
    """Return path to completed finetune weights if present (skip re-training)."""
    final = ft_dir / "final"
    if (final / "config.json").is_file() and (
        (final / "model.safetensors").is_file() or (final / "pytorch_model.bin").is_file()
    ):
        return final
    ckpt = whisper_train.latest_valid_checkpoint(str(ft_dir))
    if ckpt is None:
        return None
    # Prefer checkpoints that finished the planned epoch (trainer_state)
    state_path = Path(ckpt) / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text())
        epoch = float(state.get("epoch") or 0)
        if epoch >= 0.999:
            return Path(ckpt)
    except Exception:
        pass
    return None


def run_one_model(
    model_path: str,
    cfg: dict,
    schemes: list[str],
    output_dir: Path,
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

    def fresh_loader():
        m, _ = load_whisper(model_path, device)
        return m

    # --- baseline ---
    if not skip_baseline and "baseline" not in entry:
        print("\n-- baseline --", flush=True)
        model, processor = load_whisper(model_path, device)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        for v in lang_results.values():
            v["ces"] = None  # baseline CES is null (mate schema)
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

    # --- QAT schemes ---
    quant_results = entry.setdefault("quantization", {})
    for scheme in schemes:
        print(f"\n-- QAT scheme: {scheme} --", flush=True)
        q_entry = quant_results.setdefault(scheme, {})
        if "finetuned" in q_entry and not skip_finetune:
            print(f"  {scheme}: finetuned already in results — skip", flush=True)
            continue

        ft_dir = output_dir / model_id / scheme / "finetune"
        reused = finetune_weights_dir(ft_dir)

        if reused is not None and not skip_finetune:
            # Finetune finished earlier; only persist + eval (avoids redoing ~50min FT)
            print(f"  Reusing completed finetune weights from {reused}", flush=True)
            model, processor = load_whisper(str(reused), device)
        else:
            model, processor = load_whisper(model_path, device)
            prepare_qat(model, scheme)
            print("  QAT prepare done.", flush=True)

            # Mate step: measure under fake-quant before finetune
            if "languages" not in q_entry:
                print("  Evaluating QAT-prepared (fake quant) model...", flush=True)
                prep_langs = evaluate_languages(model, processor, langs, cfg)
                prep_size = measure_model_size_bytes(model)
                attach_ces_from_size(prep_langs, baseline_langs, baseline_size, prep_size)
                q_entry["languages"] = prep_langs
                q_entry["size"] = prep_size  # often ≈ baseline under fake quant
                save_json(existing, output_dir / "results.json")

            if skip_finetune:
                del model
                torch.cuda.empty_cache()
                continue

            print("  Starting short finetune...", flush=True)
            short_finetune(model, processor, langs, cfg, str(ft_dir))

        print("  Persisting real weight-only quantization...", flush=True)
        scheme_meta = get_qat_scheme(scheme)
        if scheme_meta.get("base_desc"):
            print(f"  PTQ backend: {scheme_meta['base_desc']}", flush=True)
        q_model = persist_after_finetune(model, fresh_loader, scheme)
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

        save_dir = output_dir / model_id / scheme / "quantized"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q_model.state_dict(), save_dir / "quantized_state_dict.pt")
        processor.save_pretrained(save_dir)
        (save_dir / "SOURCE_MODEL.txt").write_text(model_path + "\n")
        print(f"  Saved quantized state_dict → {save_dir}", flush=True)

        save_json(existing, output_dir / "results.json")
        del q_model
        torch.cuda.empty_cache()

    return existing


def _lang_missing_cer(languages: dict | None) -> bool:
    if not languages:
        return True
    return any(v.get("cer") is None for v in languages.values())


def _load_quantized_for_scheme(
    model_path: str,
    output_dir: Path,
    model_id: str,
    scheme: str,
    device: torch.device,
):
    """Rebuild quantized model from saved state_dict (or finetune/final + persist)."""
    def fresh_loader():
        m, _ = load_whisper(model_path, device)
        return m

    sd_path = output_dir / model_id / scheme / "quantized" / "quantized_state_dict.pt"
    processor = WhisperProcessor.from_pretrained(model_path)

    if sd_path.is_file():
        fresh = fresh_loader()
        quantize_model(fresh, config=get_qat_scheme(scheme)["base"])
        try:
            state = torch.load(sd_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(sd_path, map_location=device)
        fresh.load_state_dict(state, strict=False)
        fresh.to(device)
        return fresh, processor

    ft_final = output_dir / model_id / scheme / "finetune" / "final"
    if (ft_final / "config.json").is_file():
        model, processor = load_whisper(str(ft_final), device)
        q_model = persist_after_finetune(model, fresh_loader, scheme)
        q_model.to(device)
        del model
        return q_model, processor

    raise FileNotFoundError(
        f"No quantized weights for {scheme} under {output_dir / model_id / scheme}"
    )


def refill_metrics(
    model_path: str,
    cfg: dict,
    output_dir: Path,
    existing: dict,
    force: bool = False,
) -> dict:
    """
    Re-score WER/CER/combined_error and attach CES for baseline + finished schemes.
    Skips decode when CER already present (unless force); still fills CES once baseline CER exists.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = Path(model_path).name
    if model_id in ("final", "checkpoint"):
        model_id = Path(model_path).parent.name
    entry = existing.setdefault(model_id, {})
    langs = whisper_train.build_language_datasets(cfg)

    # --- baseline ---
    baseline = entry.setdefault("baseline", {"model_path": model_path})
    need_base_eval = force or _lang_missing_cer(baseline.get("languages"))
    if need_base_eval:
        print("\n-- refill baseline eval --", flush=True)
        model, processor = load_whisper(model_path, device)
        lang_results = evaluate_languages(model, processor, langs, cfg)
        for v in lang_results.values():
            v["ces"] = None
        baseline["languages"] = lang_results
        baseline["model_path"] = model_path
        baseline["params"] = count_parameters(model)
        baseline["size"] = measure_model_size_bytes(model)
        if baseline.get("macs") in (None, 0):
            baseline["macs"] = estimate_macs_whisper(model)
        del model
        torch.cuda.empty_cache()
        save_json(existing, output_dir / "results.json")
    else:
        print("Skipping baseline decode (CER present)", flush=True)

    baseline_size = baseline.get("size")
    baseline_langs = baseline.get("languages") or {}

    # --- schemes ---
    quant = entry.setdefault("quantization", {})
    for scheme, q_entry in list(quant.items()):
        ft = q_entry.get("finetuned")
        if not ft:
            print(f"  {scheme}: no finetuned — skip", flush=True)
            continue

        need_eval = force or _lang_missing_cer(ft.get("languages"))
        if need_eval:
            print(f"\n-- refill {scheme} eval --", flush=True)
            try:
                q_model, processor = _load_quantized_for_scheme(
                    model_path, output_dir, model_id, scheme, device,
                )
            except Exception as exc:
                print(f"  {scheme}: could not load quantized model ({exc}) — skip", flush=True)
                continue
            lang_results = evaluate_languages(q_model, processor, langs, cfg)
            ft["languages"] = lang_results
            ft["params"] = count_parameters(q_model)
            ft["size"] = measure_model_size_bytes(q_model)
            q_entry["size"] = ft["size"]
            if baseline_size:
                ft["size_ratio_vs_baseline"] = ft["size"] / baseline_size
            if ft.get("macs") in (None, 0):
                ft["macs"] = estimate_macs_whisper(q_model)
            del q_model
            torch.cuda.empty_cache()
        else:
            print(f"  {scheme}: CER present — only refreshing CES", flush=True)

        attach_ces_from_size(
            ft.get("languages") or {},
            baseline_langs,
            baseline_size,
            ft.get("size") or q_entry.get("size"),
        )
        # Also refresh CES on prepare-eval block if present
        if "languages" in q_entry and q_entry["languages"] is not ft.get("languages"):
            attach_ces_from_size(
                q_entry["languages"],
                baseline_langs,
                baseline_size,
                q_entry.get("size") or ft.get("size"),
            )
        save_json(existing, output_dir / "results.json")

    return existing


def fill_missing_macs(
    model_path: str,
    output_dir: Path,
    existing: dict,
    schemes: list[str] | None = None,
) -> dict:
    """Backfill macs in results.json for baseline + finished schemes (needs thop)."""
    try:
        import thop  # noqa: F401
    except ImportError as e:
        raise SystemExit("thop is required for --fill_macs: pip install thop") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = Path(model_path).name
    if model_id in ("final", "checkpoint"):
        model_id = Path(model_path).parent.name
    entry = existing.setdefault(model_id, {})

    def fresh_loader():
        m, _ = load_whisper(model_path, device)
        return m

    # baseline
    baseline = entry.get("baseline")
    if baseline is not None and baseline.get("macs") in (None, 0):
        print(f"  Filling baseline MACs from {model_path}...", flush=True)
        model, _ = load_whisper(model_path, device)
        baseline["macs"] = estimate_macs_whisper(model)
        baseline.setdefault("params", count_parameters(model))
        print(f"  baseline macs={baseline['macs']}", flush=True)
        del model
        torch.cuda.empty_cache()
        save_json(existing, output_dir / "results.json")

    quant = entry.setdefault("quantization", {})
    scheme_names = schemes or list(quant.keys())
    for scheme in scheme_names:
        q_entry = quant.get(scheme)
        if not q_entry or "finetuned" not in q_entry:
            print(f"  {scheme}: no finetuned block — skip", flush=True)
            continue
        ft = q_entry["finetuned"]
        if ft.get("macs") not in (None, 0):
            print(f"  {scheme}: macs already set ({ft['macs']}) — skip", flush=True)
            continue

        print(f"  Filling MACs for {scheme}...", flush=True)
        macs = None
        sd_path = output_dir / model_id / scheme / "quantized" / "quantized_state_dict.pt"
        try:
            if sd_path.is_file():
                fresh = fresh_loader()
                cfg = get_qat_scheme(scheme)
                quantize_model(fresh, config=cfg["base"])
                try:
                    state = torch.load(sd_path, map_location=device, weights_only=True)
                except TypeError:
                    state = torch.load(sd_path, map_location=device)
                fresh.load_state_dict(state, strict=False)
                fresh.to(device)
                macs = estimate_macs_whisper(fresh)
                del fresh
            else:
                ft_final = output_dir / model_id / scheme / "finetune" / "final"
                if (ft_final / "config.json").is_file():
                    model, _ = load_whisper(str(ft_final), device)
                    q_model = persist_after_finetune(model, fresh_loader, scheme)
                    q_model.to(device)
                    macs = estimate_macs_whisper(q_model)
                    del model, q_model
                else:
                    print(f"  {scheme}: no quantized artifacts; using baseline model for MACs", flush=True)
                    model, _ = load_whisper(model_path, device)
                    macs = estimate_macs_whisper(model)
                    del model
        except Exception as exc:
            print(f"  {scheme}: MACs failed ({exc}); falling back to baseline model", flush=True)
            model, _ = load_whisper(model_path, device)
            macs = estimate_macs_whisper(model)
            del model

        ft["macs"] = macs
        print(f"  {scheme}: macs={macs}", flush=True)
        torch.cuda.empty_cache()
        save_json(existing, output_dir / "results.json")

    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Whisper YAML (languages + training/qat knobs)")
    parser.add_argument(
        "--model_path", action="append", dest="model_paths", default=None,
        help="HF Whisper dir (final/checkpoint). Repeatable.",
    )
    parser.add_argument(
        "--quant", action="append", dest="schemes", default=None,
        help="QAT scheme (repeatable): " + " | ".join(list_schemes()),
    )
    parser.add_argument("--output_dir", required=True, help="Where to write results.json + artifacts")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument(
        "--skip_finetune", action="store_true",
        help="Only QAT-prepare + eval (no short FT / persist)",
    )
    parser.add_argument(
        "--fill_macs", action="store_true",
        help="Only backfill missing macs in results.json (needs thop). No QAT training.",
    )
    parser.add_argument(
        "--refill_metrics", action="store_true",
        help="Re-score missing CER/combined_error and attach CES (no QAT training). "
             "Use --force_refill to re-decode even when CER exists.",
    )
    parser.add_argument(
        "--force_refill", action="store_true",
        help="With --refill_metrics: re-decode all languages even if CER is present.",
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

    schemes = args.schemes or (cfg.get("qat") or {}).get("schemes") or ["int8_weight_qat"]

    results_path = output_dir / "results.json"
    results = load_json(results_path)

    if args.fill_macs and args.refill_metrics:
        raise SystemExit("Use either --fill_macs or --refill_metrics, not both")

    if args.fill_macs:
        for mp in model_paths:
            if not Path(mp).exists():
                raise FileNotFoundError(f"model_path not found: {mp}")
            print(f"\n=== fill_macs: {mp} ===", flush=True)
            results = fill_missing_macs(mp, output_dir, results, schemes=None)
            save_json(results, results_path)
        print(f"\nDone (fill_macs). Results → {results_path}", flush=True)
        return

    if args.refill_metrics:
        for mp in model_paths:
            if not Path(mp).exists():
                raise FileNotFoundError(f"model_path not found: {mp}")
            print(f"\n=== refill_metrics: {mp} ===", flush=True)
            results = refill_metrics(
                mp, cfg, output_dir, results, force=args.force_refill,
            )
            save_json(results, results_path)
        print(f"\nDone (refill_metrics). Results → {results_path}", flush=True)
        return

    for mp in model_paths:
        if not Path(mp).exists():
            raise FileNotFoundError(f"model_path not found: {mp}")
        results = run_one_model(
            mp, cfg, schemes, output_dir,
            skip_baseline=args.skip_baseline,
            skip_finetune=args.skip_finetune,
            existing=results,
        )
        save_json(results, results_path)

    print(f"\nDone. Results → {results_path}", flush=True)


if __name__ == "__main__":
    main()
