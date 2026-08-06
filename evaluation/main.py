import argparse
import json
from pathlib import Path
import sys
import copy
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
import tempfile
import nncf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
from data_cleaning.src.config import LANGUAGES
from evaluation import ASREvaluator
from compression.quantize import quantize_model


# ---------- I/O (separated for easy swap-in of existing project utilities) ----------

def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_references_hypotheses(references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
    """
    Clean up references and hypotheses for evaluation.
    This can includes stripping and removing any '?' characters.
    """
    cleaned_references = [ref.strip().replace("?", "") for ref in references]
    cleaned_hypotheses = [hyp.strip().replace("?", "") for hyp in hypotheses]
    return cleaned_references, cleaned_hypotheses

# ---------- inference ----------

def run_model_inference(model, val_loader, device) -> tuple[list[str], list[str]]:
    """Runs the model over its validation dataloader; returns (references, hypotheses)."""
    model.to(device)
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for batch in val_loader:
            signal, signal_len, tokens, token_len = batch
            signal, signal_len = signal.to(device), signal_len.to(device)

            output, output_len = model_utils.run_model_forward(model, signal, signal_len)
            hyps = model_utils.get_hypotheses(model, output, output_len)

            tokens_np, token_len_np = tokens.cpu().numpy(), token_len.cpu().numpy()
            for t, t_len, hyp in zip(tokens_np, token_len_np, hyps):
                references.append(model.tokenizer.ids_to_text(t[:t_len].tolist()))
                hypotheses.append(" ".join(hyp.words) if hasattr(hyp, "words") else str(hyp))
    model.train()
    return references, hypotheses


# ---------- per-model evaluation ----------

def resolve_language_codes(requested_languages: list[str]) -> dict[str, str]:
    """
    Map user-supplied --languages values (names or codes) to {name: code},
    using LANGUAGES' actual shape: {name: {"code": ..., "dir": ...}}.
    Raises clearly if a requested language isn't found by either name or code.
    """
    name_to_code = {name: meta["code"] for name, meta in LANGUAGES.items()}
    code_to_name = {code: name for name, code in name_to_code.items()}

    resolved = {}
    for lang in requested_languages:
        if lang in name_to_code:
            resolved[lang] = name_to_code[lang]
        elif lang in code_to_name:
            resolved[code_to_name[lang]] = lang
        else:
            raise ValueError(f"Unknown language '{lang}' -- not found in LANGUAGES (name or code).")
    return resolved

def setup_validation_for_language(model, cfg, language_code: str) -> None:
    """
    Rebuild the model's validation dataloader to point at the
    language-specific manifest ({code}_manifest_filepath under
    cfg.model.validation_ds), reusing the model's already-configured
    validation_ds template (batch_size, max_duration, etc.) and only
    swapping manifest_filepath.
    """
    manifest_key = f"{language_code}_manifest_filepath"
    manifest_path = cfg["model"]["validation_ds"].get(manifest_key)
    if not manifest_path:
        raise KeyError(
            f"No '{manifest_key}' found under model.validation_ds in config -- "
            f"cannot evaluate language code '{language_code}'."
        )

    lang_ds_cfg = copy.deepcopy(model.cfg.validation_ds)
    with open_dict(lang_ds_cfg):
        lang_ds_cfg.manifest_filepath = manifest_path
    model.setup_validation_data(lang_ds_cfg)

def evaluate_model(
        model_path: str,
        model_class,
        cfg,
        baseline_entry: dict | None,
        language_codes: dict[str, str],
        quantize: bool = False
        ) -> dict:
    model = model_class.restore_from(model_path)
    model_utils.setup_model_for_validation(model, cfg)

    trainer = model_utils.create_trainer(cfg)
    device = trainer.strategy.root_device if trainer.strategy else torch.device("cpu")
    model.to(device)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        baseline_size = model_utils.model_size_on_disk_bytes(model, tmp.name)

    if quantize:
        model = nncf.strip(quantize_model(model))  # strip so size reflects real reduced-precision storage
        setup_validation_for_language(model, cfg, next(iter(language_codes.values())))
    # (a future prune_model(model) step, if/when added, would slot in right here --
    #  size is measured AFTER whatever compression steps ran, on the same model object)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        compressed_size = model_utils.model_size_on_disk_bytes(model, tmp.name)

    print(f" Baseline size: {baseline_size / 1e6:.1f} MB, compressed size: {compressed_size / 1e6:.1f} MB "
          f"({100 * (1 - compressed_size / baseline_size):.1f}% reduction)")

    results = {
        "baseline_size(MB)": baseline_size / 1e6,
        "compressed_size(MB)": compressed_size / 1e6,
        "languages": {}
        }
    language_items = list(language_codes.items())
    baseline_languages = (baseline_entry or {}).get("languages", {})
    for i, (lang_name, lang_code) in enumerate(language_items):
        print(f" -- language: {lang_name} ({lang_code}) --")
        if i > 0:
            # first language's dataloader is already set up above --
            # only rebuild for languages 2+
            setup_validation_for_language(model, cfg, lang_code)

        references, hypotheses = run_model_inference(model, model._validation_dl, device)
        references, hypotheses = clean_references_hypotheses(references, hypotheses)

        evaluator = ASREvaluator()
        evaluator.compute_wer(references, hypotheses)
        evaluator.compute_cer(references, hypotheses)

        lang_baseline = baseline_languages.get(lang_name) or baseline_languages.get(lang_code)
        if lang_baseline is not None and "baseline_size(MB)" in baseline_entry:
            evaluator.compute_ces(baseline_entry["baseline_size(MB)"], compressed_size, cer_baseline=lang_baseline["cer"])
        else:
            print(f"  Missing baseline size/cer for '{lang_name}' -- skipping CES.")

        results["languages"][lang_name] = evaluator.__to_dict__()

    return results
# ---------- entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more ASR models (WER/CER/CES) across one or more languages.")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to .nemo model checkpoints.")
    parser.add_argument("--model_class", required=True, help="Dotted path to the model class.")
    parser.add_argument("--config", required=True, help="Path to the NeMo config (for validation dataset setup).")
    parser.add_argument("--languages", nargs="+", required=True,
                         help="Language names or codes to evaluate against (must match LANGUAGES and "
                              "have a corresponding {code}_manifest_filepath in config.model.validation_ds).")
    parser.add_argument("--baseline_file", default=None,
                         help="Optional JSON: {model_filename: {params, macs, languages: {lang: {cer, wer, combined_error}}}}.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.json into.")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model before evaluation.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model_class = model_utils.resolve_model_class(args.model_class)
    baseline_data = load_json_file(args.baseline_file) if args.baseline_file else {}
    language_codes = resolve_language_codes(args.languages)

    results = {}
    for model_path in args.model_paths:
        model_filename = Path(model_path).name
        print(f"\n=== Evaluating {model_filename} ===")
        baseline_entry = baseline_data.get(model_filename) if args.baseline_file else None
        results[model_filename] = evaluate_model(
            model_path,
            model_class,
            cfg,
            baseline_entry,
            language_codes,
            quantize=args.quantize
            )

    output_path = str(Path(args.output_dir) / "results.json")
    save_json_file(results, output_path)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()