import argparse
import json
from pathlib import Path
import sys
import copy
import tempfile
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from torchao.quantization.qat import (
    Float8FakeQuantizeConfig,
    IntxFakeQuantizeConfig,
    QATConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
from compression.quantization import quantize_model
from data_cleaning.src.config import LANGUAGES
from evaluation import ASREvaluator  # the class from the previous turn

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


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


# ---------- model stats ----------

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


class _KwargsForwardWrapper(nn.Module):
    """
    thop calls the profiled module positionally (model(*inputs)), but
    NeMo's forward() is decorated with @typecheck() and requires
    input_signal=/input_signal_length= as keywords. This thin wrapper
    accepts positional args from thop and forwards them as kwargs to the
    real model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_signal, input_signal_length):
        return self.model.forward(input_signal=input_signal, input_signal_length=input_signal_length)


def _remove_thop_hooks(model) -> None:
    """
    thop normally removes its own forward hooks once profiling completes,
    but a failed/interrupted profiling pass can leave them attached --
    silently corrupting every later forward() call on this model (as just
    happened: a failed profile broke real evaluation afterward). Strip
    anything thop may have attached, unconditionally, success or failure.
    """
    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(module, attr):
                delattr(module, attr)
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def estimate_macs(model, sample_batch):
    """
    Best-effort MACs estimate via thop. Returns None if thop isn't
    installed or profiling fails for any other reason. Always strips
    thop's hooks afterward (success or failure) so this can never leave
    the model in a broken state for subsequent forward() calls.
    """
    if thop_profile is None:
        print("thop not installed -- skipping MACs estimation.")
        return None

    wrapped = _KwargsForwardWrapper(model)
    try:
        signal, signal_len, _, _ = sample_batch
        device = next(model.parameters()).device
        signal, signal_len = signal.to(device), signal_len.to(device)
        macs, _ = thop_profile(wrapped, inputs=(signal, signal_len), verbose=False)
        return macs
    except Exception as exc:
        print(f"MACs estimation failed: {exc}")
        return None
    finally:
        _remove_thop_hooks(model)


def measure_model_size_bytes(model) -> int:
    """Serialize model weights and return on-disk size in bytes."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        return Path(tmp_path).stat().st_size
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

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


import lightning.pytorch as pl

class GradientCheckCallback(pl.Callback):
    def __init__(self, check_every_n_steps=10, param_name_filter="linear1"):
        self.check_every_n_steps = check_every_n_steps
        self.param_name_filter = param_name_filter

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.check_every_n_steps != 0:
            return
        found = False
        for name, p in pl_module.named_parameters():
            if not p.requires_grad or self.param_name_filter not in name:
                continue
            found = True
            grad_norm = p.grad.norm().item() if p.grad is not None else None
            print(f"step {trainer.global_step}: {name} grad_norm={grad_norm}")
        if not found:
            print(f"step {trainer.global_step}: no parameters matched filter '{self.param_name_filter}'")

def evaluate_model(
    model_path: str,
    model_class,
    cfg,
    baseline_entry: dict | None,
    language_codes: dict[str, str],
    quantize: bool = False,
    finetune: bool = False,
) -> dict:
    """
    Loads one model, computes params/MACs once (model-level, language-
    independent), then for each requested language: rebuilds the
    validation dataloader from that language's manifest, runs inference,
    and computes WER/CER (+ CES if a usable baseline is available for
    that language).
    """
    def evaluate_languages_for_model(model, device):
        language_items_local = list(language_codes.items())
        _, first_lang_code_local = language_items_local[0]
        setup_validation_for_language(model, cfg, first_lang_code_local)

        per_language = {}
        for i, (lang_name, lang_code) in enumerate(language_items_local):
            print(f"  -- language: {lang_name} ({lang_code}) --")
            if i > 0:
                setup_validation_for_language(model, cfg, lang_code)

            references, hypotheses = run_model_inference(model, model._validation_dl, device)
            references, hypotheses = clean_references_hypotheses(references, hypotheses)

            evaluator = ASREvaluator()
            evaluator.compute_wer(references, hypotheses)
            evaluator.compute_cer(references, hypotheses)
            per_language[lang_name] = evaluator.__to_dict__()

        return per_language

    finetune_cfg = cfg.get("finetune", {})
    finetune_epochs = int(finetune_cfg.get("epoch", finetune_cfg.get("epochs", 1)))
    finetune_lr = finetune_cfg.get("lr")

    def finetune_quantized_model(q_model):
        model_utils.setup_model(q_model, cfg, change_vocab=False)
        if finetune_lr is not None:
            optim_cfg = copy.deepcopy(q_model.cfg.optim)
            with open_dict(optim_cfg):
                optim_cfg.lr = finetune_lr
            q_model.setup_optimization(optim_config=optim_cfg)

        ft_trainer = model_utils.create_trainer(cfg)
        ft_trainer.callbacks.append(GradientCheckCallback())
        ft_trainer.fit_loop.max_epochs = finetune_epochs
        ft_trainer.fit(q_model)

    model = model_class.restore_from(model_path)
    model_utils.setup_model_for_validation(model, cfg)

    trainer = model_utils.create_trainer(cfg)
    device = trainer.strategy.root_device if trainer.strategy else torch.device("cpu")
    model.to(device)

    params = count_parameters(model)

    language_items = list(language_codes.items())
    _, first_lang_code = language_items[0]
    setup_validation_for_language(model, cfg, first_lang_code)
    sample_batch = next(iter(model._validation_dl))
    macs = estimate_macs(model, sample_batch)

    if not quantize:
        baseline_languages = (baseline_entry or {}).get("languages", {})
        baseline_params = (baseline_entry or {}).get("params")
        baseline_macs = (baseline_entry or {}).get("macs")

        results = {"params": params, "macs": macs, "languages": evaluate_languages_for_model(model, device)}

        for lang_name, lang_code in language_items:
            lang_baseline = baseline_languages.get(lang_name) or baseline_languages.get(lang_code)
            if lang_baseline is not None and baseline_params is not None and baseline_macs is not None and macs is not None:
                evaluator = ASREvaluator()
                evaluator.cer = results["languages"][lang_name]["cer"]
                results["languages"][lang_name]["ces"] = evaluator.compute_ces(
                    params_baseline=baseline_params,
                    params_pruned=params,
                    macs_baseline=baseline_macs,
                    macs_pruned=macs,
                    cer_baseline=lang_baseline["cer"],
                )
            else:
                print(f"  Missing baseline (params/macs/cer) for language '{lang_name}' -- skipping CES.")

        return results

    quantization_configs = {
        "float8_activation_float8_weight_qat": QATConfig(
            weight_config=Float8FakeQuantizeConfig(),
            step="prepare",
        ),
        "int8_activation_int8_weight_qat": QATConfig(
            weight_config=IntxFakeQuantizeConfig(
                torch.int8,
                "per_channel",
                is_symmetric=True,
            ),
            step="prepare",
        ),
    }

    baseline_size = measure_model_size_bytes(model)
    baseline_language_results = evaluate_languages_for_model(model, device)

    quantization_results = {}
    for q_name, q_cfg in quantization_configs.items():
        print(f"  == quantization: {q_name} ==")
        q_model = model_class.restore_from(model_path, map_location="cpu")
        model_utils.setup_model_for_validation(q_model, cfg)

        try:
            quantize_model(q_model, config=q_cfg)
            q_model.to(device)
            q_size = measure_model_size_bytes(q_model)
            q_lang_results = evaluate_languages_for_model(q_model, device)

            for lang_name in q_lang_results:
                baseline_lang = baseline_language_results.get(lang_name)
                if baseline_lang is None:
                    continue
                evaluator = ASREvaluator()
                evaluator.cer = q_lang_results[lang_name]["cer"]
                q_lang_results[lang_name]["ces"] = evaluator.compute_ces_from_size(
                    size_baseline=baseline_size,
                    size_quantized=q_size,
                    cer_baseline=baseline_lang["cer"],
                )

            quantization_results[q_name] = {
                "size": q_size,
                "languages": q_lang_results,
            }

            if finetune:
                finetune_quantized_model(q_model)
                q_ft_lang_results = evaluate_languages_for_model(q_model, device)
                for lang_name in q_ft_lang_results:
                    baseline_lang = baseline_language_results.get(lang_name)
                    if baseline_lang is None:
                        continue
                    evaluator = ASREvaluator()
                    evaluator.cer = q_ft_lang_results[lang_name]["cer"]
                    q_ft_lang_results[lang_name]["ces"] = evaluator.compute_ces_from_size(
                        size_baseline=baseline_size,
                        size_quantized=q_size,
                        cer_baseline=baseline_lang["cer"],
                    )
                quantization_results[q_name]["finetuned"] = {
                    "languages": q_ft_lang_results,
                }
        except Exception as exc:
            print(f"  Quantization failed for '{q_name}': {exc}")
            raise exc
            #quantization_results[q_name] = {"error": str(exc)}

    return {
        "baseline": {
            "params": params,
            "macs": macs,
            "size": baseline_size,
            "languages": baseline_language_results,
        },
        "quantization": quantization_results,
    }

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
    parser.add_argument("--quantize", action="store_true", help="Apply QAT prepare-time fake quantization before evaluation.")
    parser.add_argument("--finetune", action="store_true", help="After QAT prepare, fine-tune each model using cfg.finetune before re-evaluation.")
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
            quantize=args.quantize,
            finetune=args.finetune,
        )

    output_path = str(Path(args.output_dir) / "results.json")
    save_json_file(results, output_path)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()