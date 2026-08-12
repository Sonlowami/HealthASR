import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig
from torchao.quantization.qat import Float8FakeQuantizeConfig, IntxFakeQuantizeConfig, QATConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
from compression.pruning import precompute_prune_dimension, prune_ffns
from compression.quantization import quantize_model
from data_cleaning.src.config import LANGUAGES
from evaluation import ASREvaluator

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    import dill
except ImportError as exc:
    raise ImportError(
        "dill is required for clone_model_via_disk (stdlib pickle can't "
        "serialize NeMo's decorator-wrapped methods, e.g. FilterbankFeatures.forward). "
        "Install with: pip install dill"
    ) from exc


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
import wrapt

def inspect_model_tokenizer(model, label):
    print(f"\n===== {label} =====")

    tok = model.tokenizer

    print("model.tokenizer:", type(tok))
    print("model.tokenizer.vocab_size:",
          repr(tok.vocab_size),
          type(tok.vocab_size))

    print("model.tokenizer.tokenizer:",
          type(tok.tokenizer))

    print("underlying vocab_size:",
          repr(tok.tokenizer.vocab_size),
          type(tok.tokenizer.vocab_size))

    print("underlying vocab_size():")
    try:
        tok.tokenizer.vocab_size()
    except Exception as exc:
        print(f"Error occurred while accessing vocab_size(): {type(exc).__name__}: {exc}")

    print("original_vocab_size:",
          getattr(tok, "original_vocab_size", None))

    print("model.decoding:", type(model.decoding))
    print("model.decoding.decoding:", type(model.decoding.decoding))

    print("decoding.__dict__:")
    for k, v in model.decoding.__dict__.items():
        if "blank" in k.lower() or "token" in k.lower() or "vocab" in k.lower():
            print(" ", k, "=", repr(v), type(v))


def bisect_pickle_failure(obj, path="model", max_depth=8):
    """
    Same bisection technique used earlier for deepcopy: try pickling
    (via dill, matching clone_model_via_disk's actual call) each
    attribute individually, recurse into whichever one fails. Pinpoints
    the exact wrapt-wrapped attribute rather than guessing.
    """
    if max_depth == 0:
        print(f"{path}: max depth reached, stopping")
        return

    d = getattr(obj, "__dict__", None)
    if not isinstance(d, dict):
        print(f"{path}: no __dict__, can't narrow further (leaf-level failure)")
        return

    for key, value in d.items():
        try:
            dill.dumps(value)
        except Exception as exc:
            print(f"FAILS: {path}.{key} = {type(value)} -- {type(exc).__name__}: {exc}")
            bisect_pickle_failure(value, f"{path}.{key}", max_depth - 1)

def is_wrapt_proxy(value) -> bool:
    """
    Duck-types wrapt proxies rather than isinstance(value, wrapt.ObjectProxy),
    which can miss objects built via wrapt's C-accelerated _wrappers
    extension if wrapt.ObjectProxy resolves to a different (pure-Python)
    class object at import time -- confirmed happening in this environment.
    """
    return any(cls.__name__ == "ObjectProxy" for cls in type(value).__mro__)


def strip_wrapt_proxies(model):
    removed = {}

    for attr in ("_validation_dl", "_train_dl"):
        if attr in model.__dict__:
            removed[attr] = model.__dict__.pop(attr)

    for key, value in list(model.__dict__.items()):
        if is_wrapt_proxy(value):
            removed[key] = model.__dict__.pop(key)

    return removed

def clone_model_via_disk(model):
    """
    We cannot do copy.deepcopy(model) because @wrapt decorators do not implement __deepcopy__
    and will raise an error. Instead, we save the model to a temporary file and load a copy of it.
    """
    tokenizer = model.tokenizer
    removed = strip_wrapt_proxies(model)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model, tmp_path, pickle_module=dill)
        cloned_model = torch.load(tmp_path, pickle_module=dill, weights_only=False)
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
        for attr, value in removed.items():
            model.__dict__[attr] = value
    # cloned model overwrites the tokenizer with an instance lacking vocab_size, get_vocab method and all_special_tokens.
    # If we don't restore the tokenizer, we get a crazy error to debug that says "TypeError: unsupported operand type(s) for +: 'method' and 'int'"
    # It is a nasty error to debug as all symptoms point to nemo internals instead of here.
    cloned_model.tokenizer = tokenizer

    return cloned_model

def save_json_file(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_references_hypotheses(references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
    cleaned_references = [ref.strip().replace("?", "") for ref in references]
    cleaned_hypotheses = [hyp.strip().replace("?", "") for hyp in hypotheses]
    return cleaned_references, cleaned_hypotheses


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _remove_thop_hooks(model) -> None:
    for module in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(module, attr):
                delattr(module, attr)
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def estimate_macs(model, sample_batch):
    if thop_profile is None:
        print("thop not installed -- skipping MACs estimation.")
        return None

    wrapped = model_utils._KwargsForwardWrapper(model)
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
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        return Path(tmp_path).stat().st_size
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def run_model_inference(model, val_loader, device) -> tuple[list[str], list[str]]:
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


def resolve_language_codes(requested_languages: list[str]) -> dict[str, str]:
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


def _prepare_model_for_evaluation(model_class, model_path: str, cfg):
    model = model_class.restore_from(model_path, map_location="cpu")
    model_utils.setup_model_for_validation(model, cfg)
    return model


def _attach_ces_from_size(language_results: dict, baseline_language_results: dict, size_baseline: float, size_current: float) -> None:
    for lang_name, lang_result in language_results.items():
        baseline_lang = baseline_language_results.get(lang_name)
        if baseline_lang is None:
            continue

        evaluator = ASREvaluator()
        evaluator.cer = lang_result["cer"]
        lang_result["ces"] = evaluator.compute_ces_from_size(
            size_baseline=size_baseline,
            size_quantized=size_current,
            cer_baseline=baseline_lang["cer"],
        )


def evaluate_model(
    model_path: str,
    model_class,
    cfg,
    baseline_entry: dict | None,
    language_codes: dict[str, str],
    existing_result: dict | None = None,
    save_progress=None,
    prune: bool = False,
    prune_ratios: tuple[float, ...] = (),
    quantize: bool = False,
    finetune: bool = False,
    prune_and_quantize: bool = False,
) -> dict:
    results = existing_result if existing_result is not None else {}

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

    base_config = {
        "float8_weight_qat": Float8WeightOnlyConfig(),
        "int8_weight_qat": Int8WeightOnlyConfig(),
    }

    quantization_configs = {
        "float8_weight_qat": QATConfig(
            weight_config=Float8FakeQuantizeConfig(),
            step="prepare",
        ),
        "int8_weight_qat": QATConfig(
            weight_config=IntxFakeQuantizeConfig(
                torch.int8,
                "per_channel",
                is_symmetric=True,
            ),
            step="prepare",
        ),
    }

    def persist_quantization_after_finetune(q_model, template_model, base_quant_config):
        finetuned_state_dict = q_model.state_dict()
        fresh_model = clone_model_via_disk(template_model)
        fresh_model.load_state_dict(finetuned_state_dict, strict=False)
        quantize_model(fresh_model, config=base_quant_config)
        return fresh_model

    finetune_cfg = cfg.get("finetune", {})
    finetune_epochs = int(finetune_cfg.get("epoch", finetune_cfg.get("epochs", 1)))
    finetune_lr = finetune_cfg.get("lr")

    def finetune_model(model_to_finetune):
        model_utils.setup_model(model_to_finetune, cfg, change_vocab=False)
        if finetune_lr is not None:
            optim_cfg = copy.deepcopy(model_to_finetune.cfg.optim)
            with open_dict(optim_cfg):
                optim_cfg.lr = finetune_lr
            model_to_finetune.setup_optimization(optim_config=optim_cfg)

        ft_trainer = model_utils.create_trainer(cfg)
        ft_trainer.fit_loop.max_epochs = finetune_epochs
        ft_trainer.fit(model_to_finetune)

    model = _prepare_model_for_evaluation(model_class, model_path, cfg)
    trainer = model_utils.create_trainer(cfg)
    device = trainer.strategy.root_device if trainer.strategy else torch.device("cpu")
    model.to(device)

    params = count_parameters(model)
    language_items = list(language_codes.items())
    _, first_lang_code = language_items[0]
    setup_validation_for_language(model, cfg, first_lang_code)
    sample_batch = next(iter(model._validation_dl))
    macs = estimate_macs(model, sample_batch)
    model_size = measure_model_size_bytes(model)

    if "baseline" not in results:
        baseline_language_results = evaluate_languages_for_model(model, device)
        baseline_result = {
            "params": params,
            "macs": macs,
            "size": model_size,
            "languages": baseline_language_results,
        }

        baseline_reference_languages = (baseline_entry or {}).get("languages", {})
        baseline_reference_params = (baseline_entry or {}).get("params")
        baseline_reference_macs = (baseline_entry or {}).get("macs")

        for lang_name, lang_code in language_items:
            baseline_lang = baseline_reference_languages.get(lang_name) or baseline_reference_languages.get(lang_code)
            if (
                baseline_lang is not None
                and model_size is not None
            ):
                evaluator = ASREvaluator()
                evaluator.cer = baseline_result["languages"][lang_name]["cer"]
                baseline_result["languages"][lang_name]["ces"] = evaluator.compute_ces_from_size(
                    size_baseline=model_size,
                    size_quantized=model_size,
                    cer_baseline=baseline_lang["cer"],
                )
            elif baseline_entry is not None:
                print(f"  Missing baseline (params/macs/cer) for language '{lang_name}' -- skipping CES.")

        results["baseline"] = baseline_result
        if save_progress is not None:
            save_progress()
    else:
        baseline_language_results = results["baseline"]["languages"]

    def score_quantized_languages(language_results: dict, size_current: float) -> None:
        _attach_ces_from_size(language_results, baseline_language_results, model_size, size_current)

    def score_pruned_languages(language_results: dict, size_current: float) -> None:
        _attach_ces_from_size(language_results, baseline_language_results, model_size, size_current)

    if prune:
        pruning_results = results.setdefault("pruning", {})
        pruning_model = _prepare_model_for_evaluation(model_class, model_path, cfg)
        pruning_model.to(device)
        setup_validation_for_language(pruning_model, cfg, first_lang_code)
        prune_schedule = precompute_prune_dimension(pruning_model, prune_ratios, iterative=True)

        current_model = pruning_model
        for prune_ratio, prune_dim in zip(prune_ratios, prune_schedule):
            experiment_key = f"{int(round(prune_ratio * 100))}percent"
            print(f"  == pruning: {experiment_key} (prune_dim={prune_dim}) ==")
            prune_entry = pruning_results.setdefault(experiment_key, {})

            current_model = prune_ffns(
                current_model,
                sample_batch[0].to(device),
                sample_batch[1].to(device),
                prune_dim=prune_dim,
            )
            current_model.to(device)

            if "languages" not in prune_entry:
                prune_params = count_parameters(current_model)
                prune_size = measure_model_size_bytes(current_model)
                prune_macs = estimate_macs(current_model, sample_batch)
                prune_lang_results = evaluate_languages_for_model(current_model, device)
                score_pruned_languages(prune_lang_results, prune_size)

                prune_entry.update({
                    "params": prune_params,
                    "macs": prune_macs,
                    "size": prune_size,
                    "languages": prune_lang_results,
                })
                if save_progress is not None:
                    save_progress()

            if finetune:
                if "finetuned" not in prune_entry:
                    finetune_model(current_model)
                    prune_ft_params = count_parameters(current_model)
                    prune_ft_size = measure_model_size_bytes(current_model)
                    prune_ft_macs = estimate_macs(current_model, sample_batch)
                    prune_ft_lang_results = evaluate_languages_for_model(current_model, device)
                    score_pruned_languages(prune_ft_lang_results, prune_ft_size)
                    prune_entry["finetuned"] = {
                        "params": prune_ft_params,
                        "macs": prune_ft_macs,
                        "size": prune_ft_size,
                        "languages": prune_ft_lang_results,
                    }
                    if save_progress is not None:
                        save_progress()

            if prune_and_quantize:
                prune_entry.setdefault("quantization", {})
                for q_name, q_cfg in quantization_configs.items():
                    print(f"    == pruning+quantization: {q_name} ==")
                    if q_name in prune_entry["quantization"] and (
                        "languages" in prune_entry["quantization"][q_name]
                        and (not finetune or "finetuned" in prune_entry["quantization"][q_name])
                    ):
                        continue
                    restored = model_class.restore_from(model_path, map_location="cpu")

                    cloned = clone_model_via_disk(restored)

                    print("RESTORED:")
                    print(type(restored.tokenizer.tokenizer.vocab_size))
                    print(repr(restored.tokenizer.tokenizer.vocab_size))

                    print("\nCLONED:")
                    print(type(cloned.tokenizer.tokenizer.vocab_size))
                    print(repr(cloned.tokenizer.tokenizer.vocab_size))

                    print("restored tokenizer id:",
                        id(restored.tokenizer.tokenizer))

                    print("cloned tokenizer id:",
                        id(cloned.tokenizer.tokenizer))

                    print("restored tokenizer dict:")
                    print(restored.tokenizer.tokenizer.__dict__)

                    print("cloned tokenizer dict:")
                    print(cloned.tokenizer.tokenizer.__dict__)

                    print(type(restored.tokenizer.tokenizer))
                    print(type(cloned.tokenizer.tokenizer))

                    q_model = clone_model_via_disk(current_model)
                    inspect_model_tokenizer(q_model, "TRANSFORMED BEFORE setup_model")
                    q_template = clone_model_via_disk(current_model)
                    quantize_model(q_model, config=q_cfg)
                    q_model.to(device)

                    q_entry = prune_entry["quantization"].setdefault(q_name, {})
                    if "languages" not in q_entry:
                        q_size = measure_model_size_bytes(q_model)
                        q_lang_results = evaluate_languages_for_model(q_model, device)
                        score_quantized_languages(q_lang_results, q_size)

                        q_entry.update({
                            "languages": q_lang_results,
                            "size": q_size,
                        })
                        if save_progress is not None:
                            save_progress()

                    if finetune:
                        if "finetuned" not in q_entry:
                            q_model = clone_model_via_disk(current_model)

                            fresh = model_class.restore_from(model_path, map_location="cpu")

                            finetune_model(q_model)
                            q_model = persist_quantization_after_finetune(q_model, q_template, base_config[q_name])
                            q_model.to(device)

                            q_ft_size = measure_model_size_bytes(q_model)
                            q_ft_lang_results = evaluate_languages_for_model(q_model, device)
                            score_quantized_languages(q_ft_lang_results, q_ft_size)
                            q_entry["finetuned"] = {
                                "languages": q_ft_lang_results,
                                "size": q_ft_size,
                            }
                            if save_progress is not None:
                                save_progress()

    if quantize and not prune_and_quantize:
        quantization_results = results.setdefault("quantization", {})
        for q_name, q_cfg in quantization_configs.items():
            print(f"  == quantization: {q_name} ==")
            if q_name in quantization_results and (
                "languages" in quantization_results[q_name]
                and (not finetune or "finetuned" in quantization_results[q_name])
            ):
                continue

            q_model = _prepare_model_for_evaluation(model_class, model_path, cfg)

            quantize_model(q_model, config=q_cfg)
            q_model.to(device)

            q_entry = quantization_results.setdefault(q_name, {})
            if "languages" not in q_entry:
                q_size = measure_model_size_bytes(q_model)
                q_lang_results = evaluate_languages_for_model(q_model, device)
                score_quantized_languages(q_lang_results, q_size)

                q_entry.update({
                    "languages": q_lang_results,
                    "size": q_size,
                })
                if save_progress is not None:
                    save_progress()

            if finetune:
                if "finetuned" not in q_entry:
                    finetune_model(q_model)
                    q_model = persist_quantization_after_finetune(q_model, q_template, base_config[q_name])
                    q_model.to(device)

                    q_ft_size = measure_model_size_bytes(q_model)
                    q_ft_lang_results = evaluate_languages_for_model(q_model, device)
                    score_quantized_languages(q_ft_lang_results, q_ft_size)
                    q_entry["finetuned"] = {
                        "languages": q_ft_lang_results,
                        "size": q_ft_size,
                    }
                    if save_progress is not None:
                        save_progress()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more ASR models (WER/CER/CES) across one or more languages.")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to .nemo model checkpoints.")
    parser.add_argument("--model_class", required=True, help="Dotted path to the model class.")
    parser.add_argument("--config", required=True, help="Path to the NeMo config (for validation dataset setup).")
    parser.add_argument("--languages", nargs="+", required=True,
                        help="Language names or codes to evaluate against (must match LANGUAGES and have a corresponding {code}_manifest_filepath in config.model.validation_ds).")
    parser.add_argument("--baseline_file", default=None,
                        help="Optional JSON: {model_filename: {params, macs, languages: {lang: {cer, wer, combined_error}}}}.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.json into.")
    parser.add_argument("--prune", action="store_true", help="Iteratively prune feed-forward layers before evaluation.")
    parser.add_argument("--prune_ratios", nargs="+", type=float, default=[0.1, 0.2, 0.5],
                        help="Pruning ratios to apply iteratively, e.g. 0.1 0.2 0.5.")
    parser.add_argument("--quantize", action="store_true", help="Apply QAT prepare-time fake quantization before evaluation.")
    parser.add_argument("--finetune", action="store_true", help="After QAT prepare, fine-tune each model using cfg.finetune before re-evaluation.")
    parser.add_argument("--prune_and_quantize", action="store_true",
                        help="Quantize each pruned model before its first evaluation and finetuning.")
    args = parser.parse_args()

    if args.prune_and_quantize and not args.prune:
        raise ValueError("--prune_and_quantize requires --prune.")

    cfg = OmegaConf.load(args.config)
    model_class = model_utils.resolve_model_class(args.model_class)
    baseline_data = load_json_file(args.baseline_file) if args.baseline_file else {}
    language_codes = resolve_language_codes(args.languages)

    output_path = str(Path(args.output_dir) / "results.json")
    results = load_json_file(output_path) if Path(output_path).exists() else {}

    for model_path in args.model_paths:
        model_filename = Path(model_path).name
        print(f"\n=== Evaluating {model_filename} ===")
        baseline_entry = baseline_data.get(model_filename) if args.baseline_file else None
        model_results = results.setdefault(model_filename, {})
        results[model_filename] = evaluate_model(
            model_path,
            model_class,
            cfg,
            baseline_entry,
            language_codes,
            existing_result=model_results,
            save_progress=lambda: save_json_file(results, output_path),
            prune=args.prune,
            prune_ratios=tuple(args.prune_ratios),
            quantize=args.quantize,
            finetune=args.finetune,
            prune_and_quantize=args.prune_and_quantize,
        )
        save_json_file(results, output_path)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
