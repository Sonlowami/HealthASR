import argparse
import copy
import json
import sys
import re
import tempfile
import pandas as pd
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict
from torchao.quantization import Int8WeightOnlyConfig, IntxWeightOnlyConfig
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

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

def _clean_text_list(texts: list[str]) -> list[str]:
    """The actual cleaning fix, extracted so both clean_references_hypotheses
    (pairs) and inference-only code (single list, no references) can share it
    without either duplicating the regex chain or faking a references list."""
    texts = [re.sub(r'\u2047', '', s) for s in texts]
    texts = [re.sub(r'[.,:?]', ' ', s) for s in texts]
    texts = [re.sub(r'\s+', ' ', s) for s in texts]
    texts = [s.strip() for s in texts]
    return texts


def clean_references_hypotheses(references: list[str], hypotheses: list[str]) -> tuple[list[str], list[str]]:
    return _clean_text_list(references), _clean_text_list(hypotheses)


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

def add_predictions_column(predictions_dir, lang_name, run_label, utterance_ids, references, hypotheses):
    """
    Merges one model's predictions into the per-language WIDE predictions table,
    adding a `prediction_{run_label}` column joined on utterance_id -- rather than
    overwriting a single file with only the latest model's narrow output.
    """
    path = Path(predictions_dir) / f"{lang_name}_predictions.csv"
    new_col = f"prediction_{run_label}"

    new_data = pd.DataFrame({
        "utterance_id": utterance_ids,
        "reference": references,
        new_col: hypotheses,
    })

    if path.exists():
        existing = pd.read_csv(path)
        if new_col in existing.columns:
            print(f"    [predictions] '{new_col}' already present -- overwriting that column only")
            existing = existing.drop(columns=[new_col])
        merged = existing.merge(new_data, on=["utterance_id", "reference"], how="outer")

        # If the SAME utterance_id shows up with DIFFERENT reference text between this
        # model's run and a previous one, the merge key won't match on 'reference' and
        # utterance_id will appear twice -- surface that loudly rather than silently
        # producing a corrupted table with split rows.
        dup_ids = merged["utterance_id"][merged["utterance_id"].duplicated()].unique()
        if len(dup_ids) > 0:
            raise ValueError(
                f"Reference text mismatch for utterance_id(s) {list(dup_ids)[:5]} between "
                f"existing {path.name} and new '{new_col}' data -- check that text cleaning "
                "is consistent across model evaluation runs before trusting this table."
            )
    else:
        merged = new_data

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)

    n_missing = merged[new_col].isna().sum()
    if n_missing > 0:
        print(f"    [predictions] WARNING: {n_missing} rows missing '{new_col}' "
              f"(utterance present in other models but not this one)")
    print(f"    [predictions] added column '{new_col}' -> {path} ({len(merged)} rows, "
          f"{len(merged.columns) - 2} models so far)")
    return merged

def run_model_inference(model, val_loader, device) -> tuple[list[str], list[str], list[str]]:
    model.to(device)
    model.eval()
    references, hypotheses, utterance_ids = [], [], []
    collection = val_loader.dataset.manifest_processor.collection  # verify this path first, see above

    with torch.no_grad():
        for batch in val_loader:
            signal, signal_len, tokens, token_len, sample_ids = batch  # 5-tuple now that return_sample_id=True
            signal, signal_len = signal.to(device), signal_len.to(device)

            output, output_len = model_utils.run_model_forward(model, signal, signal_len)
            hyps = model_utils.get_hypotheses(model, output, output_len)

            tokens_np, token_len_np = tokens.cpu().numpy(), token_len.cpu().numpy()
            sample_ids_np = sample_ids.cpu().numpy() if torch.is_tensor(sample_ids) else sample_ids
            for t, t_len, hyp, sid in zip(tokens_np, token_len_np, hyps, sample_ids_np):
                references.append(model.tokenizer.ids_to_text(t[:t_len].tolist()))
                if hasattr(hyp, "words"):
                    hypotheses.append(" ".join(hyp.words))
                elif hasattr(hyp, "text"):
                    hypotheses.append(hyp.text)
                elif isinstance(hyp, str):
                    hypotheses.append(str(hyp))
                utterance_ids.append(collection[int(sid)].audio_file)
    model.train()
    return references, hypotheses, utterance_ids


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



def setup_validation_for_language(model, cfg, language_code: str, return_sample_id: bool = False) -> None:
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
        lang_ds_cfg.return_sample_id = return_sample_id
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
    save_predictions_dir: str | None = None,
    prune: bool = False,
    prune_ratios: tuple[float, ...] = (),
    quantize: bool = False,
    finetune: bool = False,
    prune_and_quantize: bool = False,
) -> dict:
    results = existing_result if existing_result is not None else {}
    os.makedirs(save_predictions_dir, exist_ok=True) if save_predictions_dir is not None else None

    def evaluate_languages_for_model(model, device, predictions_dir=save_predictions_dir):
        language_items_local = list(language_codes.items())
        _, first_lang_code_local = language_items_local[0]
        model_utils.setup_model_for_validation(model, cfg)
        setup_validation_for_language(model, cfg, first_lang_code_local, return_sample_id=True)
        
        run_label = Path(model_path).stem

        per_language = {}
        for i, (lang_name, lang_code) in enumerate(language_items_local):
            print(f"  -- language: {lang_name} ({lang_code}) --")
            if i > 0:
                setup_validation_for_language(model, cfg, lang_code, return_sample_id=True)

            references, hypotheses, utterance_ids = run_model_inference(model, model._validation_dl, device)
            references, hypotheses = clean_references_hypotheses(references, hypotheses)

            if predictions_dir is not None:
                add_predictions_column(
                    predictions_dir,
                    lang_name,
                    run_label,
                    utterance_ids,
                    references,
                    hypotheses
                    )


            evaluator = ASREvaluator()
            evaluator.compute_wer(references, hypotheses)
            evaluator.compute_cer(references, hypotheses)
            per_language[lang_name] = evaluator.__to_dict__()
            setup_validation_for_language(model, cfg, lang_code , return_sample_id=False)
        setup_validation_for_language(model, cfg, first_lang_code_local, return_sample_id=False)

        return per_language

    base_config = {
        "int6_weight_qat": IntxWeightOnlyConfig(weight_dtype=torch.int6),
        "int4_weight_qat": IntxWeightOnlyConfig(weight_dtype=torch.int4),
        "int8_weight_qat": Int8WeightOnlyConfig(),
    }

    quantization_configs = {
        "int6_weight_qat": QATConfig(
            weight_config=IntxFakeQuantizeConfig(
                torch.int6,
                "per_channel",
                is_symmetric=True),
            step="prepare",
        ),
        "int4_weight_qat": QATConfig(
                    weight_config=IntxFakeQuantizeConfig(
                        torch.int4,
                        "per_channel",
                        is_symmetric=True),
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
        model_to_finetune = model_to_finetune.train()
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
                    # Only use int8 config for pruning+quantization, since int4/int6 are not supported for pruned models.
                    if q_name not in ("int8_weight_qat",):
                        print(f"    Skipping {q_name} for pruning+quantization (not supported for pruned models).")
                        continue

                    q_model = clone_model_via_disk(current_model)
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
            q_template = clone_model_via_disk(q_model)

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

def transcribe_unlabeled_audio(
    model,
    audio_paths: list[str],
    device,
    batch_size: int = 8,
    run_label: str | None = None,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Runs inference on audio with NO ground-truth transcript, using NeMo's
    high-level model.transcribe() API (confirmed signature: `audio=`,
    `return_hypotheses`, `use_lhotse=True` by default) rather than the
    manifest-driven run_model_inference() path, which requires a `text`
    field to build its dataloader.

    Uses whatever decoding strategy is already configured on `model` --
    does not reset or override it, so it inherits e.g. the beam-search fix
    if setup_model()/change_decoding_strategy() already ran on this model.

    Returns a DataFrame with columns [utterance_id, prediction] (or
    prediction_{run_label} if given), and optionally writes it to output_csv.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        raw_output = model.transcribe(
            audio=audio_paths,
            batch_size=batch_size,
            return_hypotheses=False,
        )

    # TranscriptionReturnType's exact shape isn't nailed down by the signature alone
    # (some NeMo paths return a bare List[str], others a (hyps, ...) tuple, and
    # return_hypotheses=False isn't universally guaranteed to suppress Hypothesis
    # objects for every model class) -- normalize defensively rather than assume.
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]

    predictions = []
    for item in raw_output:
        if isinstance(item, str):
            predictions.append(item)
        elif hasattr(item, "text"):
            predictions.append(item.text)
        elif hasattr(item, "words"):
            predictions.append(" ".join(item.words))
        else:
            raise TypeError(f"Unrecognized transcribe() output element type: {type(item)}")

    predictions = _clean_text_list(predictions)

    col_name = f"prediction_{run_label}" if run_label else "prediction"
    result_df = pd.DataFrame({"utterance_id": audio_paths, col_name: predictions})

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Saved {len(result_df)} transcripts -> {output_csv}")

    model.train()
    return result_df


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
    parser.add_argument("--save_predictions_dir", default=None, help="Optional directory to save per-language predictions CSVs into.")
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
            save_predictions_dir=args.save_predictions_dir,
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
