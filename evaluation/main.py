import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

import utils.model_utils as model_utils
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


def estimate_macs(model, sample_batch):
    """
    Best-effort MACs estimate via thop. Returns None if thop isn't
    installed or profiling fails against this model's forward signature.
    CES is skipped (not computed with a missing term) when this is None.
    """
    if thop_profile is None:
        print("thop not installed -- skipping MACs estimation.")
        return None
    try:
        signal, signal_len, _, _ = sample_batch
        macs, _ = thop_profile(model, inputs=(signal, signal_len), verbose=False)
        return macs
    except Exception as exc:
        print(f"MACs estimation failed: {exc}")
        return None


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

            log_probs, encoded_len, _ = model.forward(input_signal=signal, input_signal_length=signal_len)
            hyps = model.decoding.ctc_decoder_predictions_tensor(log_probs, encoded_len)

            tokens_np, token_len_np = tokens.cpu().numpy(), token_len.cpu().numpy()
            for t, t_len, hyp in zip(tokens_np, token_len_np, hyps):
                references.append(model.tokenizer.ids_to_text(t[:t_len].tolist()))
                hypotheses.append(" ".join(hyp.words))
    model.train()
    return references, hypotheses


# ---------- per-model evaluation ----------

def evaluate_model(model_path: str, model_class, cfg, baseline_entry: dict | None) -> dict:
    """
    Loads one model, sets up its validation data from cfg, computes
    params/MACs, runs inference, computes WER/CER, and CES if a usable
    baseline entry (with cer + params + macs) is available.
    """
    model = model_class.restore_from(model_path)
    model_utils.setup_model_for_validation(model, cfg)

    trainer = model_utils.create_trainer(cfg)
    device = trainer.strategy.root_device if trainer.strategy else torch.device("cpu")

    params = count_parameters(model)
    sample_batch = next(iter(model._validation_dl))
    macs = estimate_macs(model, sample_batch)

    references, hypotheses = run_model_inference(model, model._validation_dl, device)
    references, hypotheses = clean_references_hypotheses(references, hypotheses)

    evaluator = ASREvaluator()
    evaluator.compute_wer(references, hypotheses)
    evaluator.compute_cer(references, hypotheses)

    if baseline_entry is not None:
        if "params" in baseline_entry and "macs" in baseline_entry and macs is not None:
            evaluator.compute_ces(
                params_baseline=baseline_entry["params"],
                params_pruned=params,
                macs_baseline=baseline_entry["macs"],
                macs_pruned=macs,
                cer_baseline=baseline_entry["cer"],
            )
        else:
            print(f"Baseline entry for {model_path} missing params/macs (or MACs estimation "
                  f"failed for this model) -- skipping CES.")

    return evaluator.__to_dict__()


# ---------- entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more ASR models (WER/CER/CES).")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to .nemo model checkpoints.")
    parser.add_argument("--model_class", required=True, help="Dotted path to the model class.")
    parser.add_argument("--config", required=True, help="Path to the NeMo config (for validation dataset setup).")
    parser.add_argument("--baseline_file", default=None,
                         help="Optional JSON file: {model_filename: {cer, wer, combined_error[, params, macs]}}.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results.json into.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model_class = model_utils.resolve_model_class(args.model_class)
    baseline_data = load_json_file(args.baseline_file) if args.baseline_file else {}

    results = {}
    for model_path in args.model_paths:
        model_filename = Path(model_path).name
        print(f"\n=== Evaluating {model_filename} ===")
        baseline_entry = baseline_data.get(model_filename) if args.baseline_file else None
        results[model_filename] = evaluate_model(model_path, model_class, cfg, baseline_entry)

    output_path = str(Path(args.output_dir) / "results.json")
    save_json_file(results, output_path)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()