import copy
import json
from omegaconf import open_dict
from torch.utils.data import DataLoader
import torch
from nemo.collections.asr.data.audio_to_text_dataset import get_audio_to_text_bpe_dataset_from_config
try:
    import editdistance
except ImportError:
    editdistance = None

def compute_levenshtein_distance(ref: list | str, hyp: list | str) -> int:
    """Computes minimum edit distance between two sequences."""
    if editdistance:
        return editdistance.eval(ref, hyp)
    
    # Fallback standard DP implementation if external library is missing
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def build_scoring_dataset(model, trainer, manifest_path: str):
    """
    Build a plain (non-tarred, non-shuffled) NeMo BPE dataset directly from
    a manifest, via the same construction path NeMo uses internally —
    without touching model.setup_training_data / the model's real dataloader.
    """
    ds_cfg = copy.deepcopy(model.cfg.validation_ds)  # template: already non-tarred, non-shuffled
    with open_dict(ds_cfg):
        ds_cfg.manifest_filepath = manifest_path
        ds_cfg.shuffle = False
        ds_cfg.is_tarred = False
        ds_cfg.tarred_audio_filepaths = None

    dataset = get_audio_to_text_bpe_dataset_from_config(
        config=ds_cfg,
        local_rank=trainer.local_rank,
        global_rank=trainer.global_rank,
        world_size=trainer.world_size,
        tokenizer=model.tokenizer,
        preprocessor_cfg=model.cfg.get("preprocessor", None),
    )
    return dataset, ds_cfg


def score_manifest(model, trainer, manifest_path: str, batch_size: int = 16) -> list[dict]:
    """
    Run the current model over every row in manifest_path, compute per-row
    WER, and return manifest rows sorted easiest -> hardest.

    Relies on shuffle=False + a map-style dataset to guarantee output order
    matches input manifest order (true regardless of num_workers, for
    map-style datasets with no custom sampler).
    """
    dataset, ds_cfg = build_scoring_dataset(model, trainer, manifest_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        collate_fn=dataset.collate_fn,  # NeMo ASR datasets expose this
    )

    with open(manifest_path, encoding="utf-8") as f:
        manifest_rows = [json.loads(line) for line in f if line.strip()]

    device = next(model.parameters()).device
    model.eval()

    scored = []
    row_idx = 0
    with torch.no_grad():
        for batch in loader:
            signal, signal_len, tokens, token_len = batch
            signal, signal_len = signal.to(device), signal_len.to(device)

            log_probs, encoded_len, _ = model.forward(input_signal=signal, input_signal_length=signal_len)
            hypotheses = model.decoding.ctc_decoder_predictions_tensor(log_probs, encoded_len)

            for i in range(signal.shape[0]):
                ref_text = manifest_rows[row_idx]["text"]
                hyp_text = " ".join(hypotheses[i].words)

                ref_words, hyp_words = ref_text.split(), hyp_text.split()
                edits = compute_levenshtein_distance(ref_words, hyp_words)
                wer = edits / len(ref_words) if ref_words else 0.0

                scored.append({"wer": wer, "_manifest_row": manifest_rows[row_idx]})
                row_idx += 1

    model.train()
    scored.sort(key=lambda e: e["wer"])  # easiest (lowest WER) first
    return scored


def write_stage_manifest(ranked_entries: list[dict], active_fraction: float, out_path: str) -> str:
    n = max(1, int(len(ranked_entries) * active_fraction))
    subset = [e["_manifest_row"] for e in ranked_entries[:n]]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path