"""Curriculum helpers: static difficulty ranking + optional WER eval scoring."""
import re

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

try:
    import editdistance
except ImportError:
    editdistance = None


def load_audio(path: str) -> np.ndarray:
    """Read audio as 16 kHz mono float32 (no ffmpeg/torchcodec needed for WAV)."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def _edits(ref: list[str], hyp: list[str]) -> int:
    if editdistance:
        return editdistance.eval(ref, hyp)
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


def _norm(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _estimate_snr_db(wav: np.ndarray, frame: int = 400) -> float:
    """Rough SNR: high-energy frames = signal, low-energy = noise."""
    if len(wav) < frame * 2:
        return 0.0
    n = len(wav) // frame
    energies = np.array([(wav[i * frame:(i + 1) * frame] ** 2).mean() for i in range(n)]) + 1e-12
    noise = np.percentile(energies, 20)
    signal = np.percentile(energies, 80)
    return float(10.0 * np.log10(signal / noise))


def static_difficulty(dataset, weights: dict | None = None, compute_snr: bool = False) -> list[float]:
    """
    One-shot difficulty scores (higher = harder). Uses transcript stats + duration;
    optionally opens audio for SNR. Rank once before training — no model needed.
    """
    weights = {
        "duration": 0.30,
        "transcript_len": 0.30,
        "speaking_rate": 0.15,
        "snr": 0.15,
        "complexity": 0.10,
        **(weights or {}),
    }
    n = len(dataset)
    texts = dataset["text"]
    has_dur = "duration_sec" in dataset.column_names
    duration = np.zeros(n, dtype=np.float64)
    n_words = np.zeros(n, dtype=np.float64)
    avg_word_len = np.zeros(n, dtype=np.float64)
    snr = np.zeros(n, dtype=np.float64)

    if has_dur and not compute_snr:
        duration[:] = np.asarray(dataset["duration_sec"], dtype=np.float64)
        for i, text in enumerate(tqdm(texts, desc="Static difficulty", leave=False)):
            words = _norm(text)
            n_words[i] = max(len(words), 1)
            avg_word_len[i] = np.mean([len(w) for w in words]) if words else 0.0
    else:
        paths = dataset["audio"]
        dur_col = list(dataset["duration_sec"]) if has_dur else [None] * n
        for i in tqdm(range(n), desc="Static difficulty (+SNR)" if compute_snr else "Static difficulty", leave=False):
            words = _norm(texts[i])
            n_words[i] = max(len(words), 1)
            avg_word_len[i] = np.mean([len(w) for w in words]) if words else 0.0
            wav = load_audio(paths[i])
            duration[i] = float(dur_col[i]) if dur_col[i] is not None else len(wav) / 16000.0
            if compute_snr:
                snr[i] = _estimate_snr_db(wav)

    speaking_rate = n_words / np.maximum(duration, 0.1)  # words / sec
    score = (
        weights["duration"] * _minmax(duration)
        + weights["transcript_len"] * _minmax(n_words)
        + weights["speaking_rate"] * _minmax(speaking_rate)
        + weights["complexity"] * _minmax(avg_word_len)
    )
    if compute_snr:
        score = score + weights["snr"] * (1.0 - _minmax(snr))  # low SNR → harder
    return score.tolist()


def easiest_fraction(scores: list[float], fraction: float) -> list[int]:
    """Indices of the easiest (lowest score) `fraction` of samples."""
    n = max(1, int(len(scores) * fraction))
    return sorted(range(len(scores)), key=lambda i: scores[i])[:n]


@torch.no_grad()
def score_wer(model, processor, dataset, lang_token_id: int, batch_size: int = 32):
    """
    Teacher WER pass: decode every clip, return (per_sample_wers, corpus_wer).
    Used for Nzeyimana-style curriculum ranking (once) and for --eval_only.
    Lower WER = easier.
    """
    device = next(model.parameters()).device
    language = processor.tokenizer.decode([lang_token_id])
    was_training = model.training
    model.eval()

    wers, total_edits, total_words = [], 0, 0
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Teacher WER ({language})"):
        rows = dataset[start:start + batch_size]
        feats = processor.feature_extractor(
            [load_audio(p) for p in rows["audio"]], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=model.dtype)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            ids = model.generate(feats, language=language, task="transcribe")
        hyps = processor.batch_decode(ids, skip_special_tokens=True)

        for ref_text, hyp_text in zip(rows["text"], hyps):
            ref, hyp = _norm(ref_text), _norm(hyp_text)
            edits = _edits(ref, hyp)
            wers.append(edits / len(ref) if ref else 0.0)
            total_edits += edits
            total_words += len(ref)

    if was_training:
        model.train()
    return wers, (total_edits / total_words if total_words else 0.0)
