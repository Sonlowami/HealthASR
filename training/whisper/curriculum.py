"""Per-sample WER scoring and stage selection for Whisper curriculum learning."""
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
    """Levenshtein distance; falls back to DP if editdistance isn't installed."""
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
    """Lowercase, strip punctuation, split into words (for fair WER)."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


@torch.no_grad()
def score_wer(model, processor, dataset, lang_token_id: int, batch_size: int = 32):
    """
    Transcribe every row of `dataset` (columns: audio, text) with the language
    forced to `lang_token_id`. Returns (per_sample_wers, corpus_wer).
    """
    device = next(model.parameters()).device
    language = processor.tokenizer.decode([lang_token_id])
    was_training = model.training
    model.eval()

    wers, total_edits, total_words = [], 0, 0
    for start in tqdm(range(0, len(dataset), batch_size), desc=f"Scoring ({language})"):
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


def easiest_fraction(wers: list[float], fraction: float) -> list[int]:
    """Indices of the easiest (lowest-WER) `fraction` of samples."""
    n = max(1, int(len(wers) * fraction))
    return sorted(range(len(wers)), key=lambda i: wers[i])[:n]
