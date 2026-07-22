"""Curriculum helpers: Sunbird WER ranking (once) + stage selection."""
import logging
import re

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
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


def easiest_fraction(scores: list[float], fraction: float) -> list[int]:
    """Indices of the easiest (lowest score) `fraction` of samples."""
    n = max(1, int(len(scores) * fraction))
    return sorted(range(len(scores)), key=lambda i: scores[i])[:n]


@torch.no_grad()
def score_wer(model, processor, dataset, lang_token_id: int, batch_size: int = 32,
              num_workers: int = 8, max_new_tokens: int = 128):
    """
    Rank clips by Sunbird WER (decode vs reference). Lower WER = easier.
    Uses generate() — slower than NeMo CTC scoring; parallel WAV load + batch help.
    """
    device = next(model.parameters()).device
    language = processor.tokenizer.decode([lang_token_id])
    was_training = model.training
    model.eval()
    # Whisper generation_config always has max_length; we pass max_new_tokens.
    # HF logs a harmless conflict every batch — silence just that logger.
    gen_logger = logging.getLogger("transformers.generation.utils")
    prev_gen_level = gen_logger.level
    gen_logger.setLevel(logging.ERROR)

    class _PathDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.paths = ds["audio"]
            self.texts = ds["text"]

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.paths[i], self.texts[i]

    def _collate(batch):
        paths, texts = zip(*batch)
        wavs = [load_audio(p) for p in paths]
        return list(wavs), list(texts)

    loader = DataLoader(
        _PathDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    wers, total_edits, total_words = [], 0, 0
    try:
        for wavs, texts in tqdm(loader, desc=f"Sunbird WER ({language})"):
            feats = processor.feature_extractor(
                wavs, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=model.dtype)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                ids = model.generate(
                    feats, language=language, task="transcribe",
                    max_new_tokens=max_new_tokens, num_beams=1,
                )
            hyps = processor.batch_decode(ids, skip_special_tokens=True)

            for ref_text, hyp_text in zip(texts, hyps):
                ref, hyp = _norm(ref_text), _norm(hyp_text)
                edits = _edits(ref, hyp)
                wers.append(edits / len(ref) if ref else 0.0)
                total_edits += edits
                total_words += len(ref)
    finally:
        gen_logger.setLevel(prev_gen_level)
        if was_training:
            model.train()
    return wers, (total_edits / total_words if total_words else 0.0)
