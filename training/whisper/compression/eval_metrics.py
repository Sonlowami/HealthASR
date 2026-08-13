"""Size / CES helpers for compression experiments."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_model_size_bytes(model: torch.nn.Module) -> int:
    """Serialize state_dict to a temp file and return byte size."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        return Path(tmp_path).stat().st_size
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def compute_ces(size_baseline: float, size_compressed: float, cer_baseline: float, cer_compressed: float) -> float:
    """
    Compression-Error Score (same idea as teammate ASREvaluator.compute_ces_from_size).
    Lower is better when compression helps more than CER degrades.
    Requires CER; optional for Whisper (we primarily report WER + size).
    """
    if size_baseline <= 0 or size_compressed <= 0:
        return float("nan")
    size_ratio = size_compressed / size_baseline
    # avoid div-by-zero if baseline CER is 0
    err_ratio = (cer_compressed + 1e-8) / (cer_baseline + 1e-8)
    return float(size_ratio * err_ratio)
