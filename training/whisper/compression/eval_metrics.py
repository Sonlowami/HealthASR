"""
Size / MACs / Digital Umuganda metrics (WER, CER, CES, combined_error).

ASREvaluator mirrors the teammate NeMo evaluation.ASREvaluator API.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

try:
    import editdistance
except ImportError:
    editdistance = None

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


def _levenshtein(a, b) -> int:
    """Edit distance for sequences (list[str] words or str characters)."""
    if editdistance is not None:
        return int(editdistance.eval(a, b))
    # Fallback DP
    if isinstance(a, str):
        a, b = list(a), list(b)
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = cur
    return prev[m]


class ASREvaluator:
    """WER, CER, CES, combined_error — Digital Umuganda challenge metrics."""

    def __init__(self):
        self.wer: float | None = None
        self.cer: float | None = None
        self.ces: float | None = None

    def compute_wer(self, references: list[str], hypotheses: list[str]) -> float:
        total_edits = 0
        total_words = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words, hyp_words = ref.split(), hyp.split()
            total_edits += _levenshtein(ref_words, hyp_words)
            total_words += len(ref_words)
        self.wer = total_edits / total_words if total_words > 0 else 0.0
        return self.wer

    def compute_cer(self, references: list[str], hypotheses: list[str]) -> float:
        total_edits = 0
        total_chars = 0
        for ref, hyp in zip(references, hypotheses):
            total_edits += _levenshtein(ref, hyp)
            total_chars += len(ref)
        self.cer = total_edits / total_chars if total_chars > 0 else 0.0
        return self.cer

    def compute_combined_error(self) -> float:
        """JSON field combined_error = 1 - (0.4*WER + 0.6*CER); higher is better."""
        if self.wer is None or self.cer is None:
            raise ValueError("Call compute_wer() and compute_cer() first.")
        return 1 - (0.4 * self.wer + 0.6 * self.cer)

    def compute_ces(
        self,
        params_baseline: float,
        params_pruned: float,
        macs_baseline: float,
        macs_pruned: float,
        cer_baseline: float,
    ) -> float:
        if self.cer is None:
            raise ValueError("Call compute_cer() before compute_ces().")
        param_reduction = (params_baseline - params_pruned) / params_baseline
        macs_reduction = (macs_baseline - macs_pruned) / macs_baseline
        x = 0.5 * (param_reduction + macs_reduction)
        y = max(0.0, 1 - (self.cer - cer_baseline) / cer_baseline) if cer_baseline > 0 else (
            1.0 if self.cer == 0 else 0.0
        )
        self.ces = math.sqrt((1 - x) ** 2 + (1 - y) ** 2)
        return self.ces

    def compute_ces_from_size(
        self,
        size_baseline: float,
        size_quantized: float,
        cer_baseline: float,
    ) -> float:
        if self.cer is None:
            raise ValueError("Call compute_cer() before compute_ces_from_size().")
        if size_baseline <= 0:
            raise ValueError("size_baseline must be > 0")
        x = (size_baseline - size_quantized) / size_baseline
        if cer_baseline <= 0:
            y = 1.0 if self.cer == 0 else 0.0
        else:
            y = max(0.0, 1 - (self.cer - cer_baseline) / cer_baseline)
        self.ces = math.sqrt((1 - x) ** 2 + (1 - y) ** 2)
        return self.ces

    def to_dict(self) -> dict:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "ces": self.ces,
            "combined_error": (
                self.compute_combined_error()
                if self.wer is not None and self.cer is not None
                else None
            ),
        }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_model_size_bytes(model: torch.nn.Module) -> int:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        return Path(tmp_path).stat().st_size
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def estimate_macs_whisper(model: torch.nn.Module, n_mels: int | None = None, n_frames: int = 3000) -> float | None:
    """MACs for one encoder forward with a 30s-ish mel spectrogram. None if thop unavailable."""
    if thop_profile is None:
        print("thop not installed — skipping MACs (pip install thop)", flush=True)
        return None

    # Whisper-large-v3 uses 128 mels; older models use 80. Prefer config / conv1.
    if n_mels is None:
        n_mels = 80
        cfg = getattr(model, "config", None)
        if cfg is not None and getattr(cfg, "num_mel_bins", None):
            n_mels = int(cfg.num_mel_bins)
        else:
            try:
                conv1 = model.model.encoder.conv1
                n_mels = int(conv1.weight.shape[1])
            except Exception:
                pass

    device = next(model.parameters()).device
    # thop often prefers float32; quantized wrappers may still accept it
    dtype = torch.float32
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, n_mels, n_frames, device=device, dtype=dtype)
    try:
        # Prefer encoder-only if present (cheaper / more stable under quant wrappers)
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            enc = model.model.encoder
            # Match encoder weight dtype when possible
            try:
                w = enc.conv1.weight
                dummy = dummy.to(dtype=w.dtype)
            except Exception:
                pass
            macs, _ = thop_profile(enc, inputs=(dummy,), verbose=False)
        else:
            macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return float(macs)
    except Exception as exc:
        print(f"MACs estimation failed: {exc}", flush=True)
        return None
    finally:
        # thop leaves hooks; clear best-effort
        for module in model.modules():
            for attr in ("total_ops", "total_params"):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        pass
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
        if was_training:
            model.train()


def attach_ces_from_size(
    language_results: dict,
    baseline_languages: dict,
    size_baseline: float,
    size_current: float,
) -> None:
    """In-place: set ces on each language using baseline CER + current CER/size."""
    for lang_name, lang_result in language_results.items():
        base = baseline_languages.get(lang_name) or {}
        cer_b = base.get("cer")
        cer_q = lang_result.get("cer")
        if cer_b is None or cer_q is None or size_baseline is None or size_current is None:
            lang_result["ces"] = None
            continue
        ev = ASREvaluator()
        ev.cer = float(cer_q)
        lang_result["ces"] = ev.compute_ces_from_size(
            size_baseline=float(size_baseline),
            size_quantized=float(size_current),
            cer_baseline=float(cer_b),
        )
