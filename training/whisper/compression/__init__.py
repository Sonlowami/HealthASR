"""Model-agnostic compression helpers for Whisper (torchao QAT / PTQ)."""

from .quantize import get_qat_scheme, persist_after_finetune, prepare_qat, quantize_model

__all__ = [
    "quantize_model",
    "prepare_qat",
    "persist_after_finetune",
    "get_qat_scheme",
]
