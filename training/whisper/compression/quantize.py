"""
torchao quantization helpers (Whisper / any nn.Module with Linear layers).

Mirrors the teammate NeMo flow:
  prepare QAT (fake quant) → short finetune → load weights into a fresh
  model → apply real weight-only PTQ (Int8 / Float8).

Default torchao filter only matches nn.Linear; Conv1d in Whisper's encoder
stem is left in full precision unless a custom filter_fn is passed.
"""
from __future__ import annotations

import torch
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

try:
    from torchao.quantization import Float8WeightOnlyConfig
    from torchao.quantization.qat import Float8FakeQuantizeConfig
except ImportError:  # older torchao
    Float8WeightOnlyConfig = None
    Float8FakeQuantizeConfig = None


def quantize_model(model: torch.nn.Module, config=None, filter_fn=None):
    """In-place quantization via torchao.quantize_."""
    if config is None:
        config = Int8WeightOnlyConfig()
    kwargs = {}
    if filter_fn is not None:
        kwargs["filter_fn"] = filter_fn
    quantize_(model, config, **kwargs)
    return model


def get_qat_scheme(name: str) -> dict:
    """
    Return prepare (QAT fake-quant) + base (PTQ after finetune) configs.

    Names match the teammate script: int8_weight_qat, float8_weight_qat.
    """
    name = name.lower().strip()
    if name in ("int8_weight_qat", "int8", "int8_qat"):
        return {
            "name": "int8_weight_qat",
            "prepare": QATConfig(
                weight_config=IntxFakeQuantizeConfig(
                    torch.int8,
                    "per_channel",
                    is_symmetric=True,
                ),
                step="prepare",
            ),
            "base": Int8WeightOnlyConfig(),
        }
    if name in ("float8_weight_qat", "float8", "float8_qat"):
        if Float8FakeQuantizeConfig is None or Float8WeightOnlyConfig is None:
            raise ImportError(
                "float8_weight_qat requires a newer torchao with Float8*Config. "
                "pip install -U torchao, or use --quant int8_weight_qat"
            )
        return {
            "name": "float8_weight_qat",
            "prepare": QATConfig(
                weight_config=Float8FakeQuantizeConfig(),
                step="prepare",
            ),
            "base": Float8WeightOnlyConfig(),
        }
    raise ValueError(
        f"Unknown QAT scheme '{name}'. "
        "Use: int8_weight_qat | float8_weight_qat"
    )


def prepare_qat(model: torch.nn.Module, scheme_name: str, filter_fn=None):
    """Insert fake-quant (QAT prepare) for short finetuning."""
    scheme = get_qat_scheme(scheme_name)
    return quantize_model(model, config=scheme["prepare"], filter_fn=filter_fn)


def persist_after_finetune(
    finetuned_model: torch.nn.Module,
    load_fresh_fn,
    scheme_name: str,
    filter_fn=None,
) -> torch.nn.Module:
    """
    Like teammate persist_quantization_after_finetune:
      copy finetuned state → fresh full-precision model → real weight-only PTQ.
    """
    scheme = get_qat_scheme(scheme_name)
    state = finetuned_model.state_dict()
    fresh = load_fresh_fn()
    # Fake-quant modules may add buffers; ignore extras / missing on reload
    fresh.load_state_dict(state, strict=False)
    quantize_model(fresh, config=scheme["base"], filter_fn=filter_fn)
    return fresh
