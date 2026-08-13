"""
torchao quantization helpers (Whisper / any nn.Module with Linear layers).

Mirrors the teammate NeMo flow:
  prepare QAT (fake quant) → short finetune → load weights into a fresh
  model → apply real weight-only PTQ (Int8 / Int4 / Int6 / Float8).

Default torchao filter only matches nn.Linear; Conv1d in Whisper's encoder
stem is left in full precision unless a custom filter_fn is passed.
"""
from __future__ import annotations

import torch
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig

try:
    from torchao.quantization import Float8WeightOnlyConfig, Int4WeightOnlyConfig
except ImportError:
    Float8WeightOnlyConfig = None
    Int4WeightOnlyConfig = None

try:
    from torchao.quantization.qat import Float8FakeQuantizeConfig
except ImportError:
    Float8FakeQuantizeConfig = None

# Generic intx PTQ (bit_width=6 etc.) — API varies by torchao version
try:
    from torchao.quantization import IntxWeightOnlyConfig
except ImportError:
    try:
        from torchao.quantization.quant_api import IntxWeightOnlyConfig
    except ImportError:
        IntxWeightOnlyConfig = None


def quantize_model(model: torch.nn.Module, config=None, filter_fn=None):
    """In-place quantization via torchao.quantize_."""
    if config is None:
        config = Int8WeightOnlyConfig()
    kwargs = {}
    if filter_fn is not None:
        kwargs["filter_fn"] = filter_fn
    quantize_(model, config, **kwargs)
    return model


def _make_intx_weight_only(bit_width: int):
    """Build IntxWeightOnlyConfig across torchao versions."""
    if IntxWeightOnlyConfig is None:
        return None
    for kwargs in (
        {"bit_width": bit_width, "version": 2},
        {"bit_width": bit_width},
        {"weight_dtype": getattr(torch, f"int{bit_width}", None)},
    ):
        try:
            # drop None values
            kw = {k: v for k, v in kwargs.items() if v is not None}
            return IntxWeightOnlyConfig(**kw)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def get_qat_scheme(name: str) -> dict:
    """
    Return prepare (QAT fake-quant) + base (PTQ after finetune) configs.

    Supported names:
      int8_weight_qat | int4_weight_qat | int6_weight_qat | float8_weight_qat
    """
    name = name.lower().strip()

    if name in ("int8_weight_qat", "int8", "int8_qat"):
        return {
            "name": "int8_weight_qat",
            "prepare": QATConfig(
                weight_config=IntxFakeQuantizeConfig(
                    torch.int8, "per_channel", is_symmetric=True,
                ),
                step="prepare",
            ),
            "base": Int8WeightOnlyConfig(),
        }

    if name in ("int4_weight_qat", "int4", "int4_qat"):
        if Int4WeightOnlyConfig is None:
            # Fall back to IntxWeightOnlyConfig(bit_width=4)
            base = _make_intx_weight_only(4)
            if base is None:
                raise ImportError(
                    "int4_weight_qat requires torchao Int4WeightOnlyConfig or "
                    "IntxWeightOnlyConfig. pip install -U torchao"
                )
        else:
            try:
                base = Int4WeightOnlyConfig(group_size=32)
            except TypeError:
                base = Int4WeightOnlyConfig()
        # Fake quant: int4 + group_size is the common torchao pattern
        try:
            weight_fq = IntxFakeQuantizeConfig(
                torch.int4, group_size=32, is_symmetric=True,
            )
        except TypeError:
            weight_fq = IntxFakeQuantizeConfig(
                torch.int4, "per_channel", is_symmetric=True,
            )
        return {
            "name": "int4_weight_qat",
            "prepare": QATConfig(weight_config=weight_fq, step="prepare"),
            "base": base,
        }

    if name in ("int6_weight_qat", "int6", "int6_qat"):
        base = _make_intx_weight_only(6)
        if base is None:
            raise ImportError(
                "int6_weight_qat requires torchao IntxWeightOnlyConfig with "
                "bit_width=6. pip install -U torchao, or use int4/int8."
            )
        # Intx fake quant with 6-bit — try bit_width kw if dtype int6 missing
        weight_fq = None
        for args, kwargs in (
            ((torch.int8,), {"bit_width": 6, "granularity": "per_channel", "is_symmetric": True}),
            ((6,), {"granularity": "per_channel", "is_symmetric": True}),
            ((torch.int8,), {"is_symmetric": True}),  # last resort: document as approx
        ):
            try:
                weight_fq = IntxFakeQuantizeConfig(*args, **kwargs)
                break
            except TypeError:
                continue
        if weight_fq is None:
            raise ImportError(
                "Could not construct IntxFakeQuantizeConfig for int6. "
                "Upgrade torchao or use --quant int4_weight_qat."
            )
        return {
            "name": "int6_weight_qat",
            "prepare": QATConfig(weight_config=weight_fq, step="prepare"),
            "base": base,
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
        "Use: int8_weight_qat | int4_weight_qat | int6_weight_qat | float8_weight_qat"
    )


def list_schemes() -> list[str]:
    return [
        "int8_weight_qat",
        "int4_weight_qat",
        "int6_weight_qat",
        "float8_weight_qat",
    ]


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
    fresh.load_state_dict(state, strict=False)
    quantize_model(fresh, config=scheme["base"], filter_fn=filter_fn)
    return fresh
