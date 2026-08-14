"""
Structural FFN pruning for Hugging Face Whisper (torch-pruning).

Mirrors the teammate NeMo Conformer recipe:
  - prune intermediate width of each FFN block (fc1 out / fc2 in)
  - L2 importance on fc1 output neurons
  - ratios rounded so target dim is a multiple of 128

Whisper block shape (encoder / decoder layer):
  d_model -> ffn_dim -> d_model   via  fc1, fc2
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn

try:
    import torch_pruning as tp
except ImportError as e:
    tp = None
    _TP_IMPORT_ERROR = e
else:
    _TP_IMPORT_ERROR = None


Scope = Literal["encoder", "decoder", "both"]


def _require_tp():
    if tp is None:
        raise ImportError(
            "torch-pruning is required for FFN pruning. "
            "pip install torch-pruning"
        ) from _TP_IMPORT_ERROR


def precompute_prune_dimension(
    current_ffn_dim: int,
    prune_ratios: Sequence[float],
    iterative: bool = False,
    min_dim: int = 128,
    align: int = 128,
) -> tuple[int, ...]:
    """
    Convert ratios → number of FFN channels to remove.

    Same rounding as mate: target = max(min_dim, floor(dim*(1-r)/align)*align).
    """
    prune_dims: list[int] = []
    dim = int(current_ffn_dim)
    for ratio in prune_ratios:
        target = int(dim * (1.0 - float(ratio)))
        target = max(min_dim, (target // align) * align)
        if target >= dim:
            raise ValueError(
                f"ratio={ratio} leaves target_dim={target} >= current {dim}; "
                "nothing to prune (try a larger ratio)."
            )
        prune_dims.append(dim - target)
        if iterative:
            dim = target
    return tuple(prune_dims)


def get_ffn_dim(model: nn.Module, scope: Scope = "encoder") -> int:
    """Read current FFN intermediate size from the first matching layer."""
    layers = _iter_transformer_layers(model, scope)
    for layer in layers:
        fc1 = getattr(layer, "fc1", None)
        if isinstance(fc1, nn.Linear):
            return int(fc1.out_features)
    raise ValueError(f"No Whisper FFN fc1 found for scope={scope}")


def _whisper_core(model: nn.Module) -> nn.Module:
    # WhisperForConditionalGeneration.model is WhisperModel
    return model.model if hasattr(model, "model") and hasattr(model.model, "encoder") else model


def _iter_transformer_layers(model: nn.Module, scope: Scope) -> list[nn.Module]:
    core = _whisper_core(model)
    layers: list[nn.Module] = []
    if scope in ("encoder", "both") and hasattr(core, "encoder"):
        layers.extend(list(core.encoder.layers))
    if scope in ("decoder", "both") and hasattr(core, "decoder"):
        layers.extend(list(core.decoder.layers))
    return layers


def collect_ffn_fc1(model: nn.Module, scope: Scope = "encoder") -> list[tuple[str, nn.Linear]]:
    """Return (path_label, fc1) for every FFN block to prune."""
    out: list[tuple[str, nn.Linear]] = []
    core = _whisper_core(model)
    if scope in ("encoder", "both") and hasattr(core, "encoder"):
        for i, layer in enumerate(core.encoder.layers):
            fc1 = getattr(layer, "fc1", None)
            if isinstance(fc1, nn.Linear):
                out.append((f"encoder.layers.{i}.fc1", fc1))
    if scope in ("decoder", "both") and hasattr(core, "decoder"):
        for i, layer in enumerate(core.decoder.layers):
            fc1 = getattr(layer, "fc1", None)
            if isinstance(fc1, nn.Linear):
                out.append((f"decoder.layers.{i}.fc1", fc1))
    return out


class _WhisperPruneForward(nn.Module):
    """Minimal forward that touches encoder (+ decoder) for DependencyGraph."""

    def __init__(self, model: nn.Module, scope: Scope):
        super().__init__()
        self.model = model
        self.scope = scope

    def forward(self, input_features: torch.Tensor, decoder_input_ids: torch.Tensor | None = None):
        if self.scope == "encoder":
            return self.model.model.encoder(input_features).last_hidden_state
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids required when scope includes decoder")
        out = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
        )
        return out.logits


def _example_inputs(model: nn.Module, scope: Scope, device: torch.device):
    cfg = getattr(model, "config", None)
    n_mels = int(getattr(cfg, "num_mel_bins", 80) or 80)
    # short mel to keep DG build memory down
    n_frames = 300
    feats = torch.zeros(1, n_mels, n_frames, device=device)
    if scope == "encoder":
        return feats
    # BOS-ish decoder seed (length 1)
    bos = getattr(cfg, "decoder_start_token_id", None) or getattr(cfg, "bos_token_id", 50256)
    dec = torch.full((1, 1), int(bos), device=device, dtype=torch.long)
    return (feats, dec)


def _l2_prune_indices(linear: nn.Linear, n_prune: int) -> list[int]:
    with torch.no_grad():
        scores = linear.weight.detach().float().pow(2).sum(dim=1)
        if linear.bias is not None:
            scores = scores + linear.bias.detach().float().pow(2)
    return torch.argsort(scores)[:n_prune].tolist()


def prune_ffns(
    model: nn.Module,
    prune_dim: int,
    scope: Scope = "encoder",
    verbose: bool = True,
) -> nn.Module:
    """
    Structurally prune FFN intermediate dim by ``prune_dim`` channels (mate-style).

    Example: 5120 - prune_dim=512 → 4608 (must be % 128 == 0 after prune).
    """
    _require_tp()
    if prune_dim is None or prune_dim <= 0:
        raise ValueError("prune_dim must be > 0")

    device = next(model.parameters()).device
    original_dim = get_ffn_dim(model, scope="encoder" if scope == "both" else scope)
    # If scope=both, encoder and decoder ffn dims should match for Whisper-large
    target_dim = original_dim - int(prune_dim)
    if target_dim % 128 != 0:
        raise ValueError(
            f"target_dim must be a multiple of 128, got {target_dim} "
            f"(original={original_dim}, prune_dim={prune_dim})"
        )
    if target_dim < 128:
        raise ValueError(f"target_dim {target_dim} < 128")

    model.eval()
    targets = collect_ffn_fc1(model, scope=scope)
    if not targets:
        raise ValueError(f"No FFN fc1 layers found for scope={scope}")
    print(f"Found {len(targets)} Whisper FFN fc1 layers (scope={scope}).", flush=True)
    print(f"Pruning FFN {original_dim} → {target_dim} (remove {prune_dim})", flush=True)

    example = _example_inputs(model, scope, device)
    wrapped = _WhisperPruneForward(model, scope)

    for label, fc1 in targets:
        current_dim = fc1.out_features
        if current_dim == target_dim:
            if verbose:
                print(f"  {label}: already {target_dim}, skip", flush=True)
            continue
        if current_dim < target_dim:
            raise ValueError(
                f"{label}: current FFN dim {current_dim} < target {target_dim}"
            )
        n_prune = current_dim - target_dim
        idxs = _l2_prune_indices(fc1, n_prune)

        if verbose:
            print(f"  {label}: {current_dim} → {target_dim} (remove {n_prune})", flush=True)

        # Fresh DG each time (shapes change)
        if isinstance(example, tuple):
            DG = tp.DependencyGraph().build_dependency(wrapped, example_inputs=example)
        else:
            DG = tp.DependencyGraph().build_dependency(wrapped, example_inputs=example)

        group = DG.get_pruning_group(fc1, tp.prune_linear_out_channels, idxs=idxs)
        if not DG.check_pruning_group(group):
            raise RuntimeError(f"Torch-Pruning rejected pruning group for {label}")
        group.prune()

    # Keep HF config in sync for save/load
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if scope in ("encoder", "both") and hasattr(cfg, "encoder_ffn_dim"):
            cfg.encoder_ffn_dim = target_dim
        if scope in ("decoder", "both") and hasattr(cfg, "decoder_ffn_dim"):
            cfg.decoder_ffn_dim = target_dim

    # Sanity: first encoder fc1
    check_scope: Scope = "encoder" if scope == "both" else scope
    new_dim = get_ffn_dim(model, scope=check_scope)
    assert new_dim == target_dim, f"Expected FFN dim {target_dim}, got {new_dim}"
    print("FFN pruning complete.", flush=True)
    return model


def prune_by_ratio(
    model: nn.Module,
    ratio: float,
    scope: Scope = "encoder",
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Prune one-shot by ratio; returns model + metadata."""
    # Use encoder dim as reference (Whisper keeps enc/dec ffn equal)
    ref_scope: Scope = "encoder" if scope == "both" else scope
    current = get_ffn_dim(model, scope=ref_scope)
    (prune_dim,) = precompute_prune_dimension(current, (ratio,), iterative=False)
    target = current - prune_dim
    prune_ffns(model, prune_dim=prune_dim, scope=scope, verbose=verbose)
    meta = {
        "ratio": float(ratio),
        "prune_dim": int(prune_dim),
        "ffn_dim_before": int(current),
        "ffn_dim_after": int(target),
        "scope": scope,
    }
    return model, meta


def ratio_tag(ratio: float) -> str:
    """0.1 → ffn_10, 0.2 → ffn_20, …"""
    pct = int(round(float(ratio) * 100))
    return f"ffn_{pct}"


def list_default_ratios() -> list[float]:
    return [0.1, 0.2, 0.5]
