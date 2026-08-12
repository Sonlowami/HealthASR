import torch
import torch_pruning as tp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
	print(f"Added {PROJECT_ROOT} to sys.path")

from utils.model_utils import _KwargsForwardWrapper


def precompute_prune_dimension(model, prune_ratios, iterative=False):
    """
    Precompute the number of FFN channels to prune for each requested ratio.

    The ratios are interpreted as the fraction to remove from the current
    intermediate FFN dimension. When ``iterative`` is True, each step is
    computed from the dimension remaining after the previous step; otherwise
    every step is computed from the original dimension.

    Args:
        model (torch.nn.Module): The model to be pruned.
        prune_ratios (tuple): A tuple containing prune ratios for every ffn layer.
        iterative (bool): If True, perform iterative pruning; otherwise, perform
            one-shot pruning.

    Returns:
        tuple: A tuple containing prune dimensions for every ratio.
    """
    prune_dims = []
    try:
        current_layer_dim = model.encoder.layers[0].feed_forward1.linear1.out_features
        for ratio in prune_ratios:
            target_layer_dim = int(current_layer_dim * (1 - ratio))
            target_layer_dim = max(128, (target_layer_dim // 128) * 128)
            prune_dim = current_layer_dim - target_layer_dim
            prune_dims.append(prune_dim)
            if iterative:
                current_layer_dim = target_layer_dim

        return tuple(prune_dims)

    except (AttributeError, IndexError) as e:
        print(f"Error while accessing model attributes: {e}")
        raise ValueError("The model does not have the expected structure. "
                         "Ensure that the model has an encoder with layers and feed_forward1 attributes.")

def prune_ffns(
    model,
    signal,
    signal_len,
    prune_dim=None,
    verbose=False,
):
    """
    Structurally prune the FFN intermediate dimension of every Conformer block.

    Example:
        512 -> 2048 -> 512
    becomes:
        512 -> 1536 -> 512
    if prune_dim = 512,

    prune_dim must be <= the current FFN dimension and is rounded/checkable
    against a multiple of 128.
    """
    if not prune_dim:
        raise ValueError("prune_dim must be specified and greater than 0.")
    original_dim = model.encoder.layers[0].feed_forward1.linear1.out_features
    target_dim = original_dim - prune_dim
    assert target_dim % 128 == 0, (
        f"target_dim must be a multiple of 128, got {target_dim}"
    )

    model.eval()

    # Find all FFN first projections.
    ffn_layers = []

    for layer_idx, layer in enumerate(model.encoder.layers):
        if not hasattr(layer, "feed_forward1"):
            continue

        if not hasattr(layer.feed_forward1, "linear1"):
            continue

        if not hasattr(layer.feed_forward2, "linear1"):
          continue

        linear1 = layer.feed_forward1.linear1
        linear2 = layer.feed_forward2.linear1

        if not isinstance(linear1, torch.nn.Linear):
            continue
        if not isinstance(linear2, torch.nn.Linear):
            continue

        ffn_layers.append((layer_idx, linear1))
        ffn_layers.append((layer_idx, linear2))

    print(f"Found {len(ffn_layers)} Conformer FFN layers.")

    # Prune each block separately.
    #
    # We rebuild the DependencyGraph after every pruning operation because
    # pruning changes tensor shapes and therefore invalidates the old graph.
    for layer_idx, linear1 in ffn_layers:

        current_dim = linear1.out_features

        if current_dim == target_dim:
            print(f"Layer {layer_idx}: already {target_dim}, skipping.")
            continue

        if current_dim < target_dim:
            raise ValueError(
                f"Layer {layer_idx}: current FFN dimension is "
                f"{current_dim}, which is smaller than target {target_dim}."
            )

        n_prune = current_dim - target_dim

        if verbose:
            print(
                f"\nLayer {layer_idx}: "
                f"{current_dim} -> {target_dim} "
                f"(removing {n_prune})"
            )

        # ------------------------------------------------------------
        # Importance:
        #
        # Use L2 norm of each output neuron's weights (+ bias).
        # linear1.weight has shape:
        #
        #     [FFN_dim, d_model]
        #
        # so each row corresponds to one FFN intermediate neuron.
        # ------------------------------------------------------------

        with torch.no_grad():
            scores = linear1.weight.detach().float().pow(2).sum(dim=1)

            if linear1.bias is not None:
                scores += linear1.bias.detach().float().pow(2)

        # Smallest scores = least important neurons.
        idxs = torch.argsort(scores)[:n_prune].tolist()

        # ------------------------------------------------------------
        # Build a fresh dependency graph.
        # ------------------------------------------------------------

        wrapped = _KwargsForwardWrapper(model)

        DG = tp.DependencyGraph().build_dependency(
            wrapped,
            example_inputs=(signal, signal_len),
        )

        # ------------------------------------------------------------
        # Ask Torch-Pruning to remove output channels from linear1.
        #
        # The dependency graph should propagate this to:
        #
        #   linear1: 512 -> current_dim
        #                  ↓
        #              remove neurons
        #                  ↓
        #   linear2: current_dim -> 512
        #
        # resulting in:
        #
        #   linear1: 512 -> target_dim
        #   linear2: target_dim -> 512
        # ------------------------------------------------------------

        group = DG.get_pruning_group(
            linear1,
            tp.prune_linear_out_channels,
            idxs=idxs,
        )

        if not DG.check_pruning_group(group):
            raise RuntimeError(
                f"Torch-Pruning rejected pruning group for "
                f"Conformer layer {layer_idx}"
            )

        if verbose:
            print(group.details())

        group.prune()

        # Verify immediately.
        new_linear1 = model.encoder.layers[
            layer_idx
        ].feed_forward1.linear1

        new_linear2 = model.encoder.layers[
            layer_idx
        ].feed_forward1.linear2

        if verbose:
            print(
                f"  linear1: {new_linear1.in_features} "
                f"-> {new_linear1.out_features}"
            )
            print(
                f"  linear2: {new_linear2.in_features} "
                f"-> {new_linear2.out_features}"
            )

        assert new_linear1.out_features == target_dim, (
            f"Unexpected linear1 dimension after pruning: "
            f"{new_linear1.out_features}"
        )

        assert new_linear2.in_features == target_dim, (
            f"Unexpected linear2 dimension after pruning: "
            f"{new_linear2.in_features}"
        )

    print("\nFFN pruning complete.")
    return model

    