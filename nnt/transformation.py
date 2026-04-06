import copy
import warnings

import torch.nn as nn
from mamba.mamba_block import MambaBlock
from nnt.nn_mamba_block import NNMambaBlock
from nnt.nn_linear import IsomorphicLinear


def _replace_head_linears(head: nn.Sequential, a_min: float, a_max: float) -> nn.Sequential:
    """Replace nn.Linear layers inside an nn.Sequential with IsomorphicLinear."""
    new_layers = []
    for layer in head:
        if isinstance(layer, nn.Linear):
            new_layers.append(IsomorphicLinear(layer, a_min=a_min, a_max=a_max))
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def transform_to_nn(model, a_min: float = 0.0, a_max: float = 1.0) -> nn.Module:
    """
    Return a deep-copied of a MambaModel with all weight matrices isomorphically
    transformed to be non-negative.  Mathematically equivalent to the original
    for inputs in [a_min, a_max].

    Replaces:
      - Each MambaBlock       -> NNMambaBlock
      - MLP head nn.Linear    -> IsomorphicLinear

    RMSNorm, Dropout, Activation, Embedding layers are left unchanged.
    """
    model_nn = copy.deepcopy(model)

    # --- Mamba layers ---
    for i, block in enumerate(model_nn.mamba_layers):
        if isinstance(block, MambaBlock):
            model_nn.mamba_layers[i] = NNMambaBlock(
                block, a_min=a_min, a_max=a_max,
            )
        else:
            warnings.warn(
                f"mamba_layers[{i}] is {type(block).__name__}, not MambaBlock — skipped."
            )
            
    # --- Head ---
    if model_nn.head is not None:
        if isinstance(model_nn.head, nn.Sequential):
            model_nn.head = _replace_head_linears(model_nn.head, a_min=a_min, a_max=a_max)
        elif isinstance(model_nn.head, nn.Linear):
            model_nn.head = IsomorphicLinear(model_nn.head, a_min=a_min, a_max=a_max)
        else:
            warnings.warn(
                f"head is {type(model_nn.head).__name__}, not Sequential or Linear — skipped."
            )

    return model_nn
