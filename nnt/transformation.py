import copy

import torch.nn as nn
from mamba.mamba_block import MambaBlock

from nnt.nn_mamba_block import NNMambaBlock
from nnt.nn_linear import IsomorphicLinear
from nnt.nn_embedding import IsomorphicEmbedding


def _transform_in_place(module: nn.Module, a_min: float, a_max: float) -> None:
    """
    Recursively replace submodules with their non-negative isomorphic equivalents.

    Replacements (leaves — not recursed into):
      - MambaBlock   -> NNMambaBlock   (handles its own inner Linear/Conv1D swaps)
      - nn.Linear    -> IsomorphicLinear
      - nn.Embedding -> IsomorphicEmbedding

    All other modules (RMSNorm, Dropout, activations, wrapper containers) are
    traversed but left structurally intact.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, MambaBlock):
            setattr(module, name, NNMambaBlock(child, a_min=a_min, a_max=a_max))
        elif isinstance(child, nn.Linear):
            setattr(module, name, IsomorphicLinear(child, a_min=a_min, a_max=a_max))
        elif isinstance(child, nn.Embedding):
            setattr(module, name, IsomorphicEmbedding(child))
        else:
            _transform_in_place(child, a_min, a_max)


def transform_to_nn(model, a_min: float = 0.0, a_max: float = 1.0) -> nn.Module:
    """
    Return a deep-copy of `model` with all MambaBlock, nn.Linear, and
    nn.Embedding submodules isomorphically transformed to be non-negative.

    Mathematically equivalent to the original for inputs in [a_min, a_max].
    Works for raw MambaModel as well as task-specific wrapper models
    (e.g. CIFARMambaModel, SelectiveCopyingMambaModel).
    """
    model_nn = copy.deepcopy(model)
    _transform_in_place(model_nn, a_min, a_max)
    return model_nn
