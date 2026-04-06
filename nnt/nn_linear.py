import torch
import torch.nn as nn
import torch.nn.functional as F
from nnt.utils import _iso_params

class IsomorphicLinear(nn.Module):
    """
    Equivalent to nn.Linear; uses non-negative weight matrices.
    Supports inputs in [a_min, a_max].

    y_nn = W_pos @ (x + |a_min|) + W_neg_abs @ (a_max - x) + b'
    y    = y_nn - c
    """

    def __init__(self, linear: nn.Linear, a_min: float = 0.0, a_max: float = 1.0):
        super().__init__()
        w = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        W_pos     = w.clamp(min=0)
        W_neg_abs = (-w).clamp(min=0)
        b_prime, c = _iso_params(W_pos, W_neg_abs, bias, a_min, a_max)

        self.W_pos     = nn.Parameter(W_pos)
        self.W_neg_abs = nn.Parameter(W_neg_abs)
        self.register_buffer('b_prime',  b_prime)
        self.register_buffer('c',        c)
        self.register_buffer('a_max',    torch.tensor(a_max,      dtype=w.dtype))
        self.register_buffer('abs_amin', torch.tensor(abs(a_min), dtype=w.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_nn = F.linear(x + self.abs_amin, self.W_pos) \
             + F.linear(self.a_max - x,    self.W_neg_abs, self.b_prime)
        return y_nn - self.c
