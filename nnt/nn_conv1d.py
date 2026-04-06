import torch
import torch.nn as nn
import torch.nn.functional as F
from nnt.utils import _iso_params

class IsomorphicConv1D(nn.Module):
    """
    Equivalent to a depthwise Conv1d; uses non-negative kernel.
    Only depthwise (groups == in_channels) supported..

    y = conv(x + |a_min|, w_pos) + conv(a_max - x, w_neg_abs) + bias_corr
    """
    def __init__(self, conv: nn.Conv1d, a_min: float = 0.0, a_max: float = 1.0):
        super().__init__()
        assert conv.groups == conv.in_channels, "Only depthwise conv supported"
        abs_amin = abs(a_min)
        w         = conv.weight.data                  # (C, 1, K)
        w_pos     = w.clamp(min=0)
        w_neg_abs = (-w).clamp(min=0)
        bias = conv.bias.data if conv.bias is not None else None
        b_prime, c = _iso_params(
            w_pos.view(w.shape[0], -1), w_neg_abs.view(w.shape[0], -1),
            bias, a_min, a_max
        )

        self.w_pos     = nn.Parameter(w_pos)
        self.w_neg_abs = nn.Parameter(w_neg_abs)
        self.register_buffer('b_prime', b_prime)
        self.register_buffer('c', c)
        self.register_buffer('a_max',     torch.tensor(a_max,    dtype=w.dtype))
        self.register_buffer('abs_amin',  torch.tensor(abs_amin, dtype=w.dtype))
        self.groups  = conv.groups
        self.padding = conv.padding[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p  = self.padding
        xp = F.pad(x + self.abs_amin,  (p, p), value=self.abs_amin.item())
        ax = F.pad(self.a_max - x,     (p, p), value=self.a_max.item())
        y_nn  = F.conv1d(xp, self.w_pos,     padding=0, groups=self.groups) \
           + F.conv1d(ax, self.w_neg_abs, bias=self.b_prime, padding=0, groups=self.groups)
        return y_nn - self.c[:, None] # reshape (C, 1) for broadcasting
