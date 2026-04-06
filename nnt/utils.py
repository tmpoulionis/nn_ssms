import torch

def _iso_params(w_pos: torch.Tensor, w_neg_abs: torch.Tensor, bias, a_min: float, a_max: float):
    """
    Compute b_prime and c for an isomorphic layer with inputs in [a_min, a_max].

        b_adj[i] = b[i] - a_max * neg_sum[i] - |a_min| * pos_sum[i]
        c[i]     = max(0, -b_adj[i])   (per-neuron activation shifting point)
        b_prime  = b_adj + c            (>= 0)
    """
    abs_amin = abs(a_min)
    pos_sum  = w_pos.sum(dim=1)
    neg_sum  = w_neg_abs.sum(dim=1)
    b = bias if bias is not None else neg_sum.new_zeros(neg_sum.shape)
    b_adj = b - a_max * neg_sum - abs_amin * pos_sum
    c = (-b_adj).clamp(min=0)
    return b_adj + c, c
