import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, einsum

def selective_scan_fn(
    u, 
    delta,
    A,
    B,
    C,
    D=None,
    return_last_state=False
    ):
    """
    Selective scan implementation (pure Python)
    
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    
    h = A.new_zeros((batch, dim, dstate))
    
    # Discretization
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    deltaA = torch.exp(-torch.einsum('bdl,dn->bdln', delta, A))
    # delta_A = delta_A.clamp(min=-20.0, max=0.0)
    
    ys = []
    last_state = None
    for i in range(seqlen):
        h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', h, C[:, :, i])
        
        if i == seqlen - 1:
            last_state = h.clone()
                
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
            
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)