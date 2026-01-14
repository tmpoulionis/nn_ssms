import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, einsum

def selective_scan_photonic_fn(
    u, 
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None, 
    delta_activation=nn.Softplus(),
    gate_activation=nn.SiLU(),
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
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    
    assert delta_activation is not None, "delta_activation must be provided"
    assert gate_activation is not None, "gate_activation must be provided"
    delta = delta_activation(delta)
    delta = delta.clamp(min=1e-4, max=20.0)
    
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
        
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(-torch.einsum('bdl,dn->bdln', delta, A))
    # delta_A = delta_A.clamp(min=-20.0, max=0.0)
    
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        
    # Debugging checks for non-negativity
    check_negative_values("u", u)
    check_negative_values("delta", delta)
    check_negative_values("delta bias", delta_bias)
    check_negative_values("A", A)
    check_negative_values("B", B)
    check_negative_values("C", C)
    check_negative_values("deltaA", deltaA)
    if D is not None:
        check_negative_values("D", D)
    if z is not None:
        check_negative_values("z", z)
        
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
                
        if i == u.shape[2] - 1:
            last_state = x.clone()
            
        if y.is_complex():
            y = y.real * 2
            
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    
    if z is not None:
        out = out * gate_activation(z)
            
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)

def check_negative_values(name, tensor):
    num_neg = (tensor < 0).sum().item()
    total = tensor.numel()
    
    if num_neg == 0:
        print(f"✔️ Parameter {name} has no negative values. ({total})")
    else:
        print(f"❌ Parameter {name}:")
        print(f"\t {num_neg}/{total} negative values.")
        print(f"\t min: {tensor.min().item()}, max: {tensor.max().item()}")
          