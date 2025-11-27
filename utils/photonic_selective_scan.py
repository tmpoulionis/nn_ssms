import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, einsum

class PhotonicSelectiveScanFn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_activation=None, gate_activation=None, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
            
        out = selective_scan_photonic_fn(u, delta, A, B, C, D, z, delta_bias, delta_activation, gate_activation, return_last_state)
        
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)
        ctx.delta_activation = delta_activation
        ctx.gate_activation = gate_activation
        
        if return_last_state:
            out, last_state = out
            ctx.last_state = last_state
            return out, last_state
        return out
    
    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, z, delta_bias = ctx.saved_tensors
        delta_activation = ctx.delta_activation
        gate_activation = ctx.gate_activation
        
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
            
        dtype_in = u.type
        u = u.float()
        delta = delta.float()
        dout = dout.float()
        
        batch, dim, seq_len = u.shape
        dstate = A.shape[1]
        
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        
        if delta_bias is not None:
            delta = delta + delta_bias
    
    
def selective_scan_photonic_fn(
    u, 
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None, 
    delta_activation=None,
    gate_activation=None,
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
        
    if delta_activation is not None:
        delta = delta_activation(delta)
        
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
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    
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
        if gate_activation is not None:
            out = out * gate_activation(z)
            
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)