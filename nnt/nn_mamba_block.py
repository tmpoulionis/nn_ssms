import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.activations import LinearBounded

from nnt.nn_linear import IsomorphicLinear
from nnt.nn_conv1d import IsomorphicConv1D
from nnt.nn_selective_scan import isomorphic_selective_scan


class NNMambaBlock(nn.Module):
    """
    MambaBlock with all weight matrices isomorphically transformed to be non-negative.
    Mathematically equivalent to the original for inputs in [a_min, a_max].
    """
    def __init__(
        self,
        source,
        a_min: float = 0.0,
        a_max: float = 1.0,
        ):
        super().__init__()
        self.d_model  = source.d_model
        self.d_state  = source.d_state
        self.d_conv   = source.d_conv
        self.expand   = source.expand
        self.d_inner  = source.d_inner
        self.dt_rank  = source.dt_rank

        self.in_proj = IsomorphicLinear(source.in_proj, a_min=a_min, a_max=a_max)
        self.conv1d  = IsomorphicConv1D(source.conv1d, a_min=a_min, a_max=a_max)
        self.x_proj  = IsomorphicLinear(source.x_proj, a_min=a_min, a_max=a_max)
        self.dt_proj = IsomorphicLinear(source.dt_proj, a_min=a_min, a_max=a_max)
        self.out_proj = IsomorphicLinear(source.out_proj, a_min=a_min, a_max=a_max)

        self.conv_act = source.conv_act
        self.delta_act = source.delta_act
        self.gate_act  = source.gate_act
        self.bound = source.bound

        self.A_log = nn.Parameter(source.A_log.data.clone())
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter decomposition
        abs_amin       = abs(a_min)
        D              = source.D.data.clone()
        self.D_pos     = nn.Parameter(D.clamp(min=0))
        self.D_neg_abs = nn.Parameter((-D).clamp(min=0))
        self.register_buffer('c_D', abs_amin * self.D_pos.data + a_max * self.D_neg_abs.data) # c_D = D_pos * |a_min| + D_neg_abs * a_max 
        self.register_buffer('abs_amin', torch.tensor(abs_amin, dtype=torch.float32))
        self.register_buffer('a_max',    torch.tensor(a_max,    dtype=torch.float32))
        self.D_pos._no_weight_decay = True
        self.D_neg_abs._no_weight_decay = True
        
        
    def _apply_bound(self, x):
        return self.bound(x)

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("inference_params not supported in NNMambaBlock")

        batch, seqlen, dim = hidden_states.shape

        xz = self._apply_bound(
            rearrange(
            self.in_proj(rearrange(hidden_states, "b l d -> (b l) d")), 
            "(b l) d -> b d l",
            l=seqlen
            )
        )
        
        A = torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        x = self._apply_bound(
            self.conv_act(self.conv1d(x)[..., :seqlen])
        )
        
        # ---- Input-dependent parameters ----------------------------------------------------
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self._apply_bound(dt)
    
        c_B  = self.x_proj.c[self.dt_rank:self.dt_rank + self.d_state]   # (d_state,)
        c_C  = self.x_proj.c[self.dt_rank + self.d_state:]                # (d_state,)
        B_nn      = B  + c_B    # (B*L, d_state), non-negative
        C_nn      = C  + c_C    # (B*L, d_state), non-negative

        B_nn = rearrange(B_nn, "(b l) n -> b n l", l=seqlen).contiguous()  # (B, d_state, L)
        C_nn = rearrange(C_nn, "(b l) n -> b n l", l=seqlen).contiguous()  # (B, d_state, L)

        dt = rearrange(
            self.dt_proj(dt), "(b l) d -> b d l", l=seqlen  # (B, d_inner, L)
        )
        dt = self.delta_act(dt).clamp(min=1e-4, max=20.0)
        
        # ---- Isomorphic S6 block -----------------------------------------------
        y = isomorphic_selective_scan(
            u=x, delta=dt, A=A,
            B_nn=B_nn, c_B=c_B,
            C_nn=C_nn, c_C=c_C,
            D_pos=self.D_pos, D_neg_abs=self.D_neg_abs, c_D=self.c_D,
            abs_amin=float(self.abs_amin), a_max=float(self.a_max),
            return_last_state=False,
        )
        y = self._apply_bound(y)

        # ------ Output gating ------------------------------------------------------
        if z is not None:
            gate = self.gate_act(z)
            abs_amin = float(self.abs_amin)
            y_nn = y + abs_amin          # shift y into R+
            out_nn = y_nn * gate         # non-negative product
            c_y = abs_amin * gate        # shift point (non-negative)
            y = out_nn - c_y             # recover original: y * gate
        
        # if ssm_state is not None:
        #     y, last_state = y
        #     ssm_state.copy_(last_state)
            
        y = self._apply_bound(
            rearrange(y, "b d l -> b l d")
        )
        out = self.out_proj(y)
        return out