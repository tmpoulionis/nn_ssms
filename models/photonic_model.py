# Copyright (c) 2023, Tri Dao, Albert Gu.
# Credits to https://github.com/state-spaces/mamba.git
# This is a photonic wrapper to their simple_mamba model.

import torch
import torch.nn as nn
import warnings
from utils.activations import Activation
from mamba.mamba_ssm.modules.mamba_simple import Mamba
from einops import rearrange
from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class PhotonicMamba(nn.Module):
    def __init__(self, 
        conv_activation='silu', 
        # delta_activation='softplus', 
        # gate_activation='silu', 
        **mamba_kwargs):
        super().__init__()
        
        # Force use_fast_path to False
        if mamba_kwargs["use_fast_path"] or "use_fast_path" not in mamba_kwargs:
            warnings.warn("Photonic Mamba does not support 'use_fast_path'. Setting 'use_fast_path' to False...")
            mamba_kwargs["use_fast_path"] = False
        
        self.mamba = Mamba(**mamba_kwargs)
        
        self.mamba.act = Activation(conv_activation)
        # self.delta_activation = delta_activation
        # self.gate_activation = gate_activation
        
    def forward(self, hidden_states, inference_params=None):
        return self.mamba.forward(hidden_states, inference_params)
        # """
        # Forward pass with photonic activations.
        # We override the forward function to use our selective_scan_fn.
        # """
        
        # mamba = self.mamba
        # batch, seqlen, dim = hidden_states.shape

        # conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = mamba._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = mamba.step(hidden_states, conv_state, ssm_state)
        #         return out

        # # We do matmul and transpose BLH -> HBL at the same time
        # xz = rearrange(
        #     mamba.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        #     "d (b l) -> b d l",
        #     l=seqlen,
        # )
        # if mamba.in_proj.bias is not None:
        #     xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # A = -torch.exp(mamba.A_log.float())  # (d_inner, d_state)
        # x, z = xz.chunk(2, dim=1)
        
        # # Compute short convolution
        # if conv_state is not None:
        #     # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
        #     # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
        #     conv_state.copy_(F.pad(x, (mamba.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        # x = mamba.act(mamba.conv1d(x)[..., :seqlen])
        
        # # We're careful here about the layout, to avoid extra transposes.
        # # We want dt to have d as the slowest moving dimension
        # # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        # x_dbl = mamba.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        # dt, B, C = torch.split(x_dbl, [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1)
        # dt = mamba.dt_proj.weight @ dt.t()
        # dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # y = selective_scan_fn(
        #     x,
        #     dt,
        #     A,
        #     B,
        #     C,
        #     mamba.D.float(),
        #     z=z,
        #     delta_bias=mamba.dt_proj.bias.float(),
        #     delta_softplus=False,
        #     delta_activation_type=self.delta_activation,
        #     gate_activation_type=self.gate_activation,
        #     return_last_state=ssm_state is not None,
        # )
        
        # if ssm_state is not None:
        #     y, last_state = y
        #     ssm_state.copy_(last_state)
        # y = rearrange(y, "b d l -> b l d")
        # out = mamba.out_proj(y)
        # return out
    
    def step(self, hidden_states, conv_state, ssm_state):
        return self.mamba.step(hidden_states, conv_state, ssm_state)
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
    
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        return self.mamba._get_states_from_cache(inference_params, batch_size, initialize_states)
    