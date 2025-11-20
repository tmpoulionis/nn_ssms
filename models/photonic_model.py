import torch
import torch.nn as nn
import warnings
from utils.activations import Activation
from mamba_ssm.modules.mamba_simple import Mamba

class PhotonicMamba(nn.Module):
    def __init__(self, mamba_activation='silu', **mamba_kwargs):
        super().__init__()
        
        if mamba_kwargs["use_fast_path"] or "use_fast_path" not in mamba_kwargs:
            warnings.warn("Photonic Mamba does not support 'use_fast_path'. Setting 'use_fast_path' to False...")
            mamba_kwargs["use_fast_path"] = False
        
        self.mamba = Mamba(**mamba_kwargs)
        
        self.mamba.activation = "silu"
        self.mamba.act = Activation(mamba_activation)
        
    def forward(self, hidden_states, inference_params=None):
        return self.mamba(hidden_states, inference_params)
    
    def step(self, hidden_states, conv_state, ssm_state):
        return self.mamba.step(hidden_states, conv_state, ssm_state)
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype, **kwargs)
    
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        return self.mamba._get_states_from_cache(inference_params, batch_size, initialize_states)
    