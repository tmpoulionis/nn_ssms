from mamba.mamba_ssm.modules.mamba_simple import Mamba
from models.photonic_model import PhotonicMamba
import torch.nn as nn
from utils.activations import Activation
from einops import rearrange

class MambaModel(nn.Module):
    '''
    Input shape:  (B, L, D) 
    
    d_out, n_heads, head_dims specified only for classification tasks
    '''
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        mamba_activation: str="silu",
        d_state: int=16,
        d_conv: int=4,
        expand: int=2,
        use_prenorm: bool=True,
        use_final_norm: bool=True,
        mlp_dims: dict=None,
        d_out: int=None,
        mlp_act: str='gelu',
        out_activation: str=None,
        dropout: float=0.1,
        use_layernorm: bool=True,
        return_last_state: bool=False,
        **kwargs
    ):
        super().__init__()
        
        "Mamba Stack"
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_out = d_out if d_out is not None else d_model
        self.return_last_state = return_last_state
        self.use_prenorm = use_prenorm
        
        # Create Mamba layers with pre-normalization
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # NEW: Layer norms before each Mamba block
        
        for _ in range(self.num_layers):
            if self.use_prenorm:
                self.layer_norms.append(nn.LayerNorm(d_model))
            
            self.mamba_layers.append(
                PhotonicMamba(
                    d_model=d_model,
                    mamba_activation=mamba_activation,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs
                )
            )
            
        self.final_norm = nn.LayerNorm(d_model) if use_final_norm else nn.Identity()
        
        "Classification head"
        mlp_layers = []
        if mlp_dims is None:
            self.head = None
        else:
            assert mlp_dims[0] == d_model, "First dim in mlp_dims must be d_model."
            assert mlp_dims[-1] == d_out, "Last dim in mlp_dims must be num_classes."
        
            for i in range(len(mlp_dims) - 1):
                mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
                if i < len(mlp_dims) - 2:
                    if use_layernorm:
                        mlp_layers.append(nn.LayerNorm(mlp_dims[i+1]))
                    mlp_layers.append(Activation(mlp_act))

                    if dropout > 0:
                        mlp_layers.append(nn.Dropout(dropout))
                else:
                    # Optional output activation
                    if out_activation is not None:
                        mlp_layers.append(Activation(out_activation))
                    
        self.head = nn.Sequential(*mlp_layers)
        
    def forward(self, x):
        '''
        x: (B, L, D)
        '''
        
        for i, mamba_block in enumerate(self.mamba_layers):
            if self.use_prenorm:
                # Pre-normalize then apply Mamba with residual connection
                residual = x
                x = self.layer_norms[i](x)
                x = mamba_block(x) + residual # Residual connection
            else:
                x = mamba_block(x) # (B, L, D)
        
        x = self.final_norm(x)
        last_state = x[:, -1, :] # (B, D)
        
        if self.head is not None:
            if self.return_last_state:
                return self.head(last_state), last_state 
            else:
                return self.head(last_state) # (B, D_out)
        else:
            return x
        
    def get_last_hidden_state(self, x):
        """"
        Returns the last hidden state before the classification head.
        x: (B, L, D)
        """
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)
        
        x = self.final_norm(x)
        
        return x