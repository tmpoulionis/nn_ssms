from mamba.mamba_block import MambaBlock
import torch.nn as nn


class MambaModel(nn.Module):
    """
    Mamba backbone: stack of MambaBlocks with optional per-layer pre-norm
    and final norm. Task-specific heads live in dataset-specific wrapper models
    that hold a MambaModel as a submodule.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        conv_activation: str = "silu",
        delta_activation=None,
        gate_activation=None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_prenorm: bool = True,
        use_final_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.use_prenorm = use_prenorm

        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            if self.use_prenorm:
                self.layer_norms.append(nn.RMSNorm(d_model))

            self.mamba_layers.append(
                MambaBlock(
                    d_model=d_model,
                    conv_activation=conv_activation,
                    delta_activation=delta_activation,
                    gate_activation=gate_activation,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs,
                )
            )
        self.final_norm = nn.RMSNorm(d_model) if use_final_norm else nn.Identity()

    def forward(self, x):
        for i, mamba_block in enumerate(self.mamba_layers):
            if self.use_prenorm:
                x = mamba_block(self.layer_norms[i](x)) + x
            else:
                x = mamba_block(x) + x
        return self.final_norm(x)
