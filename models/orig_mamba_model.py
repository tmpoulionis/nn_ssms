from mamba_ssm import Mamba
import torch.nn as nn


class OrigMambaModel(nn.Module):
    """
    Vanilla Mamba backbone using the upstream mamba_ssm package
    (CUDA-fused selective scan, use_fast_path=True, SiLU activations).
    Same residual / pre-norm / final-norm structure as MambaModel, so it
    drops into the existing CIFARMambaModel / SelectiveCopyingMambaModel
    wrappers via the backbone_cls kwarg.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
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
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )
        self.final_norm = nn.RMSNorm(d_model) if use_final_norm else nn.Identity()

    def forward(self, x):
        for i, block in enumerate(self.mamba_layers):
            if self.use_prenorm:
                x = block(self.layer_norms[i](x)) + x
            else:
                x = block(x) + x
        return self.final_norm(x)
