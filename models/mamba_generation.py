from mamba.mamba_block import MambaBlock
import torch.nn as nn


class GenerationMambaModel(nn.Module):
    '''
    Token-level generation wrapper around a Mamba stack.
    '''
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        conv_activation: str = "silu",
        delta_activation: str = None,
        gate_activation: str = None,
        use_prenorm: bool = True,
        use_final_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_out = vocab_size
        self.use_prenorm = use_prenorm

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.layer_norms = nn.ModuleList()
        self.mamba_layers = nn.ModuleList()
        for _ in range(num_layers):
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
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        x: (B, L) long
        returns: (B, L, vocab_size)
        '''
        h = self.embedding(x)
        h = self.embed_dropout(h)

        for i, mamba_block in enumerate(self.mamba_layers):
            if self.use_prenorm:
                residual = h
                h = self.layer_norms[i](h)
                h = mamba_block(h) + residual
            else:
                h = mamba_block(h)

        h = self.final_norm(h)
        return self.lm_head(h)
