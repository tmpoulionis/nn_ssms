import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from models.mamba_model import MambaModel
from utils.activations import Activation


class CIFARMambaModel(nn.Module):
    """
    Sequential-CIFAR wrapper: Linear input projection (RGB -> d_model),
    a MambaModel backbone, and an MLP classification head.

    Input:  (B, L, input_dim)
    Output: (B, d_out)
    """
    checkpoint_metric = "val_acc"
    checkpoint_mode = "max"

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_out: int,
        input_dim: int = 3,
        mlp_dims=None,
        mlp_act: str = "gelu",
        use_mlp_prenorm: bool = True,
        out_activation: str = None,
        dropout: float = 0.1,
        pooling: str = "mean",
        **backbone_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.backbone = MambaModel(num_layers=num_layers, d_model=d_model, **backbone_kwargs)

        mlp_dims = mlp_dims if mlp_dims is not None else [d_model, d_out]
        assert mlp_dims[0] == d_model, "First dim in mlp_dims must be d_model."
        assert mlp_dims[-1] == d_out, "Last dim in mlp_dims must be d_out."

        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                if use_mlp_prenorm:
                    mlp_layers.append(nn.RMSNorm(mlp_dims[i + 1]))
                mlp_layers.append(Activation(mlp_act))
                if dropout > 0:
                    mlp_layers.append(nn.Dropout(dropout))
            else:
                if out_activation is not None:
                    mlp_layers.append(Activation(out_activation))
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.backbone(h)
        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "last":
            pooled = h[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")
        return self.head(pooled)

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)

    def compute_metrics(self, logits, y):
        return {"acc": accuracy(logits, y, task="multiclass", num_classes=self.d_out)}
