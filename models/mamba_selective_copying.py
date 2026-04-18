import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_model import MambaModel


class SelectiveCopyingMambaModel(nn.Module):
    """
    Token-level language-model wrapper for the Selective Copying task.

    Input:  (B, L)
    Output: (B, L, vocab_size)
    """
    checkpoint_metric = "val_loss"
    checkpoint_mode = "min"

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        dropout: float = 0.0,
        **backbone_kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_out = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.backbone = MambaModel(num_layers=num_layers, d_model=d_model, **backbone_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed_dropout(self.embedding(x))
        h = self.backbone(h)
        return self.lm_head(h)

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1))

    def compute_metrics(self, logits, y):
        loss = self.compute_loss(logits, y).detach()
        preds = logits.argmax(dim=-1)
        mask = (y != -100)
        token_acc = (preds[mask] == y[mask]).float().mean()
        perplexity = torch.exp(loss)
        return {"perplexity": perplexity, "token_acc": token_acc}
