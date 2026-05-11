import torch
import torch.nn as nn
import torch.nn.functional as F


class IsomorphicEmbedding(nn.Module):
    """
    Equivalent to nn.Embedding; uses two non-negative lookup tables.

    W = W_pos - W_neg_abs
    y = F.embedding(x, W_pos) - F.embedding(x, W_neg_abs)

    Both lookup tables and both rail outputs are non-negative — suitable for
    photonic hardware, which cannot represent negative weights directly.
    """

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        W = emb.weight.data
        self.W_pos     = nn.Parameter(W.clamp(min=0))
        self.W_neg_abs = nn.Parameter((-W).clamp(min=0))

        self.num_embeddings     = emb.num_embeddings
        self.embedding_dim      = emb.embedding_dim
        self.padding_idx        = emb.padding_idx
        self.max_norm           = emb.max_norm
        self.norm_type          = emb.norm_type
        self.scale_grad_by_freq = emb.scale_grad_by_freq
        self.sparse             = emb.sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = F.embedding(
            x, self.W_pos, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
        neg = F.embedding(
            x, self.W_neg_abs, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
        return pos - neg
