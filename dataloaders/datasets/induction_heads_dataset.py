"""
Synthetic Induction Heads (Associative Recall) Dataset.

From: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
      Olsson et al. (2022) "In-context Learning and Induction Heads"

A random token pattern is generated and repeated: [pattern | pattern].
The model must learn to predict the second half by recalling associations
from the first half. This requires a 2-layer "induction head" circuit:
  - Layer 1: shifts token information by one position ("previous token" head)
  - Layer 2: matches current token to earlier occurrence, recalls what followed

This cannot be solved by positional patterns alone since the content is
random each sample — it directly tests content-aware selective reasoning.
"""

import torch
from torch.utils.data import Dataset


SUBSET_SIZES = {
    'train': 50_000,
    'valid': 5_000,
    'test': 5_000,
}


class InductionHeadsDataset(Dataset):

    def __init__(self, subset, seq_len=256, vocab_size=16, num_samples=None, **kwargs):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.subset = subset
        self.num_samples = num_samples or SUBSET_SIZES.get(subset, 10_000)

        # Paper protocol: valid/test sets are fixed, train is fresh random each step
        if subset in ('valid', 'test'):
            half_len = self.seq_len // 2
            patterns = torch.randint(0, self.vocab_size, (self.num_samples, half_len))
            seqs = torch.cat([patterns, patterns], dim=1)  # (N, seq_len)
            self.data = seqs[:, :-1]    # (N, seq_len-1)
            self.targets = seqs[:, 1:]  # (N, seq_len-1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.subset in ('valid', 'test'):
            return self.data[idx], self.targets[idx]

        half_len = self.seq_len // 2
        pattern = torch.randint(0, self.vocab_size, (half_len,))
        seq = torch.cat([pattern, pattern])
        data = seq[:-1].clone()
        target = seq[1:].clone()
        return data, target
    