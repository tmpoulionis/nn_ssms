"""
Implementation of standard Copying dataset.

Originally used in Arjovsky's Unitary RNN, maybe earlier?

Copied from https://github.com/state-spaces/s4
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np



def np_copying_data(L, M, A, batch_shape=()):
    """
    Generate a batch of copying-task examples using NumPy.

    Input layout:  [seq (M)] [zeros (L)] [markers (M)] 
    Output layout: [zeros (M+L)] [seq (M)]

    Args:
        L: Number of filler tokens
        M: Number of tokens to memorize
        A: Vocabulary size.
        batch_shape: Leading batch dimensions

    Returns:
        x: One-hot encoded input tensor of shape (*batch_shape, M+L+M, A).
        y: Target class-index tensor of shape (*batch_shape, M+L+M).
    """
    
    seq = np.random.randint(low=1, high=A-1, size=batch_shape+(M,))
    zeros_x = np.zeros(batch_shape+(L,))
    markers = (A-1) * np.ones(batch_shape+(M,)) # Markers prompt the model to produce the sequence it memorized
    zeros_y = np.zeros(batch_shape+(M+L,))

    x_ = np.concatenate([seq, zeros_x, markers], axis=-1)
    y_ = np.concatenate([zeros_y, seq], axis=-1)
    x = F.one_hot(torch.tensor(x_, dtype=torch.int64), A).float()
    y = torch.tensor(y_, dtype=torch.int64)
    return x, y

def torch_copying_data(L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
    """
    Generate a batch of copying-task examples using PyTorch.

    Input layout:  [zeros_x with tokens scattered (M+L)] [markers (M)]
    Output layout: [tokens (M)]  (model only predicts at marker positions)

    Args:
        L: Number of noise tokens
        M: Number of tokens to memorize
        A: Vocabulary size
        variable: If True, scatter tokens at random positions (selective copying).
        variable_length: If True, M is randomly choosen from a range [1, M]
        batch_shape: Leading batch dimensions.
        one_hot: If True, return x as one-hot float tensor; otherwise return raw integer indices.
        reverse: If True, reverse the target sequence (model must output tokens in reverse order).

    Returns:
        x: Input tensor of shape (*batch_shape, M+L+M [, A if one_hot]).
        y: Target token tensor of shape (*batch_shape, M).
    """
    
    if variable_length:
        M = int(random.random() * M) + 1
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    
    if variable:
        total_batch = int(np.prod(batch_shape))
        inds = torch.stack([
            torch.randperm(L+M)[:M]
            for _ in range(total_batch)
            ], 0) # Choose random indeces to place tokens, in an array (L+M,) of zeroes
        inds = inds.reshape(batch_shape+(M,))
        inds, _ = inds.sort()
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))
    zeros_x = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    zeros_x.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)

    x_ = torch.cat([zeros_x, markers], dim=-1)
    y_ = torch.cat([tokens], dim=-1)
    if reverse: y_ = y_.flip(-1)
    if one_hot: x = F.one_hot(x_, A).float()
    else: x = x_
    y = y_
    return x, y

def torch_copying_lag_data(L, M, A, batch_shape=()):
    """
    Generate a lag/delay copying task: the target is the input shifted right by M steps.

    The model sees a random token sequence of length L and must output, at each position t,
    the token that appeared M steps earlier (i.e. y[t] = x[t - M], with zero-padding at the start).

    Args:
        L: Length of the input (and output) sequence
        M: Lag — how many steps into the past the model must recall
        A: Vocabulary size
        batch_shape: Leading batch dimensions

    Returns:
        x: Input token tensor of shape (*batch_shape, L).
        y: Lagged target tensor of shape (*batch_shape, L), zero-padded for the first M positions.
    """
    x = torch.randint(low=1, high=A-1, size=batch_shape+(L,))
    y = F.pad(x, (M, 0))[..., :L]
    return x, y

class CopyingTrainDataset(Dataset):
    """
    Training dataset generation for the copying task.
    
        Args:
            L: Number of noise tokens
            M: Number of tokens to memorize
            A: Vocab size
            samples: Number of samples
            lag: If True, use the lag/delay task instead of the copying task
            variable: If True, scatter tokens at random positions (selective copying)
            variable_length: If True, M is randomly choosen from a range [1, M]
            one_hot: If True, return x as a one-hot float tensor
            reverse: If True, the target sequence is reversed
    """

    def __init__(self, L, M, A, samples, lag=False, variable=False, variable_length=False, one_hot=False, reverse=False):

        super().__init__()
        self.L = L
        self.M = M
        self.A = A
        self.samples = samples
        self.variable = variable
        self.variable_length = variable_length
        self.one_hot = one_hot
        self.lag = lag
        self.reverse = reverse

    def __getitem__(self, idx):
        assert 0 <= idx < self.samples
        if self.lag:
            x, y = torch_copying_lag_data(self.L, self.M, self.A)
        else:
            x, y = torch_copying_data(self.L, self.M, self.A, variable=self.variable, variable_length=self.variable_length, one_hot=self.one_hot, reverse=self.reverse)
        return x, y

    def __len__(self):
        return self.samples


class CopyingEvalDataset(torch.utils.data.TensorDataset):
    """
    Pre-generated evaluation dataset for the copying task.
    
        Args:
            L: Number of noise tokens
            M: Number of tokens to memorize
            A: Vocab size
            samples: Number of samples
            lag: If True, use the lag/delay task instead of the copying task
            variable: If True, scatter tokens at random positions (selective copying)
            variable_length: If True, M is randomly choosen from a range [1, M]
            one_hot: If True, return x as a one-hot float tensor
            reverse: If True, the target sequence is reversed
    """

    def __init__(self, L, M, A, samples, lag=None, variable=False, variable_length=False, one_hot=False, reverse=False):

        self.L = L
        self.M = M
        self.A = A
        self.samples = samples
        if lag:
            all_x, all_y = torch_copying_lag_data(self.L, self.M, self.A, batch_shape=(self.samples,))
        else:
            all_x, all_y = torch_copying_data(self.L, self.M, self.A, batch_shape=(self.samples,), variable=variable, variable_length=False, one_hot=one_hot, reverse=reverse)
        super().__init__(all_x, all_y)

def copying_static_dataset(L, M, A, variable, samples):
    
    all_x, all_y = torch_copying_data(L, M, A, variable, batch_shape=(samples,))
    print("Constructing Copying dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds