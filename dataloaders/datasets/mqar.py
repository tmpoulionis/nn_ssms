# Multi-Query Associative Recall synthetic task.
# Generation logic adapted from https://github.com/HazyResearch/zoology
# (Arora, Eyuboglu et al., "Zoology: Measuring and improving recall in efficient language models.")

import numpy as np
import torch
from torch.utils.data import Dataset


def _generate_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int = 8,
    power_a: float = 0.01,
    num_passes: int = 1,
    random_non_queries: bool = True,
):
    """
    Generates MQAR (key, value) recall sequences.

    Example (num_kv_pairs=2, input_seq_len=16, random_non_queries=False):
            Key   Val  Key  Val            Query                         Query
    Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0
    Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

    The -100 labels are ignored by the loss function and metrics.

    Args:
        vocab_size: Vocabulary size. Keys drawn from [1, vocab//2), values from [vocab//2, vocab).
        num_examples: Number of samples to generate.
        input_seq_len: Length of the input sequence (must be even).
        seed: RNG seed.
        num_kv_pairs: Number of unique key/value pairs per example.
        power_a: Power-law exponent controlling query position distribution
            (small values bias queries toward early positions; 1.0 = uniform).
        num_passes: Number of times the KV prefix is repeated (JRT protocol).
        random_non_queries: If True, replace non-query 0 tokens with random vocab ids.

    Returns:
        (inputs, labels): torch.int64 tensors, each of shape (num_examples, input_seq_len).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)

    context_size = num_kv_pairs * 2 * num_passes

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])

    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return inputs, labels


class MQARData(Dataset):
    """Pre-generated MQAR tensors for a single split."""

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class MQARDataset:
    """Factory matching the nn_mamba pipeline contract (process + create_dataset)."""

    def __init__(
        self,
        n_train: int = 100_000,
        n_valid: int = 3_000,
        n_test: int = 3_000,
        vocab_size: int = 8192,
        input_seq_len: int = 64,
        num_kv_pairs: int = 8,
        power_a: float = 0.01,
        num_passes: int = 1,
        random_non_queries: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        gen_kwargs = dict(
            vocab_size=vocab_size,
            input_seq_len=input_seq_len,
            num_kv_pairs=num_kv_pairs,
            power_a=power_a,
            num_passes=num_passes,
            random_non_queries=random_non_queries,
        )
        self.train = MQARData(*_generate_mqar(num_examples=n_train, seed=seed,     **gen_kwargs))
        self.valid = MQARData(*_generate_mqar(num_examples=n_valid, seed=seed + 1, **gen_kwargs))
        self.test  = MQARData(*_generate_mqar(num_examples=n_test,  seed=seed + 2, **gen_kwargs))
        self.vocab_size = vocab_size

    def process(self):
        pass

    def create_dataset(self):
        return (
            {"train": self.train, "valid": self.valid, "test": self.test},
            self.vocab_size,
        )
