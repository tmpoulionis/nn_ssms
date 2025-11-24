from torch.nn.utils.rnn import pad_sequence
from dataloaders.datasets.sc_dataset import SCDataset, sc_custom_collate
from dataloaders.datasets.hg38_dataset import HG38Dataset
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type
from torch.utils.data import Dataset

@dataclass
class DatasetConfig:
    """Configuration for dataset creation"""
    dataset_class: Type[Dataset]
    default_params: Dict[str, Any]
    collate_fn: Callable = None

    def get_params(self, **override_params) -> Dict[str, Any]:
        params = self.default_params.copy()
        params.update(override_params)
        return params
        
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "speechcommands": DatasetConfig(
        dataset_class=SCDataset,
        collate_fn=sc_custom_collate,
        default_params={
            "root": "./data/speechcommands",
            "subset": ["training", "validation", "testing"],
            "mel_transform": True,
            "n_mels": 64,
            "n_fft": 400,
            "hop_length": 150
        }
    ),
    "sc09": DatasetConfig(
        dataset_class=SCDataset,
        collate_fn=sc_custom_collate,
        default_params={
            "root": "./data/sc09",
            "subset": ["training", "validation", "testing"],
            "mel_transform": True,
            "n_mels": 64,
            "n_fft": 400,
            "hop_length": 150,
            "filter_labels": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        }
    ),
    "hg38": DatasetConfig(
        dataset_class=HG38Dataset,
        default_params={
            "root": "./data/hg38",
            "subset": ['train', 'valid', 'test'],
            "max_length": 2**17,
            "tokenizer_name": "char",
            "add_eos": True
        }
    )
}
