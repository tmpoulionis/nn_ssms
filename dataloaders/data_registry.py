from torch.nn.utils.rnn import pad_sequence
from dataloaders.datasets.sc_dataset import SCDataset, sc_custom_collate
from dataloaders.datasets.hg38_dataset import HG38Dataset
from dataloaders.datasets.induction_heads import InductionHeadsDataset
from dataloaders.datasets.selective_copying import SelectiveCopyingDataset
from dataloaders.datasets.cifar10 import sCIFAR10Dataset
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
            "max_length": 2**17,
            "add_eos": True
        }
    ),
    "induction_heads": DatasetConfig(
        dataset_class=InductionHeadsDataset,
        default_params={
            "seq_len": 256,
            "vocab_size": 16,
        }
    ),
    "selective_copying": DatasetConfig(
        dataset_class=SelectiveCopyingDataset,
        default_params={
            "n_train": 10000,
            "n_val": 1000,
            "n_test":1000,
            "L": 1000,
            "M": 10,
            "A": 10,
            "variable": True,
        }
    ),
    "scifar10": DatasetConfig(
        dataset_class=sCIFAR10Dataset,
        default_params={
            "root": "./data/cifar10",
            "download": True,
        }
    ),
}
