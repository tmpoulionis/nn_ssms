import os
from typing import Dict
import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders.data_registry import DATASET_REGISTRY

def get_dataloaders(
    dataset_name: str,
    root: str='./data', 
    batch_size: int=32, 
    num_workers: int=4,
    pin_memory: bool=True,
    **dataset_kwargs
    ):

    """
    Initializes and returns PyTorch DataLoaders for a specified dataset.

    Args:
        dataset_name (str): The name of the dataset (key in DATASET_REGISTRY)
        root (str): Directory where dataset data is stored/downloaded
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses to use for data loading
        pin_memory (bool): If True, use CUDA pinned memory for faster transfers to GPU
        **dataset_kwargs: Extra keyword arguments passed to the underlying create_dataset function

    Returns:
        dict: A dictionary containing:
            - DataLoaders (train, valid, test) or (train, test)
            - "input_shape" (list): The shape of a single input sample
            - "num_classes" (int): The total number of classes in the dataset
    """
    if dataset_name.lower() not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    # Retrieve data subsets from the registry
    datasets, num_classes = create_dataset(dataset_name=dataset_name, **dataset_kwargs)
    data_config = DATASET_REGISTRY[dataset_name.lower()]
    
    dataloaders = {}
    for subset, dataset in datasets.items():
        dataloaders[f"{subset}_loader"]= DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=(subset == 'train'),
            num_workers=num_workers, 
            pin_memory=pin_memory,
            collate_fn=data_config.collate_fn
            )

    data_sample = next(iter(dataloaders['train_loader']))
    input_sample = data_sample[0]   
    
    return {
        **dataloaders,
        "input_shape": list(input_sample.shape),
        "num_classes": num_classes
    }

def create_dataset(dataset_name: str, **kwargs) -> Dict[str, Dataset]:
    """
    Factory function to create dataset instances and determine the number of classes for the specified dataset in DATASET_REGISTRY
    
    Args:
        dataset_name (str): Name of the dataset
        **kwargs: Parameters used to override the default configuration
    
    Returns:
        Dataset instances, number of classes
    """
    
    if dataset_name.lower() not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
        
    config = DATASET_REGISTRY[dataset_name.lower()]
    params = config.get_params(**kwargs)
    subsets = params.pop("subset")
    
    if len(subsets) == 3:
        datasets = {
            'train': config.dataset_class(subset=subsets[0], **params),
            'valid': config.dataset_class(subset=subsets[1], **params),
            'test': config.dataset_class(subset=subsets[2], **params),
        }
    else:
        datasets = {
            'train': config.dataset_class(subset=subsets[0], **params),
            'test': config.dataset_class(subset=subsets[1], **params),
        }
        
    if hasattr(config.dataset_class, 'num_classes'):
        num_classes = datasets['train'].num_classes
    else:
        num_classes = None
        
    return datasets, num_classes