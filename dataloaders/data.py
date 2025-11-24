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
    Factory function to create dataset instances.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Override parameters for the dataset
    
    Returns:
        Dataset instances
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