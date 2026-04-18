import os
import torch
from torch.utils.data import DataLoader, Dataset

from dataloaders.data_registry import DATASET_REGISTRY
    

def get_dataloaders(
    dataset_name: str, 
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
    config = DATASET_REGISTRY[dataset_name.lower()]
    params = config.get_params(**dataset_kwargs) 
    data_module = config.dataset_class(**params) # Data factory class
    data_module.process() # Process data (normalization, formatting, etc...)
    dataset, num_classes = data_module.create_dataset() # Dataset creation 
    dataloaders = {}
    for subset, dataset in dataset.items():
        dataloaders[f"{subset}_loader"]= DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=(subset == 'train'),
            num_workers=num_workers, 
            pin_memory=pin_memory,
            collate_fn=config.collate_fn
            )

    data_sample = next(iter(dataloaders['train_loader']))
    input_sample = data_sample[0]   
    
    return {
        **dataloaders,
        "input_shape": list(input_sample.shape),
        "num_classes": num_classes
    }