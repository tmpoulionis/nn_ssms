import torch
import random
import torchinfo
import numpy as np
import lightning as L
from pathlib import Path
import json

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Set Lightning seed
    L.seed_everything(seed, workers=True)
    
    print(f"Random seed set to {seed}")
    
def count_parameters(model):
    """
    Count model parameters.
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }
    
def model_summary(model):
    torchinfo.summary(model)

