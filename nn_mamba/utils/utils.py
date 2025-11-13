import os
import torch
import random
import torchinfo
import numpy as np
import lightning as L
from pathlib import Path
import json
import wandb

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
    return torchinfo.summary(model)

def format_time(seconds):
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
    
def handle_wandb_login():
    """Prompts the user for W&B logging preference and handles the login."""
    print("\n--- Weights & Biases (W&B) Configuration ---")
    print("Do you want to enable W&B logging for this run?")
    print("(1) Log In / Use Existing Account")
    print("(2) Skip Logging (Run Offline)")
    
    usrname = input("Please enter your W&B username: ")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":        
        try:
            wandb.login(relogin=False)
            print("W&B login successful!")
        except Exception as e:
            print(f"W&B login failed: {e}. Running offline instead.")
    elif choice == "2":
        print("W&B logging skipped. The run will proceed without tracking.")
        # Set WANDB_MODE to offline to prevent automatic logging
        os.environ["WANDB_MODE"] = "offline"
    else:
        print("Invalid choice. Running offline.")
        os.environ["WANDB_MODE"] = "offline"
    
    return usrname