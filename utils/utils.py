import os
import torch
import random
import torchinfo
import numpy as np
import lightning as L
from pathlib import Path
import json
import wandb
import importlib

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
    
def handle_wandb_login(config):
    """Prompts the user for W&B logging preference and handles the login."""
    print("\n--- Weights & Biases (W&B) Configuration ---")
    
    if config["username"]:
        usrname = config["username"]
        print(f"W&B username: {usrname}")
    else:
        usrname = input("Please enter your W&B username: ")
    
    if config["mode"] == 1:
        choice = str(config["mode"])
    else:
        print("Do you want to enable W&B online logging for this run?")
        print("(1) Log In / Use Existing Account")
        print("(2) Skip Logging (Run Offline)")
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

def print_config(config: dict, list: list):
    if "all" not in list:
        for param in list:
            if param not in config:
                raise ValueError(f"Parameter '{param}' does not exist in {config}.")
            print(f"   ✓ {param}: {config[param]}")
    else:
        for param, key in config.items():
            print(f"   ✓ {param}: {key}")
            
def load_config(name: str):
    module = importlib.import_module(f"experiments.{name}")
    return module.config

def compute_negative_penalty(model, penalty_type='hinge', margin=0.01, exclude=None):
    penalty = 0
    for name, param in model.named_parameters():
        
        if exclude is not None and exclude in name:
            continue
        
        if penalty_type == 'l1':
            negative_val = torch.clamp(param, max=0)
            penalty = penalty + torch.sum(negative_val)
            
        if penalty_type == 'l2':
            negative_val = torch.clamp(param, max=0)
            penalty = penalty + torch.sum(negative_val**2)
            
        if penalty_type == 'hinge':
            negative_val = torch.clamp(param - margin, max=0)
            penalty = penalty + torch.sum(negative_val**2)
            
    return penalty

def check_non_negativity(model, verbose=True):
    results={}
    total_params = 0
    total_negative = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        num_negative = (param.data < 0).sum().item()
        total_params += num_params
        total_negative += num_negative
        min = param.data.min().item()
        max = param.data.max().item()
        
        results[name] = {
            'total_params': num_params,
            'negative_params': num_negative,
            'ratio': num_negative / num_params,
            'min': min,
            'max': max
        }
        
        if verbose and num_negative > 0:
            print(f"❌ Parameter '{name}")
            print(f"\t {num_negative}/{num_params} negative parameters.")
            print(f"\t min: {min}, max: {max}")
        elif verbose and num_negative == 0:
            print(f"✔️ Parameter '{name}' has no negative values. ({num_params})")
            
    print("Overall Negative Weights Summary:")
    print(f"\t {total_negative}/{total_params} negative parameters.")
    print(f"\t Overall Ratio: {total_negative / total_params}")
    print(f"Negative Weights found in:")
    for name, stats in results.items():
        if stats['negative_params'] > 0:
            print(f" - {name}")
        
    return results