import os
import torch
import random
import torchinfo
import numpy as np
import lightning as L
from pathlib import Path
import json
import wandb
from config import get_config

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
    config = get_config()
    """Prompts the user for W&B logging preference and handles the login."""
    print("\n--- Weights & Biases (W&B) Configuration ---")
    
    if config["wandb"]["username"]:
        usrname = config["wandb"]["username"]
        print(f"W&B username: {usrname}")
    else:
        usrname = input("Please enter your W&B username: ")
    
    if config["wandb"]["mode"] == 1:
        choice = str(config["wandb"]["mode"])
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

## --------- DEBUG NAN VALUES --------------
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
import itertools
import warnings
import torchvision.models as models
import random

# adapted from https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py

class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def fmt(t: object, print_stats=False) -> str:
    if isinstance(t, torch.Tensor):
        s = f"torch.tensor(..., size={tuple(t.shape)}, dtype={t.dtype}, device='{t.device}')"
        if print_stats:
            s += f" [with stats min={t.min()}, max={t.max()}, mean={t.mean()}]"
        return Lit(s)
    else:
        return t


class NaNErrorMode(TorchDispatchMode):
    def __init__(
        self, enabled=True, raise_error=False, print_stats=True, print_nan_index=False
    ):
        self.enabled = enabled
        # warning or error
        self.raise_error = raise_error
        # print min/max/mean stats
        self.print_stats = print_stats
        # print indices of invalid values in output
        self.print_nan_index = print_nan_index

    def __torch_dispatch__(self, func, types, args, kwargs):
        out = func(*args, **kwargs)
        if self.enabled:
            if isinstance(out, torch.Tensor):
                if not torch.isfinite(out).all():
                    # fmt_partial = partial(fmt, self.print_stats)
                    fmt_lambda = lambda t: fmt(t, self.print_stats)
                    fmt_args = ", ".join(
                        itertools.chain(
                            (repr(tree_map(fmt_lambda, a)) for a in args),
                            (
                                f"{k}={tree_map(fmt_lambda, v)}"
                                for k, v in kwargs.items()
                            ),
                        )
                    )
                    msg = f"NaN outputs in out = {func}({fmt_args})"
                    if self.print_nan_index:
                        msg += f"\nInvalid values detected at:\n{(~out.isfinite()).nonzero()}"
                    if self.raise_error:
                        raise RuntimeError(msg)
                    else:
                        warnings.warn(msg)

        return out
