"""
Helper utilities for checkpoint analysis.
"""
import os
import torch
from utils.utils import load_config
from utils.lightning import LightningMamba
from utils.non_negativity import check_non_negativity
from models.mamba_model import MambaModel


def get_model_config(experiment):
    """
    Returns MODEL_CONFIG from the given experiment configuration.

    Args:
        experiment (str): Experiment config name (module in experiments/)

    Returns:
        dict: Model configuration dictionary
    """
    config = load_config(experiment)
    return config["model"]


def find_checkpoint_files(folder_path):
    """
    Find unclipped and clipped checkpoint files in the given folder.

    Args:
        folder_path (str): Path to checkpoint folder

    Returns:
        tuple: (unclipped_path, clipped_path)

    Raises:
        FileNotFoundError: If checkpoint files are missing or invalid
    """
    # Check if folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Checkpoint folder not found: {folder_path}")

    # Find unclipped checkpoint
    unclipped_path = os.path.join(folder_path, "unclipped.ckpt")
    if not os.path.exists(unclipped_path):
        raise FileNotFoundError(
            f"unclipped.ckpt not found in {folder_path}\n"
            f"Expected file: {unclipped_path}"
        )

    # Find clipped checkpoint (best-epoch=*.ckpt)
    # Use os.listdir instead of glob to avoid issues with special characters in folder path
    clipped_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith("best-epoch=") and f.endswith(".ckpt")
    ]

    if len(clipped_files) == 0:
        raise FileNotFoundError(
            f"No clipped checkpoint (best-epoch=*.ckpt) found in {folder_path}"
        )
    elif len(clipped_files) > 1:
        raise FileNotFoundError(
            f"Multiple clipped checkpoints found in {folder_path}:\n" +
            "\n".join(f"  - {os.path.basename(f)}" for f in clipped_files) +
            "\nExpected exactly one best-epoch=*.ckpt file"
        )

    clipped_path = clipped_files[0]

    return unclipped_path, clipped_path


def aggregate_results(check_results):
    """
    Aggregate per-layer results from check_non_negativity.

    Args:
        check_results (dict): Results dictionary from check_non_negativity

    Returns:
        dict: Aggregated statistics with keys:
            - total_params: Total number of parameters
            - negative_params: Number of negative parameters
            - ratio: Ratio of negative to total parameters
            - max_negative: Most negative value across all layers
    """
    total_params = 0
    negative_params = 0
    min_values = []

    for layer_name, layer_result in check_results.items():
        total_params += layer_result['total_params']
        negative_params += layer_result['negative_params']
        min_values.append(layer_result['min'])

    ratio = negative_params / total_params if total_params > 0 else 0.0
    max_negative = min(min_values) if min_values else 0.0

    return {
        'total_params': total_params,
        'negative_params': negative_params,
        'ratio': ratio,
        'max_negative': max_negative
    }


def load_checkpoint_and_analyze(ckpt_path, model_config, verbose=False):
    """
    Load a checkpoint and analyze it for negative values.

    Args:
        ckpt_path (str): Path to checkpoint file
        model_config (dict): Model configuration dictionary
        verbose (bool): If True, print detailed per-layer statistics

    Returns:
        tuple: (model, aggregated_results, full_results)
            - model: Loaded LightningMamba model
            - aggregated_results: Dict with total_params, negative_params, ratio, max_negative
            - full_results: Full check_non_negativity results dict
    """
    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    model = LightningMamba.load_from_checkpoint(
        ckpt_path,
        model=MambaModel(**model_config, d_out=10),
        optimizer=torch.optim.AdamW,
        loss_fn=torch.nn.CrossEntropyLoss(),
        map_location=device
    )

    # Run check_non_negativity
    full_results = check_non_negativity(model, verbose=verbose)

    # Aggregate results
    aggregated = aggregate_results(full_results)

    return model, aggregated, full_results
