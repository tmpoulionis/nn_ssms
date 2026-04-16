"""
Evaluate a trained MambaModel checkpoint and its non-negative isomorphic
transformation side by side on the test set.

Usage:
    python evaluate_nn.py -c checkpoints/<run>/best-epoch=XX-val_acc=X.XXXX.ckpt 
    python evaluate_nn.py -c <path.ckpt> --a_min -5 --a_max 5   # override bounds
"""

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

import dataloaders.data as data
from models.mamba_model import MambaModel
from nnt.transformation import transform_to_nn


@torch.no_grad()
def evaluate(model, dataloader, num_classes, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total, total_loss


def main():
    parser = argparse.ArgumentParser(description="Compare original vs NN-isomorphic model on test set.")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument("--a_min", type=float, default=-10, help="Override a_min for the isomorphic transform.")
    parser.add_argument("--a_max", type=float, default=10, help="Override a_max for the isomorphic transform.")
    parser.add_argument("--nn_validation", "-n", default=None, help="Run non-negativity validation on the NN model. Options: 's': summary, 'p': all-passed, 'f': failures-only.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint and extract config ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["experiment_config"]
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    nn_validation = args.nn_validation

    a_min = args.a_min if args.a_min is not None else model_cfg.get("a_min", 0.0)
    a_max = args.a_max if args.a_max is not None else model_cfg.get("a_max", 1.0)

    # --- Build dataloader ---
    print("Loading test set...")
    dataset = data.get_dataloaders(**dataset_cfg)
    test_loader = dataset["test_loader"]
    num_classes = dataset["num_classes"]

    # --- Reconstruct original model and load weights ---
    print("Building model...")
    model = MambaModel(**model_cfg, d_out=num_classes)

    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)

    # --- Evaluate original ---
    print("\n--- Original Model ---")
    orig_acc, orig_loss = evaluate(model, test_loader, num_classes, device)
    print(f"  Accuracy: {orig_acc:.4%}")
    print(f"  Loss:     {orig_loss:.6f}")

    # --- Transform to non-negative isomorphic model ---
    model_nn = transform_to_nn(model, a_min=a_min, a_max=a_max)
    model_nn.to(device)

    # --- Evaluate NN model ---
    print("\n--- NN Isomorphic Model ---")
    nn_acc, nn_loss = evaluate(model_nn, test_loader, num_classes, device)
    print(f"  Accuracy: {nn_acc:.4%}")
    print(f"  Loss:     {nn_loss:.6f}")

    # --- Comparison ---
    print("\n" + "=" * 50)
    print(f"{'Metric'} {'Original'} {'NN'} {'Diff'}")
    print("-" * 50)
    print(f"{'Accuracy':<12} {orig_acc:>11.4%} {nn_acc:>11.4%} {nn_acc - orig_acc:>+11.4%}")
    print(f"{'Loss':<12} {orig_loss:>12.6f} {nn_loss:>12.6f} {nn_loss - orig_loss:>+12.6f}")
    print("=" * 50)

    if abs(nn_acc - orig_acc) < 1e-6:
        print("Identical accuracy — transformation is correct.")
    else:
        print(f"Accuracy difference: {abs(nn_acc - orig_acc):.2e}")

    if nn_validation:
        from nnt.validate_nn import validate_non_negativity
        sample_input, _ = next(iter(test_loader))
        sample_input = sample_input.to(device)
        report = validate_non_negativity(model_nn, sample_input, atol=0)
        
        if nn_validation == "s":
            print("\nNon-negativity validation report (summary):")
            print(report.summary())
        elif nn_validation == "p":
            print("\nNon-negativity validation report (all-passed):")
            print(report.all_passed())
        elif nn_validation == "f":
            print("\nNon-negativity validation report (failures-only):")
            print(report.failures_only())
        else:
            print(f"Invalid nn_validation option: {nn_validation}. Use 's': summary, 'p': all-passed, or 'f': failures-only.")
    
if __name__ == "__main__":
    main()
