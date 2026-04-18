#!/usr/bin/env python3
"""
Analyze a model checkpoint for negative parameter values.

Usage:
  python analyze_checkpoint.py path/to/checkpoint.ckpt
  python analyze_checkpoint.py path/to/checkpoint.ckpt -v
  python analyze_checkpoint.py path/to/checkpoint.ckpt -e sc09_class --num-classes 10
"""
import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils import load_config
from utils.lightning import LightningMamba
from utils.non_negativity import check_non_negativity
from models.mamba_model import MambaModel


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a model checkpoint for negative parameter values"
    )
    parser.add_argument("ckpt_path", help="Path to .ckpt file")
    parser.add_argument(
        "-e", "--experiment", required=True,
        help="Experiment config to load model architecture from (module name in experiments/)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Print per-layer statistics"
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of output classes (auto-detected from checkpoint config if available)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_path):
        print(f"Error: Checkpoint file not found: {args.ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading experiment config: {args.experiment}")
    model_config = load_config(args.experiment)["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.ckpt_path} (device={device})")

    raw = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    num_classes = args.num_classes
    if num_classes is None and "experiment_config" in raw:
        num_classes = raw["experiment_config"].get("dataset", {}).get("num_classes")
    if num_classes is None:
        raise ValueError("Could not auto-detect num_classes. Pass --num-classes explicitly.")
    mamba_model = MambaModel(**model_config, d_out=num_classes)

    if "pytorch-lightning_version" in raw:
        # Full Lightning checkpoint
        model = LightningMamba.load_from_checkpoint(
            args.ckpt_path,
            model=mamba_model,
            optimizer=torch.optim.AdamW,
            loss_fn=torch.nn.CrossEntropyLoss(),
            map_location=device
        )
        target = model
    else:
        # Raw state dict (clipped_model.ckpt saved via torch.save)
        mamba_model.load_state_dict(raw)
        target = mamba_model

    check_non_negativity(target, verbose=args.verbose)


if __name__ == "__main__":
    main()
