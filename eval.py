"""
Evaluate a trained checkpoint on the test set (and optionally on validation
and/or train sets). Uses the experiment config saved inside the checkpoint,
so no --experiment flag is needed.

Usage:
    python eval.py --ckpt "checkpoints/<run>/best-epoch=XX-<metric>=X.XXXX.ckpt"
    python eval.py --ckpt <path> --valid --train           # + overfitting check
    python eval.py --ckpt <path> --nn                      # + NN isomorphic transform & validation
    python eval.py --ckpt <path> --nn --a_min -5 --a_max 5 # override NN transform bounds
"""

import argparse

import torch
import lightning as L

import dataloaders.data as data
from utils.lightning import LightningMamba
from nnt.transformation import transform_to_nn
from nnt.validate_nn import validate_non_negativity


def run_eval(lightning_module, trainer, loaders, args):
    """Test (always), then validation/train if requested."""
    print("\n---------- Test set ----------")
    trainer.test(lightning_module, loaders["test"])

    if args.valid:
        print("\n---------- Validation set ----------")
        trainer.validate(lightning_module, loaders["valid"])

    if args.train:
        print("\n---------- Train set ----------")
        trainer.validate(lightning_module, loaders["train"])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on test/val/train, optionally via the non-negative isomorphic transform."
    )
    parser.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file.")
    parser.add_argument("--valid", action="store_true", help="Also evaluate on the validation set.")
    parser.add_argument("--train", action="store_true", help="Also evaluate on the train set.")
    parser.add_argument("--nn", action="store_true", help="Transform to non-negative isomorphic model, evaluate, and report non-negativity.")
    parser.add_argument("--a_min", type=float, default=-10.0, help="a_min bound for --nn transform (default: -10).")
    parser.add_argument("--a_max", type=float, default=10.0, help="a_max bound for --nn transform (default: +10).")
    args = parser.parse_args()

    # ------ Load checkpoint + config ------
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["experiment_config"]
    model_cls = config["model_class"]
    print(f"Model class:  {model_cls.__name__}")
    print(f"Dataset:      {config['dataset']['dataset_name']}")

    # ------ Dataloaders ------
    print("\nBuilding dataloaders...")
    dataset = data.get_dataloaders(**config["dataset"])
    loaders = {
        "train": dataset["train_loader"],
        "valid": dataset["valid_loader"],
        "test":  dataset["test_loader"],
    }

    # ------ Model + Lightning wrapper ------
    print("Building model & loading weights...")
    model = model_cls(**config["model"])
    lightning_module = LightningMamba.load_from_checkpoint(
        args.ckpt, model=model, map_location="cpu",
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ------ Original model ------
    print("\n" + "=" * 70)
    print("ORIGINAL MODEL")
    print("=" * 70)
    run_eval(lightning_module, trainer, loaders, args)

    # ------ NN isomorphic model ------
    if args.nn:
        print("\n" + "=" * 70)
        print(f"NON-NEGATIVE ISOMORPHIC MODEL  (a_min={args.a_min}, a_max={args.a_max})")
        print("=" * 70)

        model_nn = transform_to_nn(
            lightning_module.model, a_min=args.a_min, a_max=args.a_max,
        )
        lightning_module.model = model_nn

        run_eval(lightning_module, trainer, loaders, args)

        # Non-negativity validation (failures-only)
        print("\n---------- Non-Negativity Validation ----------")
        sample_input, _ = next(iter(loaders["test"]))
        device = next(model_nn.parameters()).device
        sample_input = sample_input.to(device)
        report = validate_non_negativity(model_nn, sample_input, atol=0)
        print(report.failures_only_summary())


if __name__ == "__main__":
    main()
