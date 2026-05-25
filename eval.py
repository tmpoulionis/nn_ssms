"""
Evaluate a trained checkpoint on the test set (and optionally on validation
and/or train sets). Uses the experiment config saved inside the checkpoint,
so no --experiment flag is needed.

Usage:
    python eval.py --ckpt "checkpoints/<run>/best-epoch=XX-<metric>=X.XXXX.ckpt"
    python eval.py --ckpt <path> --valid --train           # + overfitting check
    python eval.py --ckpt <path> --nn                      # + NN isomorphic transform & validation
    python eval.py --ckpt <path> --nn --a_min -5 --a_max 5 # override NN transform bounds
    python eval.py --ckpt <path> --quant                   # PTQ at default 8-bit
    python eval.py --ckpt <path> --quant --bits 4          # sweep bit resolution (8/6/5/4/3/...)
    python eval.py --ckpt <path> --quant --calib-batches 32 # more train batches for quant calibration
    python eval.py --ckpt <path> --nn --quant --bits 4     # quantize both original and NN isomorphic model
"""

import argparse
from itertools import islice

import torch
from torch import nn
import lightning as L

import dataloaders.data as data
from utils.lightning import LightningMamba
from utils.quantization import QuantizationManager
from utils.utils import model_summary
from nnt.nn_conv1d import IsomorphicConv1D
from nnt.nn_linear import IsomorphicLinear
from nnt.transformation import transform_to_nn
from nnt.validate_nn import validate_non_negativity


# Fallback quant config when the checkpoint carries none. Modeled on
# experiments/induction_heads.py; bit widths are filled in by build_qcfg().
_DEFAULT_QCFG = {
    "enabled": True,
    "weights_symmetric": False,
    "activations_symmetric": False,
    "per_channel_weights": True,
    "observer_momentum": 0.01,
    "quantize_embeddings": False,
    "skip_modules": ["norm"],
    "tensors": {"input": True, "weight": True, "bias": True, "output": True},
    "eval_quant": True,
}


def build_qcfg(config, bits):
    """Quant config for PTQ: seed from the checkpoint's config if it carries one,
    else a sensible default; force enabled + the requested bit width on both
    weights and activations (single resolution knob)."""
    qcfg = dict(config.get("quantization") or _DEFAULT_QCFG)
    qcfg["enabled"] = True
    qcfg["weight_bits"] = bits
    qcfg["activation_bits"] = bits
    qcfg["eval_quant"] = True
    return qcfg


def print_quant_wraps(model, manager, qcfg, label):
    """Print every quant-eligible layer in `model` and whether it got wrapped.
    If a layer was filtered out, show why (skip_modules pattern or
    quantize_embeddings=False) so silent misses are visible."""
    skip = qcfg.get("skip_modules", [])
    quantize_embed = qcfg.get("quantize_embeddings", False)
    candidates = (nn.Linear, nn.Conv1d, nn.Embedding, IsomorphicLinear, IsomorphicConv1D)
    rows = []
    for name, layer in model.named_modules():
        if not isinstance(layer, candidates):
            continue
        key = name.replace(".", "__") or "_root"
        if key in manager.wraps:
            status = type(manager.wraps[key]).__name__
        else:
            skip_hit = next((s for s in skip if s in name), None)
            if skip_hit is not None:
                status = f"SKIPPED (skip_modules ~ '{skip_hit}')"
            elif isinstance(layer, nn.Embedding) and not quantize_embed:
                status = "SKIPPED (quantize_embeddings=False)"
            else:
                status = "NOT WRAPPED"
        rows.append((name or "<root>", type(layer).__name__, status))
    print(f"\n[quant] {label} — {sum(1 for r in rows if 'SKIPPED' not in r[2] and 'NOT' not in r[2])}/{len(rows)} eligible layers wrapped:")
    if not rows:
        return
    name_w = max(len(r[0]) for r in rows)
    type_w = max(len(r[1]) for r in rows)
    for name, ltype, status in rows:
        print(f"  {name:<{name_w}}  {ltype:<{type_w}}  {status}")


def quantize_and_calibrate(model, qcfg, calib_loader, device, n_batches):
    """Post-training quantization: wrap layers, calibrate observers on a few
    train batches, then freeze ranges and switch fake-quant on. Returns the
    QuantizationManager — the caller MUST keep it alive (its forward hooks are
    what quantize the model during trainer.test)."""
    manager = QuantizationManager(model, qcfg)
    assert len(manager.wraps) > 0, (
        "QuantizationManager wrapped 0 layers — quantization would be a silent "
        "no-op. Check skip_modules and that the model exposes quantizable layers."
    )
    manager.to(device)
    model.to(device).eval()

    manager.set_quant_mode(True)
    manager.enable_observer()
    manager.disable_fake_quant()                       # calibration only
    with torch.no_grad():
        for batch in islice(calib_loader, n_batches):
            model(batch[0].to(device))
    manager.disable_observer()                          # freeze ranges
    manager.enable_fake_quant()
    return manager


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
    parser.add_argument("--quant", action="store_true", help="Post-training fake-quantize the model(s) before evaluation.")
    parser.add_argument("--bits", type=int, default=8, help="Bit width for --quant (weights + activations). Default: 8.")
    parser.add_argument("--calib-batches", type=int, default=100, help="Train batches used to calibrate quant ranges (default: 10).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the torchinfo parameter summary for each model.")
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

    # ------ Post-training quantization ------
    qcfg = None
    orig_mgr = None
    q_device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quant:
        qcfg = build_qcfg(config, args.bits)
        # Drop any manager restored from training hyperparams (avoids stale/double hooks).
        if getattr(lightning_module, "qat_manager", None) is not None:
            lightning_module.qat_manager.set_quant_mode(False)
            lightning_module.qat_manager = None
        orig_mgr = quantize_and_calibrate(
            lightning_module.model, qcfg, loaders["train"], q_device, args.calib_batches
        )
        print(f"\n[quant] {args.bits}-bit fake-quant — wrapped {len(orig_mgr.wraps)} "
              f"layers (calibrated on {args.calib_batches} train batches).")
        if args.verbose:
            print_quant_wraps(lightning_module.model, orig_mgr, qcfg, "original model")

    # ------ Original model ------
    print("\n" + "=" * 70)
    print("ORIGINAL MODEL")
    print("=" * 70)
    if args.verbose:
        print("\n---------- Parameter Summary ----------")
        model_summary(lightning_module.model)
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

        if args.verbose:
            print("\n---------- Parameter Summary ----------")
            model_summary(model_nn)

        if args.quant:
            if orig_mgr is not None:
                orig_mgr.set_quant_mode(False)   # its hooks point at the replaced layers
            nn_mgr = quantize_and_calibrate(
                model_nn, qcfg, loaders["train"], q_device, args.calib_batches
            )
            print(f"\n[quant] {args.bits}-bit fake-quant on NN model — wrapped "
                  f"{len(nn_mgr.wraps)} layers.")
            if args.verbose:
                print_quant_wraps(model_nn, nn_mgr, qcfg, "NN isomorphic model")

        run_eval(lightning_module, trainer, loaders, args)

        # Non-negativity validation (failures-only)
        print("\n---------- Non-Negativity Validation ----------")
        sample_input, _ = next(iter(loaders["test"]))
        if args.quant:
            # Lightning moves the model back to CPU after test, but nn_mgr's
            # fake-quant buffers live on q_device — re-colocate before the forward.
            model_nn.to(q_device)
        val_device = next(model_nn.parameters()).device
        sample_input = sample_input.to(val_device)
        report = validate_non_negativity(model_nn, sample_input, atol=0)
        print(report.failures_only_summary())


if __name__ == "__main__":
    main()
