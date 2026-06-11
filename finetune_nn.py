"""
Finetune a non-negative isomorphic model with the MAdam optimizer.

Loads a non-negative model saved by `eval.py --save_nn` (under
./checkpoints/nn/<project>/<name>/), rebuilds it, and finetunes it while keeping
the non-negative weights >= 0 by construction (no clipping):

  - non-negative params (W_pos/W_neg_abs/w_pos/w_neg_abs/D_pos/D_neg_abs)
        -> M_SPOW (multiplicative, sign-preserving)
  - free params (A_log, RMSNorm.weight)
        -> U_Add (additive, vanilla Adam step)

Best checkpoints are written to ./checkpoints/ft/<project>/<name>/.

Usage:
    python finetune_nn.py --ckpt ./checkpoints/nn/<project>/<name>/<file>.ckpt --epochs 5 --lr_out 1e-3
    python finetune_nn.py --ckpt <path> --epochs 5 --lr_out 5e-4 --lr_in 1.0 --name my_run
    python finetune_nn.py --ckpt <path> --epochs 5 --seed 1      # override the config seed (sweep seeds)
"""

import argparse
import os

import torch
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import dataloaders.data as data
from utils.lightning import LightningMamba
from utils.utils import set_seed, model_summary, checkpoint_dir
from nnt.transformation import transform_to_nn
from nnt.nn_adam import MAdam
from nnt.u_funcs import M_SPOW, U_Add
from nnt.validate_nn import validate_non_negativity


# Parameter-name suffixes that must stay non-negative (driven by the multiplicative
# u_func). Everything else (A_log, RMSNorm.weight) is sign-free -> additive u_func.
NN_SUFFIXES = ("W_pos", "W_neg_abs", "w_pos", "w_neg_abs", "D_pos", "D_neg_abs")


def split_nn_params(model):
    """Split trainable params into (non_negative, free) by name suffix."""
    nn_p, free_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nn_p if name.split(".")[-1] in NN_SUFFIXES else free_p).append(p)
    return nn_p, free_p


class NNFinetuneModule(LightningMamba):
    """LightningMamba that optimizes with two-group MAdam instead of AdamW.

    Overrides configure_optimizers only; everything else (steps, metrics,
    checkpointing) is inherited unchanged.
    """
    def __init__(self, *args, madam_kwargs, **kwargs):
        super().__init__(*args, **kwargs)
        self._madam_kwargs = madam_kwargs

    def configure_optimizers(self):
        nn_p, free_p = split_nn_params(self.model)
        # "lr" is ignored by MAdam (it uses lr_out) but lets LearningRateMonitor read a value.
        lr = self._madam_kwargs["lr_out"]
        groups = [
            {"params": nn_p,   "u_func": M_SPOW(), "lr": lr},   # multiplicative, sign-preserving
            {"params": free_p, "u_func": U_Add(),  "lr": lr},   # additive, vanilla Adam
        ]
        # u_func= is MAdam's required positional default; per-group entries override it.
        return MAdam(groups, u_func=U_Add(), **self._madam_kwargs)


def finetune(args):
    # ------ Load the non-negative checkpoint ------
    print(f"Loading non-negative checkpoint: {args.ckpt}")
    nn_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = dict(nn_ckpt["experiment_config"])
    bounds = config["nn_transform"]                  # {a_min, a_max}, embedded by eval.py --save_nn
    a_min, a_max = bounds["a_min"], bounds["a_max"]
    model_cls = config["model_class"]
    print(f"Model class:  {model_cls.__name__}")
    print(f"Dataset:      {config['dataset']['dataset_name']}")
    print(f"NN bounds:    a_min={a_min}, a_max={a_max}")

    # --seed overrides the checkpoint's config seed (for seed sweeps); records the value used.
    seed = args.seed if args.seed is not None else config.get("seed")
    if seed is not None:
        set_seed(seed)
        config["seed"] = seed
    print(f"Seed:         {seed}")

    # ------ Dataloaders ------
    print("\nBuilding dataloaders...")
    dataset_cfg = dict(config["dataset"])
    if args.batch_size is not None:
        dataset_cfg["batch_size"] = args.batch_size
    dataset = data.get_dataloaders(**dataset_cfg)
    train_loader = dataset["train_loader"]
    val_loader = dataset["valid_loader"]
    test_loader = dataset["test_loader"]
    total_steps = len(train_loader) * args.epochs

    # ------ Rebuild the non-negative model ------
    print("Building model & loading non-negative weights...")
    model = model_cls(**config["model"])
    model_nn = transform_to_nn(model, a_min=a_min, a_max=a_max)
    # Lightning stores the module weights under the "model." prefix; strip it to load
    # directly into model_nn (structure matches exactly).
    state = {k[len("model."):]: v for k, v in nn_ckpt["state_dict"].items() if k.startswith("model.")}
    model_nn.load_state_dict(state)

    # nn_transform already rides along in config; record provenance. Finetuned checkpoints
    # stay self-describing and reload through the same eval.py path.
    config["finetune_source"] = args.ckpt
    # --name overrides the <configuration> folder level (default = the run's wandb name).
    if args.name is not None:
        config = {**config, "wandb": {**config["wandb"], "name": args.name}}

    # ------ Lightning module (two-group MAdam) ------
    madam_kwargs = {
        "lr_in": args.lr_in,
        "lr_out": args.lr_out,
        "betas": tuple(args.betas),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }
    lightning_module = NNFinetuneModule(
        model=model_nn,
        total_steps=total_steps,
        optimizer=MAdam,            # unused (configure_optimizers is overridden), kept for hparams
        lr_scheduler=None,          # constant lr_out: LambdaLR can't drive MAdam's lr_out
        opt_hyperparams={},
        non_negative=None,
        noise_injection=None,
        quantization=None,
        config=config,
        madam_kwargs=madam_kwargs,
    )

    # ------ Logger ------
    # ./checkpoints/ft/<project>/<name>/  (mirrors regular/nn).
    ckpt_dir = checkpoint_dir("ft", config)
    name = config["wandb"]["name"]
    if args.wandb:
        from utils.utils import handle_wandb_login
        from lightning.pytorch.loggers import WandbLogger
        wandb_cfg = config.get("wandb", {})
        usrname = handle_wandb_login(wandb_cfg)
        logger = WandbLogger(
            project=wandb_cfg.get("project", "nn_finetune"),
            entity=usrname,
            name=f"finetune-{name}",
            log_model=False,
        )
    else:
        logger = CSVLogger(save_dir=ckpt_dir, name="logs")

    # ------ Callbacks ------
    monitor = model_cls.checkpoint_metric
    mode = model_cls.checkpoint_mode
    seed_tag = f"seed={seed}-" if seed is not None else ""
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"finetune-{seed_tag}{{epoch:02d}}-{{{monitor}:.4f}}",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
        ),
    ]

    # ------ Trainer ------
    trainer_cfg = config.get("trainer", {}) or {}
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps"),
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=True,
    )

    print("\n--------- Model ---------")
    model_summary(model_nn)
    print(f"\nFinetuning for {args.epochs} epoch(s) | lr_out={args.lr_out} lr_in={args.lr_in} "
          f"betas={tuple(args.betas)} wd={args.weight_decay}")
    print(f"Checkpoints -> {ckpt_dir}")

    # ------ Finetune + test ------
    print("\n" + "=" * 70)
    print("STARTING FINETUNING")
    print("=" * 70 + "\n")
    trainer.fit(lightning_module, train_loader, val_loader)

    print("\n" + "=" * 70)
    print("RUNNING TEST EVALUATION (best checkpoint)")
    print("=" * 70 + "\n")
    trainer.test(lightning_module, test_loader, ckpt_path="best")

    # ------ Non-negativity sanity check ------
    print("\n---------- Non-Negativity Validation ----------")
    sample_input, _ = next(iter(test_loader))
    val_device = next(model_nn.parameters()).device
    sample_input = sample_input.to(val_device)
    report = validate_non_negativity(model_nn, sample_input, atol=0)
    print(report.failures_only_summary())

    # Close the W&B run cleanly (same as train.py) — without this the run is left
    # open, gets marked "failed" after exit, and its final metrics never flush.
    if args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Finetune a non-negative isomorphic model (from ./checkpoints/nn/) with MAdam."
    )
    parser.add_argument("--ckpt", required=True, help="Path to a non-negative checkpoint saved by `eval.py --save_nn`.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of finetuning epochs (default: 5).")
    parser.add_argument("--lr_out", type=float, default=1e-4, help="MAdam outer learning rate / step size.")
    parser.add_argument("--lr_in", type=float, default=1, help="MAdam inner learning rate.")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="MAdam betas (default: 0.9 0.999).")
    parser.add_argument("--eps", type=float, default=1e-8, help="MAdam epsilon (default: 1e-8).")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="MAdam weight decay (default: 0).")
    parser.add_argument("--seed", type=int, default=None, help="Override the checkpoint's config seed (for seed sweeps). Default: use the config seed.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override the dataset batch size.")
    parser.add_argument("--name", default=None, help="Override the <configuration> folder level under ./checkpoints/ft/<project>/ (default: the run's wandb name).")
    parser.add_argument("--wandb", default=True, help="Log to Weights & Biases instead of a local CSV logger.")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
