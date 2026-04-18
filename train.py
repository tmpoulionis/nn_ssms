import argparse
import copy
import dataloaders.data as data
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils.lightning import LightningMamba
from utils.noise_injection import NoiseInjector

from utils.utils import set_seed, model_summary, format_time, handle_wandb_login, print_config, load_config
import math
import wandb
import time    

torch.set_float32_matmul_precision('medium') # or 'high'

    
def train(config, checkpoint_path=None):
    """
    Main training function with full W&B integration.
    
    Args:
        config
    """
    
    start_time = time.time()
    # Set seed for reproducibility
    if config["seed"]:
        set_seed(config["seed"])
    
    # Parse config
    MODEL_CONFIG = config["model"]
    TRAINER_CONFIG = config["trainer"]
    DATASET_CONFIG = config["dataset"]
    OPTIMIZER_CONFIG = config["optimizer"]
    SCHEDULER_CONFIG = config["lr_scheduler"]
    WANDB_CONFIG = config["wandb"]
    
    noise_cfg = config.get("noise_injector")
    if noise_cfg is not None:
        sched = noise_cfg.get("noise_schedule", {})
        if not sched.get("train") and not sched.get("eval"):
            noise_cfg = None

    nn_cfg = config.get("non_negative")
    if nn_cfg is not None and not nn_cfg.get("enabled"):
        nn_cfg = None

    # ------- Load Dataset and create DataLoaders -------
    print("\n[1/6] Preparing DataLoaders...")
    dataset = data.get_dataloaders(**DATASET_CONFIG)
    train_loader = dataset["train_loader"]
    val_loader = dataset["valid_loader"]
    test_loader = dataset["test_loader"]
    num_classes = dataset["num_classes"]
    if TRAINER_CONFIG["max_epochs"] is not None:
        total_steps = len(train_loader) * TRAINER_CONFIG["max_epochs"]
        if TRAINER_CONFIG["max_steps"] is not None:
            total_steps = min(total_steps, TRAINER_CONFIG["max_steps"])
    else:
        try:
            total_steps = TRAINER_CONFIG["max_steps"]
        except: 
            raise ValueError("Either max_steps or max_epochs must be defined.")
    
    # ------- Model -------
    print("\n[2/6] Constructing Model...")
    model_cls = config["model_class"]
    model = model_cls(**MODEL_CONFIG)
    if num_classes is not None and getattr(model, "d_out", None) is not None:
        assert model.d_out == num_classes, (
            f"model.d_out={model.d_out} does not match dataset num_classes={num_classes}"
        )

    # ------- W&B Logger -------
    print("\n[3/6] Setting up W&B Logger...")
    usrname = handle_wandb_login(WANDB_CONFIG)

    wandb_logger = WandbLogger(
        project=WANDB_CONFIG["project"],
        entity=usrname,
        name=WANDB_CONFIG["name"],
        log_model="all"
    )
    
    
    # ------- Callbacks -------
    print("\n[4/6] Setting up Callbacks...")
    checkpoint_monitor = model_cls.checkpoint_metric
    checkpoint_mode = model_cls.checkpoint_mode
    checkpoint_filename = f"best-{{epoch:02d}}-{{{checkpoint_monitor}:.4f}}"

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=f"./checkpoints/{WANDB_CONFIG['name']}",
            filename=checkpoint_filename,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=1,
            save_last=False
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=100,
            mode="min",
            verbose=True
        )
    ]
    
    # ------- Lightning Module -------
    print("\n[5/6) Setting up Lightning Module...")
    lightning_module = LightningMamba(
        model=model,
        total_steps=total_steps,
        optimizer=torch.optim.AdamW,
        lr_scheduler=SCHEDULER_CONFIG,
        opt_hyperparams=OPTIMIZER_CONFIG,
        noise_injection=noise_cfg,
        non_negative=nn_cfg,
        config=config
    )
    
    # ------- Trainer -------
    print("\n[6/6] Initializing Trainer...")
    trainer = L.Trainer(
        **TRAINER_CONFIG,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    
    # ------- Print Config -------
    print("\n" + "-"*40)
    print("TRAINING CONFIGURATION")
    print("-"*40)
    
    print("\n"+  "--------- Model ---------")
    model_summary(model)
    print_config(MODEL_CONFIG, ["all"])
    
    print("\n"+  "--------- Callbacks ---------")
    print(f"   ✓ Learning rate monitor")
    print(f"   ✓ Model checkpointing (save best)")
    print(f"   ✓ Early stopping (patience=20)")
    
    print("\n"+  "--------- Data & Trainer ---------")
    print_config(DATASET_CONFIG, ["dataset_name", "batch_size"])
    print_config(TRAINER_CONFIG, ["all"])
    print_config(dataset, ["input_shape", "num_classes"])
    
    # ------------ Training ------------
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    try:
        if checkpoint_path is None:
            trainer.fit(lightning_module, train_loader, val_loader)
            ckpt = "best"
        else:
            print(f"Loading checkpoint from {checkpoint_path} and running test evaluation...")
            ckpt = checkpoint_path
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        
    print("\n" + "="*70)
    print("RUNNING TEST EVALUATION")
    print("="*70 + "\n")
    
    trainer.test(lightning_module, test_loader, ckpt_path=ckpt)
    
    elapsed = time.time() - start_time
    print(f"Total training time: {format_time(elapsed)}")
    
    wandb.finish()
    
    return trainer, lightning_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', required=True, help="Which experiment config file from ./experiments to run.")
    parser.add_argument('--checkpoint', '-c', default=None, help="Path to checkpoint. Skips training and runs test only.")
    parser.add_argument('--iterations', '-i', type=int, default=1, help="How many times to run an experiment.")
    args = parser.parse_args()
    
    for i in range(args.iterations):
        config = load_config(args.experiment)
        trainer, model = train(config, checkpoint_path=args.checkpoint)