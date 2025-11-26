import argparse
import dataloaders.data as data
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from models.test_model import MambaModel
from utils.lightning import LightningMamba
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import set_seed, model_summary, format_time, handle_wandb_login, print_config, load_config
import wandb
import time

torch.set_float32_matmul_precision('medium') # or 'high'

    
def train(config):
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
    WANDB_CONFIG = config["wandb"]

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
    model = MambaModel(**MODEL_CONFIG, d_out=num_classes)
    
    # ------- W&B Logger -------
    print("\n[3/6] Setting up W&B Logger...")
    usrname = handle_wandb_login(WANDB_CONFIG)

    wandb_logger = WandbLogger(
        project=WANDB_CONFIG["project"],
        entity=usrname,
        name=WANDB_CONFIG["name"],
        log_model="all",
        save_dir="./wandb_logs"
    )
    
    
    # ------- Callbacks -------
    print("\n[4/6] Setting up Callbacks...")
    
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=f"./checkpoints/{wandb_logger.name}",
            filename="best-{epoch:02d}-{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=False
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            verbose=True
        )
    ]
    
    # ------- Scheduler -------
    warmup_steps = int(0.1 * total_steps)
    
    scheduler_config = {
        "scheduler": create_scheduler,
        "params": {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps
        }
    }
    
    # ------- Lightning Module -------
    print("\n[5/6) Setting up Lightning Module...")
    loss_fn = torch.nn.CrossEntropyLoss()
    
    lightning_module = LightningMamba(
        model=model,
        optimizer=torch.optim.AdamW,
        loss_fn=loss_fn,
        opt_hyperparams=OPTIMIZER_CONFIG,
        scheduler_config=scheduler_config
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
        trainer.fit(lightning_module, train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        
    print("\n" + "="*70)
    print("RUNNING TEST EVALUATION")
    print("="*70 + "\n")
    
    trainer.test(lightning_module, test_loader, ckpt_path="best")
    
    elapsed = time.time() - start_time
    print(f"Total training time: {format_time(elapsed)}")
    
    wandb.finish()
    
    return trainer, lightning_module
    
    
import math
from torch.optim.lr_scheduler import LambdaLR

def create_scheduler(optimizer, total_steps, warmup_steps=0):
    def lr_lambda(current_step):
        # Linear Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine Decay
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', required=True, help="Which experiment config file from ./experiments to run.")
    args = parser.parse_args()
    
    config = load_config(args.experiment)
    trainer, model = train(config)