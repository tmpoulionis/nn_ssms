import datasets
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from models.test_model import MambaModel
from nn_mamba.utils.lightning import LightningMamba
from config import get_config
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import set_seed, model_summary, format_time, handle_wandb_login
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
    set_seed(config["seed"])
    
    # Parse config
    MODEL_CONFIG = config["model"]
    TRAINER_CONFIG = config["trainer"]
    DATASET_CONFIG = config["dataset"]
    OPTIMIZER_CONFIG = config["optimizer"]
    WANDB_CONFIG = config["wandb"]

    # ------- Load Dataset and create DataLoaders -------
    print("\n[1/6] Preparing DataLoaders...")
    data = datasets.get_dataloaders(**DATASET_CONFIG)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    num_classes = data["num_classes"]
    if TRAINER_CONFIG["max_epochs"] is not None:
        total_steps = len(train_loader) * TRAINER_CONFIG["max_epochs"]
        if TRAINER_CONFIG["max_steps"] is not None:
            total_steps = min(total_steps, TRAINER_CONFIG["max_steps"])
    else:
        try:
            total_steps = TRAINER_CONFIG["max_steps"]
        except: 
            raise ValueError("Either max_steps or max_epochs must be defined.")
        
    print(f"  ✓ Dataset: {DATASET_CONFIG['dataset_name']}")
    print(f"  ✓ Classes: {data['num_classes']}")
    print(f"  ✓ Input shape: {data['input_shape']}")
    print(f"  ✓ Features: {data['feature_dim']}")
    print(f"  ✓ Sequence Length: {data['sequence_length']}")
    
    # ------- Model -------
    print("\n[2/6] Constructing Model...")
    model = MambaModel(**MODEL_CONFIG, d_out=num_classes)
    model_summary(model)
    
    # ------- W&B Logger -------
    print("\n[3/6] Setting up W&B Logger...")
    usrname = handle_wandb_login()

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
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            verbose=True
        )
    ]
    
    print(f"  ✓ Learning rate monitor")
    print(f"  ✓ Model checkpointing (save top 3)")
    print(f"  ✓ Early stopping (patience=20)")
    
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
    
    print(f"  ✓ Max steps: {TRAINER_CONFIG['max_steps'] if TRAINER_CONFIG['max_steps'] else 'N/A'}")
    print(f"  ✓ Max epochs: {TRAINER_CONFIG['max_epochs'] if TRAINER_CONFIG['max_epochs'] else 'N/A'}")
    print(f"  ✓ Accelerator: {TRAINER_CONFIG['accelerator']}")
    print(f"  ✓ Gradient clip: {TRAINER_CONFIG['gradient_clip_val']}")
    
    # ------- Training -------
    print("\n" + "="*20)
    print("STARTING TRAINING")
    print("="*20 + "\n")
    
    try:
        trainer.fit(lightning_module, train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        
    print("\n" + "="*20)
    print("RUNNING TEST EVALUATION")
    print("="*20 + "\n")
    
    trainer.test(lightning_module, test_loader, ckpt_path="best")
    
    elapsed = time.time() - start_time
    print(f"Total training time: {format_time(elapsed)}")
    
    wandb.finish()
    
    return trainer, lightning_module
    
    
def create_scheduler(optimizer, total_steps, warmup_steps=0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    config = get_config()
    trainer, model = train(config)