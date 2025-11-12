import datasets
import torch
import lightning as L
from models.test_model import MambaModel
from utils.modules import LightningMamba
from config import get_config
from torch.optim.lr_scheduler import LambdaLR
from utils.utils import set_seed
    
    
def train(config):
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Parse config
    MODEL_CONFIG = config["model"]
    TRAINER_CONFIG = config["trainer"]
    DATASET_CONFIG = config["dataset"]
    OPTIMIZER_CONFIG = config["optimizer"]

    # ------- Load Dataset and create DataLoaders -------
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
        
    print(f"  ✓ Dataset: {DATASET_CONFIG["dataset_name"]}")
    print(f"  ✓ Classes: {data["num_classes"]}")
    print(f"  ✓ Input shape: {data['input_shape']}")
    print(f"  ✓ Features: {data['feature_dim']}")
    print(f"  ✓ Sequence Length: {data['sequence_length']}")
    
    # ------- Create Model -------
    print("Constructing Model...")
    model = MambaModel(**MODEL_CONFIG, d_out=num_classes)
    
    # ------- Create Lightning Module -------
    print("Setting up Lightning Module...")
    loss_fn = torch.nn.CrossEntropyLoss()
    
    warmup_steps = int(0.1 * total_steps)
    scheduler_config = {
        "scheduler": create_scheduler,
        "params": {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps
        }
    }
    
    lightning_module = LightningMamba(
        model=model,
        optimizer=torch.optim.AdamW,
        loss_fn=loss_fn,
        scheduler=create_scheduler,
        opt_hyperparams=OPTIMIZER_CONFIG,
        scheduler_config=scheduler_config
    )
    
    # ------- Create Trainer -------
    print("Initializing Trainer...")
    trainer = L.Trainer(**TRAINER_CONFIG)
    trainer.fit(lightning_module, train_loader, val_loader)
    trainer.test(lightning_module, test_loader)
    
def create_scheduler(optimizer, total_steps, warmup_steps=0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    config = get_config()
    train(config)