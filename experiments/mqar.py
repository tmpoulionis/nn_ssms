from models.mamba_selective_copying import SelectiveCopyingMambaModel

config = {
    "model_class": SelectiveCopyingMambaModel,
    "model": {
        "num_layers": 2,
        "d_model": 64,
        "vocab_size": 8192,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "conv_activation": "nn_pelulike_v2",
        "delta_activation": "nn_pelulike_v2",
        "gate_activation": "nn_pelulike_v2",
        "use_prenorm": True,
        "use_final_norm": True,
        "dropout": 0.0,
    },
    "trainer": {
        "max_epochs": None,
        "max_steps": 200_000,
        "accelerator": "auto",
        "devices": 2,
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "max_time": None,
        "enable_progress_bar": True,
        "log_every_n_steps": None,
        "detect_anomaly": False,
    },
    "dataset": {
        "dataset_name": "mqar",
        "batch_size": 64,
        "n_train": 50_000,
        "n_valid": 5_000,
        "n_test": 5_000
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    },
    "lr_scheduler": {
        "warmup": 0.1,
    },
    "seed": 0,
    "wandb": {
        "project": "mqar2",
        "name": "d64-l2-s8 nn_pelulike_v2 batch=64",
        "username": "tmpoulionis-",
        "mode": 1,
    },
}
