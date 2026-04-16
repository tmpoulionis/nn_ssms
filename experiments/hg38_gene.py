config = {
    "task": "generation",
    "model": {
        "num_layers": 2,
        "d_model": 64,
        "vocab_size": 18,  # 7 special tokens + 11 chars
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "conv_activation": "nn_pelulike_v2",
        "delta_activation": "nn_pelulike_v2",
        "gate_activation": "nn_pelulike_v2",
        "a_max": 10,
        "a_min": -10,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "use_prenorm": True,
        "use_final_norm": True,
        "dropout": 0,
    },
    "trainer": {
        "max_epochs": 30,
        "max_steps": 200000,
        "accelerator": "auto",
        "devices": 1,
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "max_time": None,
        "enable_progress_bar": True,
        "log_every_n_steps": None,
        "detect_anomaly": False,
    },
    "dataset": {
        "dataset_name": "hg38",
        "max_length": 2**10,
        "batch_size": 64,
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    },
    "lr_scheduler": {
        "warmup": 0.1,
    },
    "seed": 42,
    "wandb": {
        "project": None,
        "notes": None,
        "username": "tmpoulionis-",
        "mode": 1,  # 1: online, 2: offline
    },
}

config["wandb"]["name"] = "hg38_gene: l2d16 (nn_pelulike_v2) maxlen=2**10 bs=256"
