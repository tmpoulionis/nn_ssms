config = {
    "model": {
        "num_layers": 2,
        "d_model": 64,
        "mamba_activation": 'pelulike',
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_prenorm": True,
        "use_final_norm": True,
        "mlp_dims": [64, 128, 10],
        "mlp_act": "pelulike",
        "out_activation": None,
        "dropout": 0.1,
        "use_layernorm": True,
        "return_last_state": False,
        "use_fast_path": False
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
        "detect_anomaly": True
    },
    "dataset": {
        "dataset_name": "sc09",
        "root": "./data",
        "batch_size": 128,
        "mel_transform": True,
        "n_mels": 64,
        "n_fft": 400,
        "hop_length": 150
    },
    "optimizer": { # Using AdamW
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8
    },
    "seed": None,
    "wandb": {
        "project": None,
        "name": None,
        "username": 'tmpoulionis-',
        "mode": 1 #(1: online or 2: offline)
    }
}

def get_config():
    return config