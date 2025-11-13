config = {
    "model": {
        "num_layers": 1,
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_final_norm": True,
        "mlp_dims": [64, 128, 10],
        # "d_out": 10, 
        "activation": "gelu",
        "out_activation": None,
        "dropout": 0.1,
        "use_layernorm": False
    },
    "trainer": {
        "max_epochs": None,
        "max_steps": 200000,
        "accelerator": "auto",
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "max_time": None,
        "enable_progress_bar": True,
    },
    "dataset": {
        "dataset_name": "sc09",
        "root": "./data",
        "batch_size": 16,
        "mel_transform": True,
        "n_mels": 64,
        "n_fft": 400,
        "hop_length": 150
    },
    "optimizer": { # Using AdamW
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95)
    },
    "seed": 42,
    "wandb": {
        "project": None,
        "name": None
    }
}

def get_config():
    return config