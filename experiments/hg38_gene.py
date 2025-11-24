config = {
    "model": {
        "num_layers": 8,
        "d_model": 256,
        'vocab_size': 11, # ['A', 'T', 'C', 'G', 'N', 'a', 't', 'c', 'g', 'n', '.']
        'task': 'generation',
        "mamba_activation": 'silu',
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_prenorm": True,
        "use_final_norm": True,
        "mlp_dims": None,
        "mlp_act": None,
        "out_activation": None,
        "dropout": None,
        "use_layernorm": False,
        "return_last_state": False,
        "use_fast_path": False
    },
    "trainer": {
        "max_epochs": 5,
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
        "dataset_name": "hg38",
        "batch_size": 4,
        "max_length": 2**15
    },
    "optimizer": { # Using AdamW
        "lr": 8e-3,
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