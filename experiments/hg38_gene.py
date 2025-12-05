config = {
    "model": {
        "num_layers": 6,
        "d_model": 256,
        'vocab_size': 18, # ['A', 'T', 'C', 'G', 'N', 'a', 't', 'c', 'g', 'n', '.'] + 7 special tokens
        'task': 'generation',
        "conv_activation": 'silu',
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
        "use_fast_path": True
    },
    "trainer": {
        "max_epochs": None,
        "max_steps": 200000,
        "accelerator": "auto",
        "devices": 2,
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "max_time": None,
        "enable_progress_bar": True,
        "log_every_n_steps": None,
        "detect_anomaly": False
    },
    "dataset": {
        "dataset_name": "hg38",
        "batch_size": 16,
        "max_length": 2**13,
        "num_workers": 4,
        "pin_memory": True
    },
    "optimizer": { # Using AdamW
        "lr": 8e-3,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8
    },
    "noise_injector": {
        "noise_schedule": {
            "train": False,
            "eval": False
        },
        "noise_config": {
            "input": True,
            "weight": True,
            "bias": True,
            "output": True
        },
        "noise_std": 0.1
    },
    "seed": None,
    "wandb": {
        "project": None,
        "name": None,
        "username": 'tmpoulionis-',
        "mode": 1 #(1: online or 2: offline)
    }
}