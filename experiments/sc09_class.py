from utils.utils import build_run_name

config = {
    "model": {
        "num_layers": 2,
        "d_model": 64,
        'vocab_size': None,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        'task': 'classification',
        "conv_activation": 'nn_pelulike_v2',
        "delta_activation": 'nn_pelulike_v2',
        "gate_activation": 'nn_pelulike_v2',
        "dt_min": 0.001,
        "dt_max": 0.1,
        "use_prenorm": True,
        "use_final_norm": True,
        "mlp_dims": [64, 128, 10],
        "mlp_act": 'nn_pelulike_v2',
        "out_activation": None,
        "dropout": 0.1,
        "use_mlp_prenorm": True,
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
        "detect_anomaly": False
    },
    "dataset": {
        "dataset_name": "sc09",
        "batch_size": 128,
    },
    "optimizer": { # Using AdamW
        "lr": 1e-3,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8
    },
    "lr_scheduler": {
        "warmup": 0.1
        },
    "noise_injector": {
        "noise_schedule": {
            "train": False,
            "eval": False
        },
        "noise_config":
                {"input": True,
                "weight": True,
                "bias": True,
                "output": True}
        ,
        "noise_std": 0.02
    },
    "non_negative": {
        "enabled": True,
        "penalty_type": "elastic",
        "penalty_weight": 1e-1,
        "live_clipping": False,
        "exclude_bias": True,
        "clip_mode": "step", # "step" or "epoch"
        "clip_interval": 25,
        "scheduler":
                {"l2_weight_start": 1,
                "l2_weight_end": 0,
                "delay": 0,
                "warmup": 0}
    },
    "seed": 42,
    "wandb": {
        "project": None,
        "notes": None,  # optional freeform suffix appended to auto-generated name
        "username": 'tmpoulionis-',
        "mode": 1 #(1: online or 2: offline)
    }
}

config["wandb"]["name"] = build_run_name(config)
