config = {
    "model": {
        "num_layers": 2,
        "d_model": 64,
        "vocab_size": 16,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "task": "generation",
        "conv_activation": "nn_pelulike_v2",
        "delta_activation": "nn_pelulike_v2",
        "gate_activation": "nn_pelulike_v2",
        "use_prenorm": True,
        "use_final_norm": True,
        "mlp_dims": None,
        "mlp_act": None,
        "out_activation": None,
        "dropout": 0.0,
        "use_mlp_prenorm": False,
        "return_last_state": False,
        "use_fast_path": False
    },
    "trainer": {
        "max_epochs": 20,
        "max_steps": None,
        "accelerator": "auto",
        "devices": 1,
        "enable_checkpointing": True,
        "gradient_clip_val": 1.0,
        "max_time": None,
        "enable_progress_bar": True,
        "log_every_n_steps": None,
        "detect_anomaly": False
    },
    "dataset": {
        "dataset_name": "induction_heads",
        "batch_size": 64,
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.01,
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
        "noise_config": {
            "input": True,
            "weight": True,
            "bias": True,
            "output": True
        },
        "noise_std": 0.02
    },
    "non_negative": {
        "enabled": True,
        "penalty_type": "elastic",
        "penalty_weight": 5e-1,
        "live_clipping": True,
        "clip_interval": 1,
        "scheduler": {
            "l2_weight_start": 1,
            "l2_weight_end": 0,
            "delay": 0,
            "warmup": 0.4
        }
    },
    "seed": 42,
    "wandb": {
        "project": None,
        "name": "induction_heads: nn_pelulike_v2 elastic w=5e-1 clip(1) seed=42",
        "username": "tmpoulionis-",
        "mode": 1
    }
}
