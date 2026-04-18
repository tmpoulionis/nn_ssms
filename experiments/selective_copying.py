from models.mamba_selective_copying import SelectiveCopyingMambaModel

config = {
    "model_class": SelectiveCopyingMambaModel,
    "model": {
        "num_layers": 2,
        "d_model": 32,
        "vocab_size": 10,
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
        "max_epochs": 50,
        "max_steps": 1e+10,
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
        "dataset_name": "selective_copying",
        "batch_size": 128,
        "L": 400,
        "M": 10,
        "A": 10
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
    "noise_injector": {
        "noise_schedule": {
            "train": False,
            "eval": False,
        },
        "noise_config": {
            "input": True,
            "weight": True,
            "bias": True,
            "output": True,
        },
        "noise_std": 0.02,
    },
    "non_negative": {
        "enabled": False,
        "penalty_type": "elastic",
        "penalty_weight": 5e-1,
        "live_clipping": True,
        "clip_mode": "step",
        "clip_interval": 1,
        "exclude_bias": False,
        "scheduler": {
            "l2_weight_start": 1,
            "l2_weight_end": 0,
            "delay": 0,
            "warmup": 0.4,
        },
    },
    "seed": None,
    "wandb": {
        "project": None,
        "name": "selective_copying: d32-l2-s8 nn_pelulike_v2 L=400 M=10 A=10 bs=128 seed=None",
        "username": "tmpoulionis-",
        "mode": 1,
    },
}
