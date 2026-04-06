from utils.utils import build_run_name

config = {
    "model": {
        "num_layers": 3,
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "conv_activation": 'nn_pelulike_v2',
        "delta_activation": 'nn_pelulike_v2',
        "gate_activation": 'nn_pelulike_v2',
        "a_max": 10,
        "a_min": -10,
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
    "task": "classification",
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
    "seed": None,
    "wandb": {
        "project": None,
        "notes": None,  # optional freeform suffix appended to auto-generated name
        "username": 'tmpoulionis-',
        "mode": 1 #(1: online or 2: offline)
    }
}

config["wandb"]["name"] = "l3d64: act(nn_elulike_v2) a(-10, 10) run=2"