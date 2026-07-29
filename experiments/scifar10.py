from models.mamba_class import ClassificationMambaModel
from models.mamba_model import MambaModel

config = {
    "model_class": ClassificationMambaModel,
    "model": {
        "backbone_cls": MambaModel,
        "num_layers": 2,
        "d_model": 32,
        "input_dim": 24,
        "d_out": 10,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "conv_activation": "nn_pelulike_v2",
        "delta_activation": "nn_pelulike_v2",
        "gate_activation": "nn_pelulike_v2",
        "use_prenorm": True,
        "use_final_norm": True,
        "mlp_dims": [32, 128, 10],
        "mlp_act": "nn_pelulike_v2",
        "use_mlp_prenorm": True,
        "out_activation": None,
        "dropout": 0.1,
        "pooling": "mean",
    },
    "trainer": {
        "max_epochs": None,
        "max_steps": 200000,
        "accelerator": "auto",
        "devices": 3,
        "enable_checkpointing": True,
        "gradient_clip_val": 0.1,
        "max_time": None,
        "enable_progress_bar": True,
        "log_every_n_steps": None,
        "detect_anomaly": False,
    },
    "dataset": {
        "dataset_name": "scifar10",
        "batch_size": 64,
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
    "seed": 1,
    "wandb": {
        "project": "cifar10",
        "name": "d32-l2-s16 nn_pelulike_v2 batch=64",
        "username": "tmpoulionis-",
        "mode": 1,
    },
}
