# nn-mamba — running experiments

Non-negative Mamba state-space models with photonic activations.
Everything is driven by a **config file** in [`experiments/`](experiments/).

---

## 1. Install

```bash
git clone <repo-url> && cd nn_mamba
pip install -e .          # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU for anything non-trivial.
Datasets download themselves into `./data/` on first run.

---

## 2. Run an experiment

```bash
python train.py -e scifar10          # run experiments/scifar10.py
python train.py -e sc -i 3           # run it 3x, seed incremented each time
python train.py -e mqar -r path.ckpt # resume from a checkpoint
```

| Flag | Meaning |
|---|---|
| `-e, --experiment` | Module name in `experiments/` (no `.py`). **Required.** |
| `-r, --resume` | Path to a `.ckpt` to resume training from. |
| `-i, --iterations` | Repeat the run N times, adding `+1` to `seed` each time. |

Training runs, then automatically evaluates the **best** checkpoint on the test set.
Checkpoints land in `./checkpoints/regular/<wandb.project>/<wandb.name>/`.

Example configs: `scifar10`, `sc`, `selective_copying`, `mqar`.

---

## 3. Anatomy of a config file

A config is a plain Python module exporting a dict named `config`.
Import your model classes at the top, then fill in the keys below.

```python
from models.mamba_cifar import ClassificationMambaModel
from models.mamba_model import MambaModel

config = {
    "model_class": ClassificationMambaModel,   # task wrapper (head + loss + metrics)
    "model":        {...},            # -> model_class(**model)
    "trainer":      {...},            # -> lightning.Trainer(**trainer)
    "dataset":      {...},            # -> get_dataloaders(**dataset)
    "optimizer":    {...},            # -> torch.optim.AdamW(params, **optimizer)
    "lr_scheduler": {...},
    "seed":         1,
    "wandb":        {...},
    "quantization": {...},            # optional
}
```

### `model_class` — picks the task wrapper

| Class | Task | Input → Output | Checkpoint metric |
|---|---|---|---|
| `ClassificationMambaModel` (`models/mamba_class.py`) | sequence classification | `(B, L, input_dim)` → `(B, d_out)` | `val_acc` (max) |
| `GenerationMambaModel` (`models/mamba_generation.py`) | token-level LM | `(B, L)` → `(B, L, vocab_size)` | `val_loss` (min) |

Each wrapper owns its own `compute_loss` / `compute_metrics`, so adding a task means
adding a wrapper class — not touching the training loop.

### `model` — architecture

**Wrapper (classification / `CIFARMambaModel`)**

| Key | Default | What it does |
|---|---|---|
| `backbone_cls` | `MambaModel` | Backbone class to stack. |
| `input_dim` | `3` | Width of one input token; a `Linear` projects it to `d_model`. |
| `d_out` | — | Number of classes. Must equal the dataset's `num_classes`. |
| `mlp_dims` | `[d_model, d_out]` | Classifier head widths. First must be `d_model`, last `d_out`. |
| `mlp_act` | `"gelu"` | Activation between head layers. |
| `use_mlp_prenorm` | `True` | RMSNorm before each hidden head activation. |
| `out_activation` | `None` | Activation on the logits (normally leave `None`). |
| `dropout` | `0.1` | Dropout in the head. |
| `pooling` | `"mean"` | How to pool the sequence: `"mean"` or `"last"`. |

**Wrapper (token-level / `SelectiveCopyingMambaModel`)**

| Key | Default | What it does |
|---|---|---|
| `vocab_size` | — | Embedding size and output width. |
| `dropout` | `0.0` | Dropout after the embedding. |

**Backbone (`MambaModel` → `MambaBlock`)**

| Key | Default | What it does |
|---|---|---|
| `num_layers` | — | Number of stacked Mamba blocks. |
| `d_model` | — | Residual-stream width. |
| `d_state` | `16` | SSM hidden state size `N`. |
| `d_conv` | `4` | Depthwise causal conv kernel width. |
| `expand` | `2` | Inner width multiplier (`d_inner = expand * d_model`). |
| `conv_activation` | `"silu"` | Activation after the conv branch. |
| `delta_activation` | `"softplus"` | Activation producing the SSM step size `Δ`. Must be positive. |
| `gate_activation` | `"silu"` | Activation on the gating branch. |
| `use_prenorm` | `True` | RMSNorm before each block. |
| `use_final_norm` | `True` | RMSNorm + clamp after the stack. |
| `a_min`, `a_max` | `-10`, `10` | Clamp bounds for `LinearBounded` (models the photonic dynamic range). |
| `dt_rank` | `"auto"` | Rank of the `Δ` projection; `"auto"` = `ceil(d_model/16)`. |
| `dt_min`, `dt_max`, `dt_init`, `dt_scale`, `dt_init_floor` | — | `Δ` bias initialisation. Rarely changed. |
| `conv_bias`, `bias` | `True`, `False` | Bias terms on the conv / linear projections. |

**Activations** (`utils/activations.py`) - Any of the activation functions included can be used, or add a custom one in `utils/activations.py`.

### `trainer` — passed verbatim to `lightning.Trainer`

| Key | Note |
|---|---|
| `max_steps` / `max_epochs` | **At least one must be set** (the other can be `None`). Both set → training stops at whichever comes first. |
| `accelerator` | `"auto"`, `"gpu"`, `"cpu"`. |
| `devices` | Number of GPUs (or a device list). |
| `gradient_clip_val` | Gradient-norm clip. `0.1` is the working default here. |
| `enable_checkpointing` | Keep `True` — test evaluation loads `ckpt_path="best"`. |
| `max_time` | Wall-clock limit, e.g. `"00:12:00:00"`. |
| `enable_progress_bar`, `log_every_n_steps`, `detect_anomaly` | Standard Lightning knobs. |

Callbacks are fixed in `train.py`: LR monitor, best-checkpoint saver (on the wrapper's
`checkpoint_metric`), and early stopping on `val_loss` with `patience=50`.

### `dataset` — passed to `get_dataloaders`

| Key | Note |
|---|---|
| `dataset_name` | Registry key: `scifar10`, `sc09`, `speechcommands`, `selective_copying`, `mqar`. |
| `batch_size` | Per-device batch size. |
| `num_workers`, `pin_memory` | Optional; default `4` / `True`. |
| *anything else* | Overrides that dataset's defaults in `dataloaders/data_registry.py`. |

Defaults per dataset live in `DATASET_REGISTRY`, e.g. `selective_copying` accepts
`L` (sequence length), `M` (tokens to copy), `A` (vocab), `variable`; `mqar` accepts
`input_seq_len`, `num_kv_pairs`, `vocab_size`, …

To add a dataset: write the class in `dataloaders/datasets/`, exposing `process()` and
`create_dataset() -> (dict_of_splits, num_classes)`, then register it in the registry.

### `optimizer` — AdamW kwargs

`lr`, `weight_decay`, `betas`, `eps`.

### `lr_scheduler`

| Key | Note |
|---|---|
| `warmup` | Fraction of total steps spent in **linear warmup**, e.g. `0.1`. |

After warmup the LR follows cosine decay, floored at `0.1 ×` the base LR.
Set `"lr_scheduler": None` for a constant LR.

### `seed`

Int seeding torch/numpy/random. `None` disables seeding. With `-i N` it becomes
`seed, seed+1, … seed+N-1`.

### `wandb`

| Key | Note |
|---|---|
| `project` | W&B project name. Also the checkpoint folder. |
| `name` | Run name. Also the checkpoint subfolder — keep it descriptive and unique. |
| `username` | W&B entity. `None` prompts interactively. |
| `mode` | `1` = log online, `2` = offline. |

### `quantization` (optional — QAT)

Omit the key, or set `"enabled": False`, to train in full precision.

| Key | What it does |
|---|---|
| `enabled` | Master switch. |
| `bits` | Bit width for all fake-quants. |
| `symmetric` | Signed range if `True`, unsigned if `False`. |
| `per_channel_weights` | One scale per output channel for weights. |
| `observer_momentum` | EMA alpha for input/output observers (e.g. `0.01`). |
| `tensors` | Which tensors to quantize: `{"input":, "weight":, "bias":, "output":}`. |
| `skip_modules` | Substring blocklist on layer names, e.g. `["norm"]`. |
| `quantize_embeddings` | Wrap `nn.Embedding` too. |
| `calibration_steps` | Observe-only steps before fake-quant switches on. |
| `freeze_observer_step` | Step at which observer EMAs freeze (`None` = never). |
| `train_quant` / `eval_quant` | Apply fake-quant during training / during val+test. |

---

## 4. After training: non-negative transform, eval, finetune

The non-negativity is applied **post-training** as an exact isomorphic transform
(`nnt/`), decomposing every weight as `W = W_pos − W_neg_abs`. It is mathematically
equivalent, so accuracy is preserved by construction.

```bash
# evaluate a checkpoint; --nn also transforms and validates non-negativity
python eval.py --ckpt <path.ckpt> --nn --save_nn

# post-training quantization on top
python eval.py --ckpt <path.ckpt> --nn --quant --bits 8

# finetune the saved non-negative model (stays non-negative via MAdam)
python finetune_nn.py --ckpt checkpoints/nn/<project>/<name>/<file>.ckpt --epochs 5
```

`eval.py`: `--valid` / `--train` add those splits, `--a_min` / `--a_max` set the
transform bounds, `--calib-batches` sets PTQ calibration batches, `-v` prints the
parameter summary.

`finetune_nn.py`: `--lr_out` / `--lr_in` are the MAdam outer/inner step sizes;
`--epochs`, `--betas`, `--eps`, `--weight_decay`, `--seed`, `--batch_size`, `--name`
behave as expected.

Checkpoint layout: `./checkpoints/{regular,nn,ft}/<project>/<name>/`.

---

## 5. Repository map

```
train.py                 entry point — config → dataloaders → model → Trainer
eval.py                  checkpoint evaluation + NN transform + PTQ
finetune_nn.py           finetuning of non-negative models
experiments/             config files (one per experiment)
models/                  task wrappers + MambaModel backbone
mamba/                   MambaBlock and the selective scan
nnt/                     isomorphic non-negative transformation
dataloaders/             dataset registry, datasets, DataLoader factory
utils/                   activations, photonic activations, quantization,
                         LightningMamba module, scheduler/helpers
```
