"""
Non-negative weight transformation for photonic inference.

For each weight matrix W, compute α = max(0, -min(W)) and shift:
    W' = W + α   (all elements ≥ 0)

Correction restores mathematical equivalence:
    W'x = Wx + α·Σxᵢ   →   Wx = W'x − α·Σxᵢ   (digital correction)

Photonic hardware computes W'x (non-negative weights); digital electronics
subtract α·Σxᵢ to recover the original output before the activation.

Usage:
    from models.nn_transformed_model import transform_to_nn
    model_nn = transform_to_nn(trained_model)  # deepcopy, no retraining needed
"""

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Helpers ─────────────────────────────────────────────────────────────────

def _shift(t: torch.Tensor) -> float:
    """t.min() find the min value, .clamp(max=0.0) ensure it's negative, .neg() make it positive """
    return float(t.min().clamp(max=0.0).neg())

# ── Shifted Layers ───────────────────────────────────────────────────────────
class ShiftedLinear(nn.Module):
    """
    Equivalent to nn.Linear with non-negative weight matrix.
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        alpha = _shift(weight)
        self.weight = nn.Parameter(weight + alpha)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
        self.register_buffer('alpha', torch.tensor(alpha, dtype=weight.dtype)) # a buffer is a parameter that is being saved in the model's state, but it's not a trainable parameter.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # Correction: α · (sum of all input features per sample)
        return y - self.alpha * x.sum(dim=-1, keepdim=True)


class ShiftedConv1D(nn.Module):
    """
    Equivalent to a depthwise Conv1d with non-negative weights.
    """
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        assert conv.groups == conv.in_channels, "Only depthwise conv (groups==in_channels) supported"
        w = conv.weight.data.clone()  #(C, 1, K)
        alpha = _shift(w)
        self.weight = nn.Parameter(w + alpha)
        self.bias = nn.Parameter(conv.bias.data.clone()) if conv.bias is not None else None
        self.register_buffer('alpha', torch.tensor(alpha, dtype=w.dtype))
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = F.conv1d(x, self.weight, self.bias, kernel_size=self.kernel_size, groups=self.groups, padding=self.padding)
        # Need the sliding sum (sum of the input x elements inside the kernel window in each step during sliding) to calculate the correction for each output element.
        # Each output element is calculated as x^ij_k*(W_k - alpha) = x^ij_k*W_k - alpha*x^ij_k = y - alpha*x^ij_k --> alpha*sum(x^ij) because alpha is constant.
        # where i channel index, j convolution slide step index, k index in kernel window.
        ones_k = torch.ones(self.groups, 1, self.kernel_size, device=x.device, dtype=x.dtype) # (C, 1, K)
        sliding_sum = F.conv1d(x, ones_k, groups=self.groups, padding=self.padding) # Returns a matrix containing sum(x^ij) for every i and j
        return y - self.alpha * sliding_sum

# ── Modified Selective Scan ──────────────────────────────────────────────────

def _selective_scan_nn(
    u, delta, A, B, C, D, z,
    delta_bias, delta_activation, gate_activation,
    return_last_state=False,
):
    """
    Selective scan equivalent to selective_scan_photonic_fn, but shifts B and C
    to be non-negative. Single global scalar shift per tensor.

    A is always positive (exp(A_log)) — no shift needed.
    deltaA = exp(-delta·A) is always in (0,1) — no shift needed.

    For B (B,N,L):
        beta  = max(0, -min(B))             one scalar for the entire tensor
        B_nn  = B + beta                    (≥ 0), computed before the loop
        per step: deltaB_u = delta_i * B_i_nn * u_i − delta_i * beta * u_i
        (B correction applied inside the loop — it enters the state recurrence)

    For C (B,N,L):
        gamma = max(0, -min(C))             one scalar for the entire tensor
        C_nn  = C + gamma                   (≥ 0), computed before the loop
        per step: y_nn_i = C_i_nn @ x_i    (photonic, no correction yet)
        after loop: y = y_nn − gamma * x_sums   (x_sums accumulated per step)

    Shapes: u (B,D,L), delta (B,D,L), A (D,N), B (B,N,L), C (B,N,L)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()

    delta = delta_activation(delta).clamp(min=1e-4, max=20.0)

    B = B.float()   # (B, N, L)
    C = C.float()   # (B, N, L)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    x = A.new_zeros(batch, dim, dstate)

    # A > 0 always; deltaA ∈ (0,1)
    deltaA = torch.exp(-torch.einsum('bdl,dn->bdln', delta, A))  # (B,D,L,N)

    # Shift parameter matrices
    beta  = _shift(B)
    gamma = _shift(C)
    B_nn = B + beta     # (B, N, L)
    C_nn = C + gamma    # (B, N, L)
    deltaB_u_nn = torch.einsum('bdn,bnl,bdl->bdln', delta, B_nn, u)
    
    ys = []
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i]*x + deltaB_u_nn[:,:,i]
        y_shifted = torch.einsum('bdn,bn->bd', x, C_nn[:,:,i])
        y = y_shifted - gamma*x

    if z is not None:
        out = out * gate_activation(z.float())

    out = out.to(dtype_in)
    return out if not return_last_state else (out, last_state)


# ── NN-Transformed PhotonicMamba ─────────────────────────────────────────────

class NNPhotonicMamba(nn.Module):
    """
    PhotonicMamba with all weight matrices shifted to be non-negative.
    Mathematically equivalent to the original at inference.
    Constructed from an existing PhotonicMamba instance.

    Layers replaced:
        in_proj   → ShiftedLinear
        conv1d    → ShiftedConv1D
        x_proj    → ShiftedLinear
        dt_proj   → ShiftedLinear (weight only; bias kept as buffer for delta_bias)
        out_proj  → ShiftedLinear

    A_log and D are kept as-is: A = exp(A_log) > 0, and deltaA ∈ (0,1).
    """

    def __init__(self, source):
        """
        Args:
            source: a PhotonicMamba instance (trained, on any device)
        """
        super().__init__()
        self.d_model = source.d_model
        self.d_state = source.d_state
        self.d_conv = source.d_conv
        self.expand = source.expand
        self.d_inner = source.d_inner
        self.dt_rank = source.dt_rank

        # Shifted projections
        self.in_proj = ShiftedLinear(
            source.in_proj.weight.data,
            source.in_proj.bias.data if source.in_proj.bias is not None else None,
        )
        self.conv1d = ShiftedConv1D(source.conv1d)
        self.x_proj = ShiftedLinear(
            source.x_proj.weight.data,
            source.x_proj.bias.data if source.x_proj.bias is not None else None,
        )
        # dt_proj: weight only — bias passed as delta_bias to the scan (original behavior)
        self.dt_proj = ShiftedLinear(source.dt_proj.weight.data, bias=None)
        self.register_buffer('dt_proj_bias', source.dt_proj.bias.data.clone())

        self.out_proj = ShiftedLinear(
            source.out_proj.weight.data,
            source.out_proj.bias.data if source.out_proj.bias is not None else None,
        )

        # SSM parameters — A > 0 always, no shift needed
        self.A_log = nn.Parameter(source.A_log.data.clone())
        self.D = nn.Parameter(source.D.data.clone())

        # Activations (same objects — stateless)
        self.act = source.act
        self.delta_activation = source.delta_activation
        self.gate_activation = source.gate_activation

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("inference_params not supported in NNPhotonicMamba")

        batch, seqlen, _ = hidden_states.shape
        A = torch.exp(self.A_log.float())  # (d_inner, d_state), always > 0

        # ── in_proj (shifted) ──
        x_flat = rearrange(hidden_states, "b l d -> (b l) d")
        xz = rearrange(self.in_proj(x_flat), "(b l) d -> b d l", l=seqlen)
        x, z = xz.chunk(2, dim=1)                             # each (B, d_inner, L)

        # ── conv1d (shifted depthwise) + activation ──
        x = self.act(self.conv1d(x)[..., :seqlen])            # (B, d_inner, L)

        # ── x_proj (shifted) → dt, B, C ──
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (B·L, dt_rank+2·d_state)
        dt_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # ── dt_proj (shifted, weight only) ──
        dt = rearrange(self.dt_proj(dt_raw), "(b l) d -> b d l", l=seqlen)  # (B, d_inner, L)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()         # (B, d_state, L)
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()         # (B, d_state, L)

        # ── SSM with shifted B, C ──
        y = _selective_scan_nn(
            x, dt, A, B, C,
            D=self.D.float(),
            z=z,
            delta_bias=self.dt_proj_bias,
            delta_activation=self.delta_activation,
            gate_activation=self.gate_activation,
            return_last_state=False,
        )

        # ── out_proj (shifted) ──
        out = self.out_proj(rearrange(y, "b d l -> b l d"))
        return out


# ── Factory ─────────────────────────────────────────────────────────────────

def _replace_head_linears(head: nn.Sequential) -> nn.Sequential:
    """Replace nn.Linear layers inside an nn.Sequential with ShiftedLinear."""
    new_layers = []
    for layer in head:
        if isinstance(layer, nn.Linear):
            new_layers.append(
                ShiftedLinear(
                    layer.weight.data,
                    layer.bias.data if layer.bias is not None else None,
                )
            )
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def transform_to_nn(model) -> nn.Module:
    """
    Return a deep-copied MambaModel with all weight matrices shifted to be
    non-negative. Mathematically equivalent to the original at inference.
    No retraining required.

    Replaces:
      - Each PhotonicMamba block → NNPhotonicMamba
      - MLP head nn.Linear layers → ShiftedLinear

    RMSNorm layers are left unchanged (future work).

    Args:
        model: trained MambaModel instance

    Returns:
        New model (deepcopy) ready for non-negative photonic inference.
    """
    from models.photonic_model import PhotonicMamba

    model_nn = copy.deepcopy(model)

    for i, block in enumerate(model_nn.mamba_layers):
        if isinstance(block, PhotonicMamba):
            model_nn.mamba_layers[i] = NNPhotonicMamba(block)
        else:
            warnings.warn(
                f"mamba_layers[{i}] is {type(block).__name__}, not PhotonicMamba — skipped."
            )

    if model_nn.head is not None and isinstance(model_nn.head, nn.Sequential):
        model_nn.head = _replace_head_linears(model_nn.head)

    return model_nn
