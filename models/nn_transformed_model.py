"""
Non-negative isomorphic transformation for photonic inference.

Generalized for inputs in [a_min, a_max] (paper Case 4.1.2):

    W_pos = max(W, 0),  W_neg_abs = max(-W, 0)   (both >= 0)

    Hardware (per output neuron i):
        y_nn[i] = W_pos[i] @ (x + |a_min|) + W_neg_abs[i] @ (a_max - x) + b'[i]
    Digital:
        y[i]    = y_nn[i] - c[i]                              (recovers original)

    b_adj[i] = b[i] - a_max * W_neg_abs[i].sum() - |a_min| * W_pos[i].sum()
    c[i]     = max(0, -b_adj[i])
    b'[i]    = b_adj[i] + c[i]  >= 0

    When a_min=0 this reduces to the standard [0, a_max] case.

For the SSM, B and C are decomposed as B = B_nn - c_B, C = C_nn - c_C.
A dual-state recurrence h', h_delta recovers the correct output using only
non-negative quantities (hardware-computable), with a digital correction at the end.

Usage:
    from models.nn_transformed_model import transform_to_nn
    model_nn = transform_to_nn(trained_model, a_min=0.0, a_max=1.0)
"""

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.activations import LinearBounded


# ── Helpers ──────────────────────────────────────────────────────────────────

def _iso_params(w_pos: torch.Tensor, w_neg_abs: torch.Tensor, bias, a_min: float, a_max: float):
    """
    Compute b_prime and c for an isomorphic layer with inputs in [a_min, a_max].

        b_adj[i] = b[i] - a_max * neg_sum[i] - |a_min| * pos_sum[i]
        c[i]     = max(0, -b_adj[i])   (per-neuron activation shifting point)
        b_prime  = b_adj + c            (>= 0)
    """
    abs_amin = abs(a_min)
    pos_sum  = w_pos.sum(dim=1)
    neg_sum  = w_neg_abs.sum(dim=1)
    b = bias if bias is not None else neg_sum.new_zeros(neg_sum.shape)
    b_adj = b - a_max * neg_sum - abs_amin * pos_sum
    c = (-b_adj).clamp(min=0)
    return b_adj + c, c


# ── Isomorphic Layers ─────────────────────────────────────────────────────────

class IsomorphicLinear(nn.Module):
    """
    Equivalent to nn.Linear; uses non-negative weight matrices.
    Supports inputs in [a_min, a_max].

    y_nn = W_pos @ (x + |a_min|) + W_neg_abs @ (a_max - x) + b'
    y    = y_nn - c
    """

    def __init__(self, linear: nn.Linear, a_min: float = 0.0, a_max: float = 1.0):
        super().__init__()
        w = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        W_pos     = w.clamp(min=0)
        W_neg_abs = (-w).clamp(min=0)
        b_prime, c = _iso_params(W_pos, W_neg_abs, bias, a_min, a_max)

        self.W_pos     = nn.Parameter(W_pos)
        self.W_neg_abs = nn.Parameter(W_neg_abs)
        self.register_buffer('b_prime',  b_prime)
        self.register_buffer('c',        c)
        self.register_buffer('a_max',    torch.tensor(a_max,      dtype=w.dtype))
        self.register_buffer('abs_amin', torch.tensor(abs(a_min), dtype=w.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_nn = F.linear(x + self.abs_amin, self.W_pos) \
             + F.linear(self.a_max - x,    self.W_neg_abs, self.b_prime)
        return y_nn - self.c

class IsomorphicConv1D(nn.Module):
    """
    Equivalent to a depthwise Conv1d; uses non-negative kernel.
    Only depthwise (groups == in_channels) supported..

    y = conv(x + |a_min|, w_pos) + conv(a_max - x, w_neg_abs) + bias_corr
    """
    def __init__(self, conv: nn.Conv1d, a_min: float = 0.0, a_max: float = 1.0):
        super().__init__()
        assert conv.groups == conv.in_channels, "Only depthwise conv supported"
        abs_amin = abs(a_min)
        w         = conv.weight.data                  # (C, 1, K)
        w_pos     = w.clamp(min=0)
        w_neg_abs = (-w).clamp(min=0)
        bias = conv.bias.data if conv.bias is not None else None
        b_prime, c = _iso_params(
            w_pos.view(w.shape[0], -1), w_neg_abs.view(w.shape[0], -1),
            bias, a_min, a_max
        )

        self.w_pos     = nn.Parameter(w_pos)
        self.w_neg_abs = nn.Parameter(w_neg_abs)
        self.register_buffer('b_prime', b_prime)
        self.register_buffer('c', c)
        self.register_buffer('a_max',     torch.tensor(a_max,    dtype=w.dtype))
        self.register_buffer('abs_amin',  torch.tensor(abs_amin, dtype=w.dtype))
        self.groups  = conv.groups
        self.padding = conv.padding[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p  = self.padding
        xp = F.pad(x + self.abs_amin,  (p, p), value=self.abs_amin.item())
        ax = F.pad(self.a_max - x,     (p, p), value=self.a_max.item())
        y_nn  = F.conv1d(xp, self.w_pos,     padding=0, groups=self.groups) \
           + F.conv1d(ax, self.w_neg_abs, bias=self.b_prime, padding=0, groups=self.groups)
        return y_nn - self.c[:, None] # reshape (C, 1) for broadcasting


# ── Isomorphic Selective Scan ─────────────────────────────────────────────────

def isomorphic_selective_scan(
    u,
    delta,
    A,
    B_nn, c_B,
    C_nn, c_C,
    D_pos=None, D_neg_abs=None, c_D=None,
    abs_amin=None, a_max=None,
    return_last_state=False,
):
    """
    SSM with isomorphic B, C, and D decomposition for inputs u in [a_min, a_max].

    B = B_nn - c_B,  C = C_nn - c_C  (c_B, c_C >= 0 constants)

    Dual hidden state:
        h'_t      = deltaA * h'_{t-1}      + deltaB_nn_t * u_t    [non-neg increments]
        h_delta_t = deltaA * h_delta_{t-1} + delta_t * c_B * u_t  [non-neg increments]

    Four non-negative output terms:
        y'  = C_nn @ h'          y1 = C_nn @ h_delta
        y2  = c_C  @ h'          y3 = c_C  @ h_delta

    Final: y = y' - (y1 + y2 - y3)  ==  C @ h  (mathematically identical to original) ✓

    Generalized isomorphic D skip (u in [a_min, a_max]):
        D * u = D_pos * (u + |a_min|) + D_neg_abs * (a_max - u) - c_D
        c_D   = D_pos * |a_min| + D_neg_abs * a_max

    Shapes: u (B,D,L), delta (B,D,L), A (D,N), B_nn/C_nn (B,N,L), c_B/c_C (N,)
    """
    dtype_in = u.dtype
    u     = u.float()
    delta = delta.float()
    A     = A.float()
    B_nn  = B_nn.float()
    C_nn  = C_nn.float()
    c_B   = c_B.float()
    c_C   = c_C.float()

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    h_prime = A.new_zeros(batch, dim, dstate)   # (B, D, N)
    h_delta = A.new_zeros(batch, dim, dstate)   # (B, D, N)

    # Discretization
    deltaA     = torch.exp(-torch.einsum('bdl,dn->bdln', delta, A))
    deltaBu_nn = torch.einsum('bdl,bnl,bdl->bdln', delta, B_nn, u)
    ys = []
    last_state = None

    for i in range(seqlen):
        dA_i = deltaA[:, :, i]   # (B, D, N)

        # Update non-negative state: h' = deltaA * h' + deltaB_nn_t
        h_prime = dA_i * h_prime + deltaBu_nn[:, :, i]

        # Update correction state: h_delta = deltaA * h_delta + delta_t * c_B * u_t
        h_delta = dA_i * h_delta + delta[:, :, i, None] * c_B[None, None, :] * u[:, :, i, None]

        # Four non-negative output terms
        C_t  = C_nn[:, :, i]     # (B, N)
        y_nn = torch.einsum('bn,bdn->bd', C_t, h_prime)
        y1   = torch.einsum('bn,bdn->bd', C_t, h_delta)
        y2   = torch.einsum('n,bdn->bd',  c_C, h_prime)
        y3   = torch.einsum('n,bdn->bd',  c_C, h_delta)
        ys.append(y_nn - (y1 + y2 - y3))

        if i == seqlen - 1:
            last_state = h_prime.clone()

    y = torch.stack(ys, dim=2)   # (B, D, L)

    # Generalized isomorphic D skip: D*u = D_pos*(u+|amin|) + D_neg_abs*(amax-u) - c_D
    if D_pos is not None and D_neg_abs is not None and c_D is not None:
        D_p = rearrange(D_pos.float(),     'd -> d 1')
        D_n = rearrange(D_neg_abs.float(), 'd -> d 1')
        c_d = rearrange(c_D.float(),       'd -> d 1')
        out = y + D_p * (u + abs_amin) + D_n * (a_max - u) - c_d
    else:
        out = y

    out = out.to(dtype_in)
    return out if not return_last_state else (out, last_state)


# ── NN-Transformed PhotonicMamba ──────────────────────────────────────────────

class NNPhotonicMamba(nn.Module):
    """
    PhotonicMamba with all weight matrices isomorphically transformed to be non-negative.
    Mathematically equivalent to the original for inputs in [a_min, a_max].

    Inter-component bounding enforces the input range assumption at every layer boundary.
    Two bounding strategies (select via inter_bounding):
        'linear_bounded' — LinearBounded clamp to [a_min, a_max]  (default)
        'norm'           — RMSNorm (see commented blocks in forward)
        None             — no bounding (for testing equivalence on known-bounded inputs)

    Replaces:
        in_proj, x_proj, dt_proj, out_proj → IsomorphicLinear
        conv1d                             → IsomorphicConv1D
        SSM B, C → dual-state recurrence (c_B, c_C sliced from x_proj.c)
        SSM D    → isomorphic element-wise decomposition (D_pos, D_neg_abs)
    """
    def __init__(self, source, a_min: float = 0.0, a_max: float = 1.0,
                 inter_bounding: str = 'linear_bounded'):
        super().__init__()
        self.d_model  = source.d_model
        self.d_state  = source.d_state
        self.d_conv   = source.d_conv
        self.expand   = source.expand
        self.d_inner  = source.d_inner
        self.dt_rank  = source.dt_rank

        # ── Inter-component bounding ────────────────────────────────────────
        if inter_bounding == 'linear_bounded':
            self.bound = LinearBounded(a_min, a_max)
        elif inter_bounding == 'norm':
            # RMSNorm alternative — does not enforce hard [a_min, a_max] bounds,
            # but normalizes activations. Useful to compare vs LinearBounded.
            self.bound_in_proj = nn.RMSNorm(self.d_inner)
            self.bound_conv    = nn.RMSNorm(self.d_inner)
            self.bound_ssm     = nn.RMSNorm(self.d_inner)
            self.bound = None   # forward uses per-site norm attributes
        else:
            self.bound = nn.Identity()

        self._inter_bounding = inter_bounding

        # ── Isomorphic projections ──────────────────────────────────────────
        self.in_proj = IsomorphicLinear(
            source.in_proj.weight.data,
            source.in_proj.bias.data if source.in_proj.bias is not None else None,
            a_min=a_min, a_max=a_max,
        )
        self.conv1d  = IsomorphicConv1D(source.conv1d, a_min=a_min, a_max=a_max)
        self.x_proj  = IsomorphicLinear(
            source.x_proj.weight.data,
            source.x_proj.bias.data if source.x_proj.bias is not None else None,
            a_min=a_min, a_max=a_max,
        )

        # dt_proj: weight only — original bias stored separately.
        # dt_c_correction = W_dt @ c_dt corrects for the x_proj per-neuron shift on the dt slice.
        c_dt_slice = self.x_proj.c[:self.dt_rank].clone()           # (dt_rank,)
        self.register_buffer(
            'dt_c_correction',
            source.dt_proj.weight.data @ c_dt_slice,                # (d_inner,)
        )
        self.dt_proj = IsomorphicLinear(
            source.dt_proj.weight.data, bias=None, a_min=a_min, a_max=a_max,
        )
        self.register_buffer('dt_proj_bias', source.dt_proj.bias.data.clone())

        self.out_proj = IsomorphicLinear(
            source.out_proj.weight.data,
            source.out_proj.bias.data if source.out_proj.bias is not None else None,
            a_min=a_min, a_max=a_max,
        )

        # ── SSM A — exp(A_log) > 0 always, no transformation needed ─────────
        self.A_log = nn.Parameter(source.A_log.data.clone())

        # ── Isomorphic D skip: D = D_pos - D_neg_abs ────────────────────────
        abs_amin       = abs(a_min)
        D              = source.D.data.clone()
        self.D_pos     = nn.Parameter(D.clamp(min=0))
        self.D_neg_abs = nn.Parameter((-D).clamp(min=0))
        # c_D = D_pos * |a_min| + D_neg_abs * a_max  (generalized correction)
        self.register_buffer('c_D', abs_amin * self.D_pos.data + a_max * self.D_neg_abs.data)
        self.register_buffer('abs_amin', torch.tensor(abs_amin, dtype=torch.float32))
        self.register_buffer('a_max',    torch.tensor(a_max,    dtype=torch.float32))

        # ── Activations (stateless, safe to share references) ───────────────
        self.act              = source.act
        self.delta_activation = source.delta_activation
        self.gate_activation  = source.gate_activation

    def _apply_bound(self, x, site: str):
        """Apply inter-component bounding at the given site ('in_proj','conv','ssm')."""
        if self._inter_bounding == 'norm':
            # Norm operates on (B, L, D) — transpose, normalize, transpose back
            norm = getattr(self, f'bound_{site}')
            return norm(x.transpose(1, 2)).transpose(1, 2)
        return self.bound(x)

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("inference_params not supported in NNPhotonicMamba")

        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(
            self.in_proj(rearrange(hidden_states, "b l d -> (b l) d")), 
            "(b l) d -> b d l",
            l=seqlen
            )
        
        A = torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        # ── Bound x before conv1d ────────────────────────────────────────────
        x = self._apply_bound(x, 'in_proj')
        # ── Norm alternative: comment _apply_bound above, uncomment below ──
        # x = self.bound_in_proj(x.transpose(1,2)).transpose(1,2)

        # ── conv1d (isomorphic) + activation ────────────────────────────────
        x = self.act(self.conv1d(x)[..., :seqlen])                  # (B, d_inner, L)

        # ── Bound x before x_proj ────────────────────────────────────────────
        x = self._apply_bound(x, 'conv')
        # ── Norm alternative: comment _apply_bound above, uncomment below ──
        # x = self.bound_conv(x.transpose(1,2)).transpose(1,2)

        # ── x_proj (isomorphic) ─────────────────────────────────────────────
        # x_proj.c is the per-neuron correction vector (shape: dt_rank+2*d_state).
        # Slicing recovers c_B, c_C, c_dt for the isomorphic SSM scan.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))      # (B*L, dt_rank+2*d_state)
        dt_raw, B_raw, C_raw = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        c_B  = self.x_proj.c[self.dt_rank:self.dt_rank + self.d_state]   # (d_state,)
        c_C  = self.x_proj.c[self.dt_rank + self.d_state:]                # (d_state,)
        c_dt = self.x_proj.c[:self.dt_rank]                               # (dt_rank,)

        B_nn      = B_raw  + c_B    # (B*L, d_state), non-negative
        C_nn      = C_raw  + c_C    # (B*L, d_state), non-negative
        dt_raw_nn = dt_raw + c_dt   # (B*L, dt_rank), non-negative

        # ── dt_proj (isomorphic, weight only) ───────────────────────────────
        # delta = act(W_dt @ dt_raw + dt_proj_bias)
        #       = act(W_dt @ (dt_raw_nn - c_dt) + dt_proj_bias)
        #       = act(dt_nn - dt_c_correction + dt_proj_bias)
        dt_nn = rearrange(
            self.dt_proj(dt_raw_nn), "(b l) d -> b d l", l=seqlen  # (B, d_inner, L)
        )
        delta = self.delta_activation(
            dt_nn
            - self.dt_c_correction[:, None]     # (d_inner, 1) broadcasts over L
            + self.dt_proj_bias[:, None]
        ).clamp(min=1e-4, max=20.0)             # (B, d_inner, L)

        B_nn = rearrange(B_nn, "(b l) n -> b n l", l=seqlen).contiguous()  # (B, d_state, L)
        C_nn = rearrange(C_nn, "(b l) n -> b n l", l=seqlen).contiguous()  # (B, d_state, L)

        # ── SSM with isomorphic B, C, D ─────────────────────────────────────
        y = isomorphic_selective_scan(
            u=x, delta=delta, A=A,
            B_nn=B_nn, c_B=c_B,
            C_nn=C_nn, c_C=c_C,
            D_pos=self.D_pos, D_neg_abs=self.D_neg_abs, c_D=self.c_D,
            abs_amin=float(self.abs_amin), a_max=float(self.a_max),
            return_last_state=False,
        )                                                            # (B, d_inner, L)

        # ── Bound y before out_proj ───────────────────────────────────────────
        y = self._apply_bound(y, 'ssm')
        # ── Norm alternative: comment _apply_bound above, uncomment below ──
        # y = self.bound_ssm(y.transpose(1,2)).transpose(1,2)

        # ── out_proj (isomorphic) ────────────────────────────────────────────
        return self.out_proj(rearrange(y, "b d l -> b l d"))


# ── Factory ───────────────────────────────────────────────────────────────────

def _replace_head_linears(head: nn.Sequential, a_min: float, a_max: float) -> nn.Sequential:
    """Replace nn.Linear layers inside an nn.Sequential with IsomorphicLinear."""
    new_layers = []
    for layer in head:
        if isinstance(layer, nn.Linear):
            new_layers.append(IsomorphicLinear(
                layer.weight.data,
                layer.bias.data if layer.bias is not None else None,
                a_min=a_min, a_max=a_max,
            ))
        else:
            new_layers.append(layer)
    return nn.Sequential(*new_layers)


def transform_to_nn(model, a_min: float = 0.0, a_max: float = 1.0,
                    inter_bounding: str = 'linear_bounded') -> nn.Module:
    """
    Return a deep-copied MambaModel with all weight matrices isomorphically transformed
    to be non-negative. Mathematically equivalent to the original for inputs in [a_min, a_max].
    No retraining required.

    Replaces:
      - Each PhotonicMamba block  → NNPhotonicMamba
      - MLP head nn.Linear layers → IsomorphicLinear

    RMSNorm layers are left unchanged.

    Args:
        model:          trained MambaModel instance
        a_min:          lower bound on layer inputs (can be negative)
        a_max:          upper bound on layer inputs
        inter_bounding: 'linear_bounded' | 'norm' | None

    Returns:
        New model (deepcopy) ready for non-negative photonic inference.
    """
    from models.photonic_model import PhotonicMamba

    model_nn = copy.deepcopy(model)

    for i, block in enumerate(model_nn.mamba_layers):
        if isinstance(block, PhotonicMamba):
            model_nn.mamba_layers[i] = NNPhotonicMamba(
                block, a_min=a_min, a_max=a_max, inter_bounding=inter_bounding,
            )
        else:
            warnings.warn(
                f"mamba_layers[{i}] is {type(block).__name__}, not PhotonicMamba — skipped."
            )

    if model_nn.head is not None and isinstance(model_nn.head, nn.Sequential):
        model_nn.head = _replace_head_linears(model_nn.head, a_min=a_min, a_max=a_max)

    return model_nn
