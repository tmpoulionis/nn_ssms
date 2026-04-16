"""
Non-negativity validation tool for isomorphic Mamba models.

Runs a forward pass through a transformed model and checks that every
matrix-multiplication operand is non-negative.  Produces a structured
report with per-check pass/fail, violation counts, and minimum values.

Usage:
    from nnt.validate_nn import validate_non_negativity

    report = validate_non_negativity(model_nn, sample_input, atol=1e-7)
    print(report.summary())
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nnt.nn_linear import IsomorphicLinear
from nnt.nn_conv1d import IsomorphicConv1D
from nnt.nn_mamba_block import NNMambaBlock


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single non-negativity check."""
    layer_name: str       # e.g. "mamba_layers.0.in_proj"
    check_name: str       # e.g. "W_pos >= 0"
    passed: bool
    num_violations: int   # count of elements < -atol
    min_value: float      # most-negative value in the tensor
    tensor_numel: int     # total number of elements


@dataclass
class ValidationReport:
    """Aggregated results from all non-negativity checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures_only(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        n_passed = sum(c.passed for c in self.checks)
        n_failed = len(self.checks) - n_passed
        status = "PASSED" if self.all_passed else "FAILED"

        lines = [
            f"Non-Negativity Validation: {status}",
            f"Total checks: {len(self.checks)}, Passed: {n_passed}, Failed: {n_failed}",
            "",
            f"{'Layer':<50} {'Check':<25} {'Status':<8} {'Violations':<12} {'Min Value'}",
            "-" * 110,
        ]
        for c in self.checks:
            tag = "OK" if c.passed else "FAIL"
            lines.append(
                f"{c.layer_name:<50} {c.check_name:<25} {tag:<8} "
                f"{c.num_violations:<12} {c.min_value:.6e}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_tensor(layer_name: str, check_name: str, t: torch.Tensor, atol: float) -> CheckResult:
    violations = (t < -atol)
    return CheckResult(
        layer_name=layer_name,
        check_name=check_name,
        passed=not violations.any().item(),
        num_violations=int(violations.sum().item()),
        min_value=float(t.min().item()),
        tensor_numel=t.numel(),
    )


# ---------------------------------------------------------------------------
# 1. Static parameter / buffer checks
# ---------------------------------------------------------------------------

def _check_static_params(model: torch.nn.Module, atol: float) -> list[CheckResult]:
    results = []
    for name, module in model.named_modules():
        if isinstance(module, IsomorphicLinear):
            results.append(_check_tensor(name, "W_pos >= 0", module.W_pos.data, atol))
            results.append(_check_tensor(name, "W_neg_abs >= 0", module.W_neg_abs.data, atol))
            results.append(_check_tensor(name, "b_prime >= 0", module.b_prime, atol))
        elif isinstance(module, IsomorphicConv1D):
            results.append(_check_tensor(name, "w_pos >= 0", module.w_pos.data, atol))
            results.append(_check_tensor(name, "w_neg_abs >= 0", module.w_neg_abs.data, atol))
            results.append(_check_tensor(name, "b_prime >= 0", module.b_prime, atol))
        elif isinstance(module, NNMambaBlock):
            results.append(_check_tensor(name, "D_pos >= 0", module.D_pos.data, atol))
            results.append(_check_tensor(name, "D_neg_abs >= 0", module.D_neg_abs.data, atol))
            results.append(_check_tensor(name, "c_D >= 0", module.c_D, atol))
        elif isinstance(module, nn.RMSNorm):
            results.append(_check_tensor(name, "gamma >= 0", module.weight.data, atol))
    return results


# ---------------------------------------------------------------------------
# 2. Forward hooks for IsomorphicLinear / IsomorphicConv1D
# ---------------------------------------------------------------------------

def _linear_hook(module, input, output, *, results: list, name: str, atol: float):
    x = input[0]
    results.append(_check_tensor(name, "x + abs_amin >= 0", x + module.abs_amin, atol))
    results.append(_check_tensor(name, "a_max - x >= 0", module.a_max - x, atol))


def _conv1d_hook(module, input, output, *, results: list, name: str, atol: float):
    x = input[0]
    p = module.padding
    xp = F.pad(x + module.abs_amin, (p, p), value=module.abs_amin.item())
    ax = F.pad(module.a_max - x, (p, p), value=module.a_max.item())
    results.append(_check_tensor(name, "pad(x+abs_amin) >= 0", xp, atol))
    results.append(_check_tensor(name, "pad(a_max-x) >= 0", ax, atol))


# ---------------------------------------------------------------------------
# 3. Gate check: capture in_proj output, re-derive gate in block post-hook
# ---------------------------------------------------------------------------

def _in_proj_capture_hook(module, input, output, *, storage: dict, key: str):
    storage[key] = output.detach()


def _block_gate_hook(module, input, output, *, results: list, name: str, atol: float, storage: dict):
    key = f"{name}._in_proj_out"
    in_proj_out = storage.pop(key, None)
    if in_proj_out is None:
        return
    hidden_states = input[0]
    seqlen = hidden_states.shape[1]
    xz = module._apply_bound(
        rearrange(in_proj_out, "(b l) d -> b d l", l=seqlen)
    )
    _x, z = xz.chunk(2, dim=1)
    gate = module.gate_act(z)
    results.append(_check_tensor(name, "gate >= 0", gate, atol))


# ---------------------------------------------------------------------------
# 4. Selective-scan wrapper
# ---------------------------------------------------------------------------

def _make_checking_scan(original_fn, results: list, block_names: list[str], atol: float):
    call_idx = [0]

    def checked_scan(u, delta, A, B_nn, c_B, C_nn, c_C,
                     D_pos=None, D_neg_abs=None, c_D=None,
                     abs_amin=None, a_max=None, **kwargs):
        idx = min(call_idx[0], len(block_names) - 1)
        bname = f"{block_names[idx]}.sscan"
        call_idx[0] += 1

        results.append(_check_tensor(bname, "u >= 0", u, atol))
        results.append(_check_tensor(bname, "delta >= 0", delta, atol))
        results.append(_check_tensor(bname, "B_nn >= 0", B_nn, atol))
        results.append(_check_tensor(bname, "C_nn >= 0", C_nn, atol))
        results.append(_check_tensor(bname, "c_B >= 0", c_B, atol))
        results.append(_check_tensor(bname, "c_C >= 0", c_C, atol))

        if D_pos is not None:
            results.append(_check_tensor(bname, "D_pos >= 0", D_pos, atol))
        if D_neg_abs is not None:
            results.append(_check_tensor(bname, "D_neg_abs >= 0", D_neg_abs, atol))
        if abs_amin is not None:
            results.append(_check_tensor(bname, "u + abs_amin >= 0", u + abs_amin, atol))
        if a_max is not None:
            results.append(_check_tensor(bname, "a_max - u >= 0", a_max - u, atol))

        return original_fn(
            u, delta, A, B_nn, c_B, C_nn, c_C,
            D_pos=D_pos, D_neg_abs=D_neg_abs, c_D=c_D,
            abs_amin=abs_amin, a_max=a_max, **kwargs,
        )

    return checked_scan


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_non_negativity(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    atol: float = 0.0,
) -> ValidationReport:
    """
    Run a forward pass and check every matmul operand for non-negativity.

    Args:
        model:        A transformed model (containing IsomorphicLinear,
                      IsomorphicConv1D, and/or NNMambaBlock modules).
        sample_input: A representative input tensor, shape (B, L, D).
        atol:         Absolute tolerance. Values >= -atol count as non-negative.

    Returns:
        ValidationReport with per-check results.
    """
    model.eval()
    results: list[CheckResult] = []
    handles: list = []
    storage: dict = {}

    # -- 1. Static parameter checks ------------------------------------------
    results.extend(_check_static_params(model, atol))

    # -- 2. Register hooks on isomorphic layers -------------------------------
    for name, module in model.named_modules():
        if isinstance(module, IsomorphicLinear):
            h = module.register_forward_hook(
                functools.partial(_linear_hook, results=results, name=name, atol=atol)
            )
            handles.append(h)

        elif isinstance(module, IsomorphicConv1D):
            h = module.register_forward_hook(
                functools.partial(_conv1d_hook, results=results, name=name, atol=atol)
            )
            handles.append(h)

        elif isinstance(module, NNMambaBlock):
            # Capture in_proj output for gate derivation
            key = f"{name}._in_proj_out"
            h = module.in_proj.register_forward_hook(
                functools.partial(_in_proj_capture_hook, storage=storage, key=key)
            )
            handles.append(h)
            # Post-hook on block to check gate
            h = module.register_forward_hook(
                functools.partial(
                    _block_gate_hook,
                    results=results, name=name, atol=atol, storage=storage,
                )
            )
            handles.append(h)

    # -- 3. Monkey-patch selective scan for operand checking -------------------
    import nnt.nn_mamba_block as _block_mod

    orig_scan = _block_mod.isomorphic_selective_scan
    block_names = [n for n, m in model.named_modules() if isinstance(m, NNMambaBlock)]

    if block_names:
        _block_mod.isomorphic_selective_scan = _make_checking_scan(
            orig_scan, results, block_names, atol,
        )

    # -- 4. Forward pass ------------------------------------------------------
    try:
        model(sample_input)
    finally:
        for h in handles:
            h.remove()
        _block_mod.isomorphic_selective_scan = orig_scan

    return ValidationReport(checks=results)
