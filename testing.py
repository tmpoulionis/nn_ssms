"""
QAT Pipeline Trace
Builds a 1-layer SelectiveCopyingMambaModel with quantization on, registers verbose
hooks on every nn.Module, monkey-patches selective_scan_fn, runs a tiny dummy sample,
and prints a step-by-step narration of every computation showing where quantization
applies and where the path stays fp32.
"""
import torch
import torch.nn as nn
from utils.utils import load_config
from utils.quantization import QuantizationManager
import mamba.selective_scan as ss_mod
import mamba.mamba_block as mb_mod

# --------------------------------------------------------------------------- 1
cfg = load_config("testing")
ModelCls = cfg["model_class"]
model = ModelCls(**cfg["model"])
model.eval()

# --------------------------------------------------------------------------- 2
qcfg = cfg["quantization"]
manager = QuantizationManager(model, qcfg)
manager.set_quant_mode(True)
manager.enable_observer()
manager.disable_fake_quant()                                 # calibration-only first
with torch.no_grad():
    for _ in range(3):
        model(torch.randint(0, qcfg.get("vocab_size", 10) or cfg["model"]["vocab_size"], (2, 16)))
manager.enable_fake_quant()                                  # now both observer + fake-quant on

# # Map id(layer) -> wrap so the tracer can find each layer's quant info
wrap_for = {id(w.layer): w for w in manager.wraps.values()}

# # --------------------------------------------------------------------------- 3  pretty-printers
def tinfo(t, name="t"):
    if not torch.is_tensor(t):
        return f"{name}={t!r}"
    flat = t.detach().reshape(-1)
    if flat.is_floating_point():
        f = flat.float()
        return (f"{name} shape={tuple(t.shape)} dtype={t.dtype} "
                f"min={f.min().item():+.4g} max={f.max().item():+.4g} "
                f"mean={f.mean().item():+.4g} unique={flat.unique().numel()}")
    return (f"{name} shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={flat.min().item()} max={flat.max().item()} "
            f"unique={flat.unique().numel()}")

def quant_status(fq, label):
    if fq is None:
        return f"{label}: SKIP (not built)"
    obs_on = bool(fq.observer_enabled.item())
    fq_on = bool(fq.fake_quant_enabled.item())
    sc = fq.scale.detach().reshape(-1)
    zp = fq.zero_point.detach().reshape(-1)
    sc_show = [round(x, 4) for x in sc[:3].tolist()]
    zp_show = zp[:3].tolist()
    return (f"{label}: obs={int(obs_on)} fq={int(fq_on)} "
            f"qmin={fq.quant_min} qmax={fq.quant_max} "
            f"scale[..3]={sc_show} zp[..3]={zp_show}")

# --------------------------------------------------------------------------- 4  hooks
indent = [0]
PAD = "  "

def make_pre_hook(name):
    def pre(m, inputs):
        prefix = PAD * indent[0]
        x = inputs[0] if inputs else None
        print(f"{prefix}>>> {name or '(root)'} ({type(m).__name__})")
        print(f"{prefix}{PAD}IN : {tinfo(x, 'x')}")
        wrap = wrap_for.get(id(m))
        if wrap is not None:
            print(f"{prefix}{PAD}QUANT pre-op (will fire after this print):")
            print(f"{prefix}{PAD*2}{quant_status(getattr(wrap, 'input_fq',  None), 'input_fq ')}")
            print(f"{prefix}{PAD*2}{quant_status(getattr(wrap, 'weight_fq', None), 'weight_fq')}")
            print(f"{prefix}{PAD*2}{quant_status(getattr(wrap, 'bias_fq',   None), 'bias_fq  ')}")
        else:
            print(f"{prefix}{PAD}QUANT: NOT wrapped — fp32 forward")
        indent[0] += 1
    return pre

def make_post_hook(name):
    def post(m, inputs, output):
        indent[0] -= 1
        prefix = PAD * indent[0]
        out_t = output if torch.is_tensor(output) else (output[0] if isinstance(output, tuple) else None)
        print(f"{prefix}{PAD}OUT: {tinfo(out_t, 'y')}")
        wrap = wrap_for.get(id(m))
        if wrap is not None:
            print(f"{prefix}{PAD}QUANT post-op:")
            print(f"{prefix}{PAD*2}{quant_status(getattr(wrap, 'output_fq', None), 'output_fq')}")
        print(f"{prefix}<<< {name or '(root)'}")
    return post

handles = []
for name, m in model.named_modules():
    if m is model:                                            # skip top-level
        continue
    handles.append(m.register_forward_pre_hook(make_pre_hook(name), prepend=True))
    handles.append(m.register_forward_hook(make_post_hook(name)))

# --------------------------------------------------------------------------- 5  selective_scan_fn tracer
orig_scan = ss_mod.selective_scan_fn

def traced_scan(u, delta, A, B, C, D=None, return_last_state=False):
    pfx = PAD * indent[0]
    print(f"{pfx}--- selective_scan_fn ENTER ---")
    print(f"{pfx}{PAD}{tinfo(u,     'u    ')}    (came from quantized conv1d output_fq)")
    print(f"{pfx}{PAD}{tinfo(delta, 'delta')}    (came from quantized dt_proj output_fq)")
    print(f"{pfx}{PAD}{tinfo(A,     'A    ')}    (FP32 — derived from A_log, NOT quantized)")
    print(f"{pfx}{PAD}{tinfo(B,     'B    ')}    (came from quantized x_proj output_fq slice)")
    print(f"{pfx}{PAD}{tinfo(C,     'C    ')}    (came from quantized x_proj output_fq slice)")
    if D is not None:
        print(f"{pfx}{PAD}{tinfo(D, 'D    ')}    (FP32 — D parameter, NOT quantized)")

    # Show the einsum results without altering the actual computation
    u_f = u.float(); delta_f = delta.float(); A_f = A.float(); B_f = B.float(); C_f = C.float()
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta_f, B_f, u_f)
    deltaA   = torch.exp(-torch.einsum('bdl,dn->bdln', delta_f, A_f))
    print(f"{pfx}{PAD}einsum 'bdl,bnl,bdl->bdln'  → {tinfo(deltaB_u, 'deltaB_u')}")
    print(f"{pfx}{PAD}exp(-einsum 'bdl,dn->bdln') → {tinfo(deltaA,   'deltaA  ')}")
    print(f"{pfx}{PAD}scan loop: h_{{t+1}} = deltaA[:,:,t] * h_t + deltaB_u[:,:,t]   (FP32, accumulator)")
    print(f"{pfx}{PAD}              y_t   = einsum 'bdn,bn->bd'(h_{{t+1}}, C[:,:,t])  (FP32)")

    out = orig_scan(u, delta, A, B, C, D=D, return_last_state=return_last_state)
    out_t = out[0] if isinstance(out, tuple) else out
    print(f"{pfx}{PAD}{tinfo(out_t, 'scan_out (= y + u*D if D is not None)')}")
    print(f"{pfx}--- selective_scan_fn EXIT ---")
    return out

ss_mod.selective_scan_fn = traced_scan
mb_mod.selective_scan_fn = traced_scan

# --------------------------------------------------------------------------- 6  run a dummy sample
try:
    print("=" * 90)
    print("FORWARD PASS — full QAT trace")
    print("=" * 90)
    x = torch.randint(0, cfg["model"]["vocab_size"], (1, 8))
    print(f"input tokens: {x.tolist()}   shape={tuple(x.shape)}")
    print("-" * 90)
    with torch.no_grad():
        y = model(x)
    print("-" * 90)
    print(f"output {tinfo(y, 'logits')}")
finally:
    # Restore everything
    ss_mod.selective_scan_fn = orig_scan
    mb_mod.selective_scan_fn = orig_scan
    for h in handles:
        h.remove()
