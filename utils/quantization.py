"""
Quantization Aware Training (QAT) for the photonic Mamba project.

Config cheatsheet — keys under config["quantization"]:
    enabled               : bool             — master on/off switch
    bits                  : int              — bit width for all fake-quants
    symmetric             : bool             — signed range if True, unsigned if False (input/weight/output)
    per_channel_weights   : bool             — one scale per output channel for weights
    observer_momentum     : float            — EMA alpha for input/output observers (e.g. 0.01)
    tensors               : dict[str, bool]  — which tensors to quantize: input / weight / bias / output
    skip_modules          : list[str]        — substring blocklist on layer names (e.g. ["norm"])
    quantize_embeddings   : bool             — wrap nn.Embedding layers too
    calibration_steps     : int              — observe-only steps before fake-quant turns on
    freeze_observer_step  : int | None       — step at which to freeze observer EMAs (None = never)
    train_quant           : bool             — apply fake-quant during training
    eval_quant            : bool             — apply fake-quant during val/test
"""
import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


def _qrange(bits: int, symmetric: bool):
    if symmetric:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def make_weight_fake_quant(bits: int, symmetric: bool, per_channel: bool, ch_axis: int = 0):
    qmin, qmax = _qrange(bits, symmetric)
    dtype = torch.qint8 if symmetric else torch.quint8
    if per_channel:
        return FakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=qmin,
            quant_max=qmax,
            dtype=dtype,
            qscheme=torch.per_channel_symmetric if symmetric else torch.per_channel_affine,
            ch_axis=ch_axis,
            averaging_constant=1.0,
        )
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=dtype,
        qscheme=torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine,
        averaging_constant=1.0,
    )


def make_io_fake_quant(bits: int, symmetric: bool, momentum: float):
    qmin, qmax = _qrange(bits, symmetric)
    dtype = torch.qint8 if symmetric else torch.quint8
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=qmin,
        quant_max=qmax,
        dtype=dtype,
        qscheme=torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine,
        averaging_constant=momentum,
    )


class LayerQuantWrap(nn.Module):
    """
    Hook-based fake-quant wrapper.
    """

    def __init__(self, layer: nn.Module, qcfg: dict, weight_ch_axis: int = 0):
        super().__init__()
        # Keep the layer reference outside nn.Module's submodule registry so it is not
        # double-counted in state_dict.
        self._layer_ref = [layer]
        self.cfg = qcfg
        t = qcfg["tensors"]

        if t.get("input", True):
            self.input_fq = make_io_fake_quant(
                qcfg["activation_bits"],
                qcfg["activations_symmetric"],
                qcfg["observer_momentum"],
            )
        if t.get("weight", True):
            self.weight_fq = make_weight_fake_quant(
                qcfg["weight_bits"],
                qcfg["weights_symmetric"],
                qcfg["per_channel_weights"],
                ch_axis=weight_ch_axis,
            )
        if t.get("bias", True) and getattr(layer, "bias", None) is not None:
            self.bias_fq = make_io_fake_quant(
                qcfg["activation_bits"], symmetric=True, momentum=1.0
            )
        if t.get("output", True):
            self.output_fq = make_io_fake_quant(
                qcfg["activation_bits"],
                qcfg["activations_symmetric"],
                qcfg["observer_momentum"],
            )

        self._is_quant = False
        self._pre_handle = None
        self._post_handle = None
        self._saved_weight = None
        self._saved_bias = None

    @property
    def layer(self):
        return self._layer_ref[0]

    def forward(self, x):
        return self.layer(x)

    def set_quant_mode(self, on: bool):
        if on and not self._is_quant:
            self._pre_handle = self.layer.register_forward_pre_hook(self._pre_hook)
            self._post_handle = self.layer.register_forward_hook(self._post_hook)
            self._is_quant = True 
        elif not on and self._is_quant:
            if self._pre_handle is not None:
                self._pre_handle.remove()
                self._pre_handle = None
            if self._post_handle is not None:
                self._post_handle.remove()
                self._post_handle = None
            self._is_quant = False

    def _pre_hook(self, m, inputs):
        x = inputs[0]
        if hasattr(self, "input_fq"):
            x = self.input_fq(x)
        if hasattr(self, "weight_fq"):
            self._saved_weight = m.weight.data
            m.weight.data = self.weight_fq(m.weight).detach()
        if hasattr(self, "bias_fq") and getattr(m, "bias", None) is not None:
            self._saved_bias = m.bias.data
            m.bias.data = self.bias_fq(m.bias).detach()
        return (x,) + tuple(inputs[1:])

    def _post_hook(self, m, inputs, output):
        if self._saved_weight is not None:
            m.weight.data = self._saved_weight
            self._saved_weight = None
        if self._saved_bias is not None:
            m.bias.data = self._saved_bias
            self._saved_bias = None
        if hasattr(self, "output_fq"):
            output = self.output_fq(output)
        return output


class QuantizationManager(nn.Module):
    """Owns all LayerQuantWrap instances."""

    def __init__(self, model: nn.Module, qcfg: dict):
        super().__init__()
        self.qcfg = qcfg
        self.wraps = nn.ModuleDict()
        skip = qcfg.get("skip_modules", [])
        quantize_embed = qcfg.get("quantize_embeddings", False)

        for name, layer in model.named_modules():
            if any(s in name for s in skip):
                continue
            if isinstance(layer, (nn.Linear, nn.Conv1d)):
                key = name.replace(".", "__") or "_root"
                self.wraps[key] = LayerQuantWrap(layer, qcfg, weight_ch_axis=0)
            elif isinstance(layer, nn.Embedding) and quantize_embed:
                key = name.replace(".", "__") or "_root"
                self.wraps[key] = LayerQuantWrap(layer, qcfg, weight_ch_axis=0)

    def set_quant_mode(self, on: bool):
        for w in self.wraps.values():
            w.set_quant_mode(on)

    def _toggle_fqs(self, method_name: str):
        for w in self.wraps.values():
            for fq_name in ("input_fq", "weight_fq", "bias_fq", "output_fq"):
                fq = getattr(w, fq_name, None)
                if fq is not None:
                    getattr(fq, method_name)()

    def enable_observer(self):
        self._toggle_fqs("enable_observer")

    def disable_observer(self):
        self._toggle_fqs("disable_observer")

    def enable_fake_quant(self):
        self._toggle_fqs("enable_fake_quant")

    def disable_fake_quant(self):
        self._toggle_fqs("disable_fake_quant")

    def freeze_observers(self):
        self.disable_observer()
