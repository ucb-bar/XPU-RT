"""torchAO-compatible FP8 E4M3 quantization config with power-of-two scaling.

Provides ``FP8E4M3Po2Config``, an ``AOBaseConfig`` subclass that can be used
with ``torchao.quantization.quantize_()``:

    >>> from torchao.quantization import quantize_
    >>> from compgen.quantization import FP8E4M3Po2Config
    >>> quantize_(model, FP8E4M3Po2Config())

The handler replaces ``nn.Linear`` weights with ``FP8E4M3Po2Tensor``, which
stores weights in ``torch.float8_e4m3fn`` with a per-tensor power-of-two
scale.  This matches the NPU's E8M0 scale registers and ``vmatmul``
accumulation model.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import register_quantize_module_handler

from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor


@dataclass
class FP8E4M3Po2Config(AOBaseConfig):
    """Configuration for FP8 E4M3 quantization with power-of-two scaling.

    Attributes:
        scaling_mode: ``"po2"`` for power-of-two scales (NPU-optimal) or
            ``"scaled"`` for absmax scales.
        quantize_activations: Whether to dynamically quantize activations
            to FP8 during forward (always True for NPU deployment).
    """

    scaling_mode: Literal["po2", "scaled"] = "po2"
    quantize_activations: bool = True


def _module_extra_repr(
    self: nn.Module,
    *,
    original_extra_repr: str,
    parameter_name: str,
) -> str:
    """Enhanced repr showing FP8 quantization info."""
    param = getattr(self, parameter_name, None)
    if isinstance(param, FP8E4M3Po2Tensor):
        return f"{original_extra_repr}, weight_dtype=float8_e4m3fn, scale={param._scale}"
    return original_extra_repr


@register_quantize_module_handler(FP8E4M3Po2Config)
def _fp8_e4m3_po2_transform(
    module: nn.Module,
    config: FP8E4M3Po2Config,
    *,
    parameter_name: str = "weight",
) -> nn.Module:
    """Transform handler: replace module weight with FP8E4M3Po2Tensor.

    Called by ``quantize_()`` for each ``nn.Linear`` (or whichever modules
    match the filter function).

    Args:
        module: The ``nn.Module`` whose weight to quantize.
        config: The quantization config.
        parameter_name: Name of the parameter to quantize.

    Returns:
        The same module with its weight replaced by ``FP8E4M3Po2Tensor``.
    """
    if not hasattr(module, parameter_name):
        return module

    weight = getattr(module, parameter_name)
    if isinstance(weight, FP8E4M3Po2Tensor):
        return module  # Already quantized

    fp8_weight = FP8E4M3Po2Tensor.from_float(weight)
    setattr(
        module,
        parameter_name,
        nn.Parameter(fp8_weight, requires_grad=False),
    )

    # Enhance repr to show quantization info
    module.extra_repr = types.MethodType(
        partial(
            _module_extra_repr,
            original_extra_repr=module.extra_repr(),
            parameter_name=parameter_name,
        ),
        module,
    )

    return module


__all__ = ["FP8E4M3Po2Config"]
