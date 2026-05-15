"""Template: Custom Quantization Method

Copy this file into the ``methods/`` directory and implement your
quantization scheme as a torchAO-compatible config.

See ``compgen.quantization.fp8_config`` for a working example (FP8 E4M3).
See the torchAO docs for the ``AOBaseConfig`` / ``register_quantize_module_handler`` API.

Steps:
    1. Copy this file: ``cp _template.py my_quant.py``
    2. Define your config class extending ``AOBaseConfig``
    3. Register handler with ``@register_quantize_module_handler``
    4. Add scheme name to ``capture/torchao_pipeline.py:apply_quantization()``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Requires: pip install torchao>=0.16
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import register_quantize_module_handler


@dataclass
class TemplateQuantConfig(AOBaseConfig):
    """Template quantization configuration.

    Replace with your own quantization parameters.
    This class is passed to ``torchao.quantization.quantize_(model, config)``.
    """

    bit_width: int = 8
    symmetric: bool = True
    per_channel: bool = False


@register_quantize_module_handler(TemplateQuantConfig)
def _template_quant_transform(
    module: nn.Module,
    config: TemplateQuantConfig,
    *,
    parameter_name: str = "weight",
) -> nn.Module:
    """Transform handler: quantize the module's weight.

    Called by ``quantize_()`` for each module matching the filter.

    Args:
        module: The nn.Module to quantize.
        config: Your quantization config.
        parameter_name: Which parameter to quantize.

    Returns:
        Modified module with quantized weight.
    """
    if not hasattr(module, parameter_name):
        return module

    weight = getattr(module, parameter_name)

    # TODO: Replace with your quantization logic
    # Example: simple round-to-nearest
    scale = weight.abs().max() / (2 ** (config.bit_width - 1) - 1)
    quantized = torch.round(weight / scale) * scale
    setattr(module, parameter_name, nn.Parameter(quantized, requires_grad=False))

    return module


# To integrate with CompGen's pipeline, add to capture/torchao_pipeline.py:
#   if config.scheme == "my_quant":
#       from compgen.quantization.methods.my_quant import TemplateQuantConfig
#       quantize_(model, TemplateQuantConfig(**config.extra_args))
#       return model
