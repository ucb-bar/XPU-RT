"""Export-friendly module wrappers for torch.export compatibility.

Provides ``ExportableFP8Linear`` and ``ExportableFP8Conv2d``, which are plain
``nn.Module`` subclasses that store FP8 weight data + po2 scale as explicit
parameters/buffers.  Their forward methods use only standard ATen ops,
ensuring clean ``torch.export`` traces with visible FP8 quantization ops.

The rewrite flow is:

    1. ``quantize_(model, FP8E4M3Po2Config())`` -- replaces weights with
       ``FP8E4M3Po2Tensor`` subclasses (for eager dispatch).
    2. ``rewrite_for_export(model)`` -- replaces ``FP8E4M3Po2Tensor``-backed
       modules with ``ExportableFP8Linear`` / ``ExportableFP8Conv2d``
       (for ``torch.export``).

This two-step approach follows the Understanding-PI0 pattern
(``mx_exportable.py``), separating eager inference from export preparation.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from compgen.quantization.fp8_ops import (
    quantize_fp8_e4m3_po2,
)
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor

logger = structlog.get_logger()


class ExportableFP8Linear(nn.Module):
    """Export-friendly FP8 linear layer with explicit quantize/dequantize ops.

    Stores weight as a plain ``float8_e4m3fn`` tensor (not a subclass) plus a
    scalar scale factor.  Forward uses only standard ATen ops:

    1. Dequantize weight: ``w_bf16 = w_fp8.to(bf16) * scale``
    2. Quantize activation: ``x_fp8, x_scale = quantize(x)``
    3. Dequantize activation: ``x_bf16 = x_fp8.to(bf16) * x_scale``
    4. Matmul in BF16: ``y = x_bf16 @ w_bf16^T + bias``

    All ops are visible in the exported graph, enabling downstream MLIR
    lowering to see explicit FP8 cast/scale patterns.

    Args:
        weight_fp8: Quantized weight tensor (float8_e4m3fn).
        weight_scale: Per-tensor po2 scale for the weight.
        bias: Optional bias tensor (BF16).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: float,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_scale = weight_scale

        # Store FP8 weight as a buffer (not a parameter, since it's not trainable)
        self.register_buffer("weight_fp8", weight_fp8)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit FP8 quantize/dequantize ops.

        Args:
            x: Input activation tensor.

        Returns:
            Output tensor in BF16.
        """
        # Dequantize weight to BF16
        w_bf16 = self.weight_fp8.to(torch.float32) * self.weight_scale
        w_bf16 = w_bf16.to(torch.bfloat16)

        # Ensure activation is BF16 (dynamo may pass mixed dtypes)
        x_bf16 = x.to(torch.bfloat16)

        # Quantize activation to FP8 -> dequantize to BF16 (bakes in FP8 noise)
        x_fp8, x_scale = quantize_fp8_e4m3_po2(x_bf16)
        x_bf16 = (x_fp8.to(torch.float32) * x_scale).to(torch.bfloat16)

        # Ensure bias is also BF16
        bias = self.bias.to(torch.bfloat16) if self.bias is not None else None

        # Matmul in BF16
        return F.linear(x_bf16, w_bf16, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_dtype=float8_e4m3fn, scale={self.weight_scale}, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def from_quantized_linear(cls, linear: nn.Linear) -> ExportableFP8Linear:
        """Create from an nn.Linear with FP8E4M3Po2Tensor weight.

        Args:
            linear: Linear module whose weight is FP8E4M3Po2Tensor.

        Returns:
            New ExportableFP8Linear with extracted FP8 data.
        """
        weight = linear.weight
        if isinstance(weight, FP8E4M3Po2Tensor):
            fp8_data = weight._quantized_data
            scale = weight._scale
        else:
            # Not already quantized -- quantize now
            fp8_data, scale = quantize_fp8_e4m3_po2(weight)

        bias = linear.bias
        if isinstance(bias, FP8E4M3Po2Tensor):
            bias = bias.dequantize()

        return cls(
            weight_fp8=fp8_data,
            weight_scale=scale,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )


class ExportableFP8Conv2d(nn.Module):
    """Export-friendly FP8 Conv2d with explicit quantize/dequantize ops.

    Same pattern as ``ExportableFP8Linear`` but for 2D convolutions.

    Args:
        weight_fp8: Quantized weight tensor (float8_e4m3fn), shape (C_out, C_in, kH, kW).
        weight_scale: Per-tensor po2 scale.
        bias: Optional bias.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Number of groups.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: float,
        bias: torch.Tensor | None,
        stride: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] = (0, 0),
        dilation: tuple[int, ...] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.weight_scale = weight_scale
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer("weight_fp8", weight_fp8)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit FP8 ops."""
        # Dequantize weight
        w_bf16 = (self.weight_fp8.to(torch.float32) * self.weight_scale).to(torch.bfloat16)

        # Ensure activation is BF16
        x_bf16 = x.to(torch.bfloat16)

        # Quantize activation
        x_fp8, x_scale = quantize_fp8_e4m3_po2(x_bf16)
        x_bf16 = (x_fp8.to(torch.float32) * x_scale).to(torch.bfloat16)

        return F.conv2d(x_bf16, w_bf16, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_quantized_conv2d(cls, conv: nn.Conv2d) -> ExportableFP8Conv2d:
        """Create from an nn.Conv2d with FP8E4M3Po2Tensor weight."""
        weight = conv.weight
        if isinstance(weight, FP8E4M3Po2Tensor):
            fp8_data = weight._quantized_data
            scale = weight._scale
        else:
            fp8_data, scale = quantize_fp8_e4m3_po2(weight)

        bias = conv.bias
        if isinstance(bias, FP8E4M3Po2Tensor):
            bias = bias.dequantize()

        return cls(
            weight_fp8=fp8_data,
            weight_scale=scale,
            bias=bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )


def rewrite_for_export(model: nn.Module) -> nn.Module:
    """Replace FP8E4M3Po2Tensor-backed modules with export-friendly wrappers.

    Walks the module tree and replaces:
    - ``nn.Linear`` with ``FP8E4M3Po2Tensor`` weight -> ``ExportableFP8Linear``
    - ``nn.Conv2d`` with ``FP8E4M3Po2Tensor`` weight -> ``ExportableFP8Conv2d``

    This should be called after ``quantize_()`` / ``apply_smolvla_quantization()``
    but before ``torch.export.export()``.

    Args:
        model: Quantized model to rewrite.

    Returns:
        The same model with modules replaced in-place.
    """
    replacements: list[tuple[str, nn.Module, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and isinstance(getattr(module, "weight", None), FP8E4M3Po2Tensor):
            replacement = ExportableFP8Linear.from_quantized_linear(module)
            replacements.append((name, module, replacement))
        elif isinstance(module, nn.Conv2d) and isinstance(getattr(module, "weight", None), FP8E4M3Po2Tensor):
            replacement = ExportableFP8Conv2d.from_quantized_conv2d(module)
            replacements.append((name, module, replacement))

    # Apply replacements
    for name, _old, new in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new)
        logger.debug("rewrite_for_export", module=name, new_type=type(new).__name__)

    logger.info("rewrite_for_export_complete", replaced=len(replacements))
    return model


__all__ = [
    "ExportableFP8Conv2d",
    "ExportableFP8Linear",
    "rewrite_for_export",
]
