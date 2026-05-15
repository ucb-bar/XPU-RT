"""Golden test data generation for kernel patterns.

Generates inputs and expected outputs (golden data) for each kernel pattern,
providing the correctness oracle that autocomp validates generated kernels
against.  Supports both small (fast iteration) and real (production shape)
variants.

Framework-agnostic output: tensors saved as both .pt and .npy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from compgen.kernels.patterns.catalog import KernelPattern


@dataclass
class GoldenTestCase:
    """Golden inputs + expected outputs for one pattern variant.

    Attributes:
        pattern_id: Which pattern this test case is for.
        variant_name: Shape variant identifier (``"small"``, ``"real"``, etc.).
        params: Shape parameters (e.g., ``{"M": 64, "K": 768, "N": 3072}``).
        inputs: Named input tensors.
        expected_output: Golden output tensor.
        metadata: Additional info (dtypes, shapes, FLOPs).
    """

    pattern_id: str = ""
    variant_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    expected_output: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-pattern golden data generators
# ---------------------------------------------------------------------------


def _gen_matmul(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    M = params.get("M", 64)
    K = params.get("K", 768)
    N = params.get("N", 3072)
    batch = params.get("batch", 1)
    torch.manual_seed(42)
    if batch > 1:
        activation = torch.randn(batch, M, K, dtype=torch.bfloat16)
    else:
        activation = torch.randn(1, M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    output = F.linear(activation, weight)
    return {"activation": activation, "weight": weight}, output


def _gen_batch_matmul(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = params.get("batch", 1)
    M = params.get("M", 15)
    seq_q = params.get("K", 291)  # K used as seq_q for bmm
    seq_k = params.get("N", 64)  # N used as seq_k
    torch.manual_seed(42)
    A = torch.randn(batch, M, seq_q, seq_k, dtype=torch.bfloat16)
    B = torch.randn(batch, M, seq_k, seq_q, dtype=torch.bfloat16)
    output = torch.matmul(A, B)
    return {"A": A, "B": B}, output


def _gen_fused_linear_silu(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    M = params.get("M", 64)
    K = params.get("K", 768)
    N = params.get("N", 3072)
    torch.manual_seed(42)
    activation = torch.randn(1, M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    output = F.silu(F.linear(activation, weight))
    return {"activation": activation, "weight": weight}, output


def _gen_fused_linear_gelu(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    M = params.get("M", 64)
    K = params.get("K", 768)
    N = params.get("N", 3072)
    torch.manual_seed(42)
    activation = torch.randn(1, M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    output = F.gelu(F.linear(activation, weight), approximate="tanh")
    return {"activation": activation, "weight": weight}, output


def _gen_softmax(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    shape = params.get("input_shapes", [(1, 15, 291, 291)])
    if isinstance(shape, list) and shape:
        shape = shape[0]
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    output = torch.softmax(x, dim=-1)
    return {"x": x}, output


def _gen_elementwise(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    shape = params.get("input_shapes", [(1, 241, 960)])
    if isinstance(shape, list) and shape:
        shape = shape[0]
    torch.manual_seed(42)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    output = a + b
    return {"a": a, "b": b}, output


def _gen_elementwise_unary(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    shape = params.get("input_shapes", [(1, 241, 960)])
    if isinstance(shape, list) and shape:
        shape = shape[0]
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    output = torch.exp(x)
    return {"x": x}, output


def _gen_reduction(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    shape = params.get("input_shapes", [(1, 241, 960)])
    if isinstance(shape, list) and shape:
        shape = shape[0]
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    output = x.sum(dim=-1)
    return {"x": x}, output


def _gen_conv2d(params: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    shapes = params.get("input_shapes", [(1, 3, 512, 512), (768, 3, 16, 16)])
    x_shape = shapes[0] if len(shapes) > 0 else (1, 3, 512, 512)
    w_shape = shapes[1] if len(shapes) > 1 else (768, 3, 16, 16)
    torch.manual_seed(42)
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    w = torch.randn(w_shape, dtype=torch.bfloat16)
    stride = w_shape[-1]  # Patch embedding: stride = kernel size
    output = F.conv2d(x, w, stride=stride)
    return {"x": x, "weight": w}, output


_GENERATORS: dict[str, Any] = {
    "matmul": _gen_matmul,
    "batch_matmul": _gen_batch_matmul,
    "fused_linear_silu": _gen_fused_linear_silu,
    "fused_linear_gelu": _gen_fused_linear_gelu,
    "softmax": _gen_softmax,
    "silu": _gen_elementwise_unary,
    "gelu": _gen_elementwise_unary,
    "elementwise_binary": _gen_elementwise,
    "elementwise_unary": _gen_elementwise_unary,
    "reduction": _gen_reduction,
    "conv2d": _gen_conv2d,
}


def generate_golden_for_pattern(
    pattern: KernelPattern,
    variant: str = "small",
) -> GoldenTestCase:
    """Generate golden test data for a kernel pattern.

    Args:
        pattern: The kernel pattern to generate data for.
        variant: ``"small"`` for fast testing shapes, ``"real"`` for
            production shapes from the first priority variant.

    Returns:
        ``GoldenTestCase`` with inputs and expected output.
    """
    gen = _GENERATORS.get(pattern.pattern_id)
    if gen is None:
        # Fallback: use first shape variant params
        return GoldenTestCase(
            pattern_id=pattern.pattern_id,
            variant_name=variant,
            metadata={"error": f"No generator for pattern {pattern.pattern_id}"},
        )

    # Pick params based on variant
    if variant == "small":
        # Use small shapes for fast iteration
        params = _small_params(pattern)
    elif variant == "real" and pattern.priority_shapes:
        params = pattern.priority_shapes[0]
    elif pattern.shape_variants:
        params = pattern.shape_variants[0]
    else:
        params = {}

    inputs, expected = gen(params)

    return GoldenTestCase(
        pattern_id=pattern.pattern_id,
        variant_name=variant,
        params=params,
        inputs=inputs,
        expected_output=expected,
        metadata={
            "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "output_shape": list(expected.shape),
            "input_dtypes": {k: str(v.dtype) for k, v in inputs.items()},
            "output_dtype": str(expected.dtype),
        },
    )


def _small_params(pattern: KernelPattern) -> dict[str, Any]:
    """Generate small-scale parameters for fast testing."""
    if pattern.pattern_id in ("matmul", "fused_linear_silu", "fused_linear_gelu", "fused_linear_relu"):
        return {"M": 16, "K": 64, "N": 128}
    if pattern.pattern_id == "batch_matmul":
        return {"batch": 1, "M": 4, "K": 16, "N": 16}
    if pattern.pattern_id == "softmax":
        return {"input_shapes": [(1, 4, 16, 16)]}
    if pattern.pattern_id in ("elementwise_binary", "elementwise_unary", "silu", "gelu", "reduction"):
        return {"input_shapes": [(1, 16, 64)]}
    if pattern.pattern_id == "conv2d":
        return {"input_shapes": [(1, 3, 32, 32), (16, 3, 4, 4)]}
    return {}


__all__ = ["GoldenTestCase", "generate_golden_for_pattern"]
