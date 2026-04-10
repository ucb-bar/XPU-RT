"""FX graph pattern detection for kernel fusion and grouping.

Walks captured FX graph partitions and identifies multi-op sequences that
form known compute patterns (matmul, attention, RMSNorm, RoPE, etc.).
Annotates each node with its detected pattern so the catalog builder can
group them into reusable kernel patterns.

Framework-agnostic: works with any model captured via TorchDynamo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DetectedPattern:
    """A detected compute pattern instance in an FX graph.

    Attributes:
        pattern_type: Pattern identifier (``"matmul"``, ``"rms_norm"``, etc.).
        nodes: FX node names belonging to this pattern instance.
        primary_node: The main compute node (e.g., the matmul node).
        input_shapes: Concrete input shapes extracted from metadata.
        output_shape: Concrete output shape.
        input_dtypes: Input dtype strings.
        output_dtype: Output dtype string.
        params: Extracted parameters (M, K, N for matmul, etc.).
        flops: Estimated FLOPs for this pattern instance.
        graph_idx: Which graph partition this was found in.
    """

    pattern_type: str = ""
    nodes: list[str] = field(default_factory=list)
    primary_node: str = ""
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shape: tuple[int, ...] = ()
    input_dtypes: list[str] = field(default_factory=list)
    output_dtype: str = "bf16"
    params: dict[str, Any] = field(default_factory=dict)
    flops: int = 0
    graph_idx: int = 0


# ---------------------------------------------------------------------------
# Target normalization (handles both dynamo built-in fns and ATen overloads)
# ---------------------------------------------------------------------------

def _target_name(node: Any) -> str:
    """Normalize an FX node target to a short name for pattern matching."""
    raw = node.target
    # ATen OpOverload
    if hasattr(raw, "name") and callable(getattr(raw, "name", None)):
        return raw.name()
    # Python function
    name = getattr(raw, "__name__", "")
    if name:
        return name
    s = str(raw)
    # Extract aten.xxx from string
    if "aten." in s:
        for part in s.split():
            if "aten." in part:
                return part.strip("()'\"<>,")
    return s


def _get_meta_val(node: Any) -> Any:
    """Get the tensor metadata from an FX node."""
    if not hasattr(node, "meta"):
        return None
    return node.meta.get("val") or node.meta.get("example_value")


def _shape(node: Any) -> tuple[int, ...] | None:
    val = _get_meta_val(node)
    if val is not None and hasattr(val, "shape"):
        return tuple(int(d) for d in val.shape)
    return None


def _dtype_str(node: Any) -> str:
    val = _get_meta_val(node)
    if val is not None and hasattr(val, "dtype"):
        d = val.dtype
        if d == torch.bfloat16:
            return "bf16"
        if d == torch.float32:
            return "f32"
        if d == torch.float16:
            return "f16"
        if hasattr(torch, "float8_e4m3fn") and d == torch.float8_e4m3fn:
            return "fp8_e4m3"
    return "bf16"


# ---------------------------------------------------------------------------
# Single-op pattern detectors
# ---------------------------------------------------------------------------

_MATMUL_NAMES = {"linear", "mm", "addmm", "aten.linear.default", "aten.mm.default", "aten.addmm.default"}
_BMM_NAMES = {"bmm", "aten.bmm.default", "matmul", "aten.matmul.default"}
_SOFTMAX_NAMES = {"softmax", "_softmax", "aten._softmax.default", "aten.softmax.int"}
_SILU_NAMES = {"silu", "aten.silu.default"}
_GELU_NAMES = {"gelu", "aten.gelu.default"}
_RELU_NAMES = {"relu", "aten.relu.default"}
_CONV_NAMES = {"conv2d", "aten.conv2d", "aten.convolution.default"}

_BINARY_NAMES = {"add", "sub", "mul", "truediv", "iadd", "imul", "isub", "itruediv",
                 "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"}
_UNARY_NAMES = {"exp", "exp2", "log2", "sin", "cos", "tanh", "sqrt", "reciprocal", "rsqrt",
                "abs", "neg", "clamp", "pow",
                "aten.exp.default", "aten.sin.default", "aten.cos.default", "aten.tanh.default",
                "aten.sqrt.default", "aten.reciprocal.default", "aten.pow.Tensor_Scalar",
                "aten.clamp.default", "aten.abs.default", "aten.neg.default",
                "aten.log2.default", "aten.exp2.default"}
_REDUCTION_NAMES = {"sum", "mean", "amax", "aten.sum.default", "aten.sum.dim_IntList",
                    "aten.mean.dim", "aten.amax.default"}


def _estimate_matmul_flops(shapes: list[tuple[int, ...]]) -> int:
    if len(shapes) < 2:
        return 0
    a, b = shapes[0], shapes[1]
    if len(a) >= 2 and len(b) >= 2:
        m, k = a[-2], a[-1]
        n = b[-1]
        batch = 1
        for d in a[:-2]:
            batch *= d
        return 2 * batch * m * k * n
    return 0


def _matmul_params(shapes: list[tuple[int, ...]]) -> dict[str, int]:
    if len(shapes) < 2:
        return {}
    a, b = shapes[0], shapes[1]
    params: dict[str, int] = {}
    if len(a) >= 2:
        params["M"] = a[-2]
        params["K"] = a[-1]
    if len(b) >= 2:
        params["N"] = b[-1]
    if len(a) > 2:
        params["batch"] = 1
        for d in a[:-2]:
            params["batch"] *= d
    return params


# ---------------------------------------------------------------------------
# Multi-op pattern detectors (look-ahead from a node)
# ---------------------------------------------------------------------------

def _get_users(node: Any) -> list[Any]:
    """Get consumer nodes of an FX node."""
    return list(node.users.keys()) if hasattr(node, "users") else []


def _check_activation_fusion(node: Any) -> str | None:
    """Check if a matmul/linear node feeds directly into an activation."""
    users = _get_users(node)
    if len(users) != 1:
        return None
    user = users[0]
    if user.op != "call_function":
        return None
    tname = _target_name(user)
    if tname in _SILU_NAMES:
        return "silu"
    if tname in _GELU_NAMES:
        return "gelu"
    if tname in _RELU_NAMES:
        return "relu"
    return None


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_patterns_in_graphs(
    graphs: list[torch.fx.GraphModule],
) -> list[DetectedPattern]:
    """Detect compute patterns across all captured FX graph partitions.

    Walks every ``call_function`` node, identifies its pattern type, extracts
    concrete shapes and parameters from node metadata.

    Args:
        graphs: List of captured ``torch.fx.GraphModule`` partitions.

    Returns:
        List of ``DetectedPattern`` instances, one per detected pattern.
    """
    patterns: list[DetectedPattern] = []
    seen_nodes: set[str] = set()

    for gi, graph in enumerate(graphs):
        nodes_by_name: dict[str, Any] = {}
        for node in graph.graph.nodes:
            nodes_by_name[node.name] = node

        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue
            if node.name in seen_nodes:
                continue

            tname = _target_name(node)
            out_shape = _shape(node) or ()
            out_dtype = _dtype_str(node)

            # Collect input shapes from args
            in_shapes: list[tuple[int, ...]] = []
            in_dtypes: list[str] = []
            for arg in node.args:
                if hasattr(arg, "meta"):
                    s = _shape(arg)
                    if s is not None:
                        in_shapes.append(s)
                        in_dtypes.append(_dtype_str(arg))

            # --- Matmul patterns ---
            if tname in _MATMUL_NAMES:
                seen_nodes.add(node.name)
                activation = _check_activation_fusion(node)
                if activation:
                    user = _get_users(node)[0]
                    seen_nodes.add(user.name)
                    patterns.append(DetectedPattern(
                        pattern_type=f"fused_linear_{activation}",
                        nodes=[node.name, user.name],
                        primary_node=node.name,
                        input_shapes=in_shapes,
                        output_shape=_shape(user) or out_shape,
                        input_dtypes=in_dtypes,
                        output_dtype=out_dtype,
                        params=_matmul_params(in_shapes),
                        flops=_estimate_matmul_flops(in_shapes),
                        graph_idx=gi,
                    ))
                else:
                    patterns.append(DetectedPattern(
                        pattern_type="matmul",
                        nodes=[node.name],
                        primary_node=node.name,
                        input_shapes=in_shapes,
                        output_shape=out_shape,
                        input_dtypes=in_dtypes,
                        output_dtype=out_dtype,
                        params=_matmul_params(in_shapes),
                        flops=_estimate_matmul_flops(in_shapes),
                        graph_idx=gi,
                    ))
                continue

            # --- Batch matmul ---
            if tname in _BMM_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="batch_matmul",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    params=_matmul_params(in_shapes),
                    flops=_estimate_matmul_flops(in_shapes),
                    graph_idx=gi,
                ))
                continue

            # --- Conv2d ---
            if tname in _CONV_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="conv2d",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    params={},
                    flops=0,
                    graph_idx=gi,
                ))
                continue

            # --- Softmax ---
            if tname in _SOFTMAX_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="softmax",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    params={},
                    flops=in_shapes[0][-1] * 5 if in_shapes else 0,  # exp+sum+div per element
                    graph_idx=gi,
                ))
                continue

            # --- Standalone activations ---
            if tname in _SILU_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="silu",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    graph_idx=gi,
                ))
                continue

            if tname in _GELU_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="gelu",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    graph_idx=gi,
                ))
                continue

            # --- Elementwise binary ---
            if tname in _BINARY_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="elementwise_binary",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    graph_idx=gi,
                ))
                continue

            # --- Elementwise unary ---
            if tname in _UNARY_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="elementwise_unary",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    graph_idx=gi,
                ))
                continue

            # --- Reductions ---
            if tname in _REDUCTION_NAMES:
                seen_nodes.add(node.name)
                patterns.append(DetectedPattern(
                    pattern_type="reduction",
                    nodes=[node.name],
                    primary_node=node.name,
                    input_shapes=in_shapes,
                    output_shape=out_shape,
                    input_dtypes=in_dtypes,
                    output_dtype=out_dtype,
                    graph_idx=gi,
                ))
                continue

    return patterns


__all__ = ["DetectedPattern", "detect_patterns_in_graphs"]
