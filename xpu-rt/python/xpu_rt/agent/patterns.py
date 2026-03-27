"""Computation pattern library and matcher for FX graphs.

Detects known computation patterns (MLP chains, attention, normalization)
in torch dynamo FX graphs. Works directly on FX graph nodes — not xDSL IR —
because FX has the clearest computation structure with explicit data flow
via node.users.

Each pattern has a signature (op sequence) and a kernel opportunity
(what optimized kernel could replace it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatternSignature:
    """Definition of a known computation pattern.

    Attributes:
        name: Pattern name (e.g., "linear_chain", "gqa_attention").
        op_targets: ATen op targets that form this pattern, in data-flow order.
        kernel_opportunity: What kernel could replace this pattern.
        description: Human-readable description.
    """

    name: str
    op_targets: tuple[str, ...]
    kernel_opportunity: str
    description: str = ""


# Pattern library — known computation patterns in modern neural networks
PATTERN_LIBRARY: dict[str, PatternSignature] = {
    "linear": PatternSignature(
        name="linear",
        op_targets=("aten.linear.default",),
        kernel_opportunity="optimized_linear",
        description="Single linear layer (matmul + bias)",
    ),
    "linear_chain": PatternSignature(
        name="linear_chain",
        op_targets=("aten.linear.default", "aten.gelu.default", "aten.linear.default"),
        kernel_opportunity="fused_mlp",
        description="MLP: linear → activation → linear",
    ),
    "linear_relu_chain": PatternSignature(
        name="linear_relu_chain",
        op_targets=("aten.linear.default", "aten.relu.default", "aten.linear.default"),
        kernel_opportunity="fused_mlp_relu",
        description="MLP: linear → ReLU → linear",
    ),
    "gqa_attention": PatternSignature(
        name="gqa_attention",
        op_targets=(
            "aten.linear.default",  # Q projection
            "aten.linear.default",  # K projection
            "aten.linear.default",  # V projection
            "aten.matmul.default",  # QK^T
            "aten._softmax.default",  # softmax
            "aten.matmul.default",  # AV
        ),
        kernel_opportunity="fused_gqa_attention",
        description="Grouped query attention: Q/K/V projections + QK^T softmax AV",
    ),
    "gate_up_down_mlp": PatternSignature(
        name="gate_up_down_mlp",
        op_targets=(
            "aten.linear.default",  # gate
            "aten.linear.default",  # up
            "aten.mul.Tensor",      # gate * up
            "aten.linear.default",  # down
        ),
        kernel_opportunity="fused_gate_mlp",
        description="Gemma-style MLP: gate + up → multiply → down",
    ),
    "rmsnorm": PatternSignature(
        name="rmsnorm",
        op_targets=("aten.pow.Tensor_Scalar", "aten.mean.dim", "aten.add.Tensor",
                     "aten.rsqrt.default", "aten.mul.Tensor"),
        kernel_opportunity="fused_rmsnorm",
        description="RMSNorm: pow → mean → add(eps) → rsqrt → mul",
    ),
    "rope": PatternSignature(
        name="rope",
        op_targets=("aten.mul.Tensor", "aten.mul.Tensor", "aten.add.Tensor"),
        kernel_opportunity="fused_rope",
        description="Rotary positional embedding: x*cos + rotate(x)*sin",
    ),
}


@dataclass(frozen=True)
class FXNodeInfo:
    """Extracted info from one FX graph node."""

    name: str
    target: str                      # e.g., "aten.linear.default"
    input_names: tuple[str, ...]     # names of input nodes
    user_names: tuple[str, ...]      # names of consumer nodes
    shape: tuple[int, ...] | None    # output shape
    dtype: str
    flops: int
    bytes_total: int


@dataclass(frozen=True)
class MatchedPattern:
    """A pattern match found in the FX graph.

    Attributes:
        pattern_name: Which pattern from the library.
        cluster_id: Unique cluster identifier.
        node_names: FX node names in this cluster (data-flow order).
        kernel_opportunity: What kernel could replace this.
    """

    pattern_name: str
    cluster_id: str
    node_names: tuple[str, ...]
    kernel_opportunity: str


def extract_fx_nodes(exported_program: Any) -> list[FXNodeInfo]:
    """Extract structured info from FX graph nodes."""
    nodes: list[FXNodeInfo] = []
    for node in exported_program.graph.nodes:
        if node.op != "call_function":
            continue

        target = str(node.target)
        input_names = tuple(a.name for a in node.args if hasattr(a, "name"))
        user_names = tuple(u.name for u in node.users)

        val = node.meta.get("val")
        shape = tuple(val.shape) if hasattr(val, "shape") else None
        dtype = str(val.dtype).replace("torch.", "") if hasattr(val, "dtype") else "f32"

        # Estimate FLOPs
        flops = 0
        if "linear" in target and shape and len(input_names) >= 2:
            # linear(x, w, b): FLOPs ≈ 2 * M * K * N
            in_shape = None
            for a in node.args:
                if hasattr(a, "meta"):
                    v = a.meta.get("val")
                    if hasattr(v, "shape") and len(v.shape) == 2:
                        in_shape = tuple(v.shape)
                        break
            if in_shape and shape:
                flops = 2 * in_shape[0] * in_shape[1] * shape[-1]

        # Estimate bytes
        bytes_total = 0
        if shape:
            elem = 1
            for s in shape:
                elem *= s
            bytes_total = elem * 4  # assume f32

        nodes.append(FXNodeInfo(
            name=node.name,
            target=target,
            input_names=input_names,
            user_names=user_names,
            shape=shape,
            dtype=dtype,
            flops=flops,
            bytes_total=bytes_total,
        ))

    return nodes


def match_patterns(fx_nodes: list[FXNodeInfo]) -> list[MatchedPattern]:
    """Match FX node sequences against the pattern library.

    Uses greedy longest-match: tries longer patterns first, marks matched
    nodes as consumed so they don't double-match.
    """
    # Build lookup
    node_by_name: dict[str, FXNodeInfo] = {n.name: n for n in fx_nodes}
    consumed: set[str] = set()
    matches: list[MatchedPattern] = []
    cluster_counters: dict[str, int] = {}

    # Sort patterns by length (longest first for greedy matching)
    sorted_patterns = sorted(PATTERN_LIBRARY.values(), key=lambda p: len(p.op_targets), reverse=True)

    for pattern in sorted_patterns:
        targets = pattern.op_targets

        # Try to match starting at each unconsumed node
        for node in fx_nodes:
            if node.name in consumed:
                continue
            if node.target != targets[0]:
                continue

            # Try to extend the match along the data flow
            matched_names = _try_match_chain(node, targets, node_by_name, consumed)
            if matched_names is not None:
                count = cluster_counters.get(pattern.name, 0)
                cluster_id = f"{pattern.name}_{count}"
                cluster_counters[pattern.name] = count + 1

                matches.append(MatchedPattern(
                    pattern_name=pattern.name,
                    cluster_id=cluster_id,
                    node_names=tuple(matched_names),
                    kernel_opportunity=pattern.kernel_opportunity,
                ))
                consumed.update(matched_names)

    return matches


def _try_match_chain(
    start: FXNodeInfo,
    targets: tuple[str, ...],
    node_by_name: dict[str, FXNodeInfo],
    consumed: set[str],
) -> list[str] | None:
    """Try to match a chain of targets starting from a node, following data flow."""
    if len(targets) == 1:
        return [start.name]

    chain: list[str] = [start.name]
    current = start

    for target in targets[1:]:
        # Find a user of current that matches the next target and isn't consumed
        found = False
        for user_name in current.user_names:
            if user_name in consumed:
                continue
            user = node_by_name.get(user_name)
            if user is None:
                continue
            if user.target == target:
                chain.append(user_name)
                current = user
                found = True
                break
        if not found:
            return None

    return chain


__all__ = [
    "FXNodeInfo",
    "MatchedPattern",
    "PATTERN_LIBRARY",
    "PatternSignature",
    "extract_fx_nodes",
    "match_patterns",
]
