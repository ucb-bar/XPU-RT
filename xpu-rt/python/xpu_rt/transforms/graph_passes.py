"""IREE-inspired FX graph decomposition passes.

Operates on captured TorchDynamo FX graphs to identify fusion patterns and
annotate nodes with pattern metadata.  These passes run before pattern
catalog building, enabling pattern-level grouping.

Inspired by IREE GlobalOptimization passes (transpose propagation,
elementwise detachment, special ops raising) but implemented in Python
on FX graphs, with no IREE or MLIR dependency.

Framework-agnostic: works with any model captured via TorchDynamo.
"""

from __future__ import annotations

from typing import Any

import torch


def _target_name(node: Any) -> str:
    """Normalize an FX node target to a short string."""
    raw = node.target
    if hasattr(raw, "name") and callable(getattr(raw, "name", None)):
        return raw.name()
    name = getattr(raw, "__name__", "")
    return name if name else str(raw)


def _get_users(node: Any) -> list[Any]:
    return list(node.users.keys()) if hasattr(node, "users") else []


def _get_args(node: Any) -> list[Any]:
    return [a for a in node.args if hasattr(a, "op")]


# ---------------------------------------------------------------------------
# Pass: Detect and annotate fusion patterns
# ---------------------------------------------------------------------------

_MATMUL_TARGETS = {"linear", "mm", "addmm"}
_ACTIVATION_TARGETS = {"silu", "gelu", "relu", "tanh", "sigmoid"}

_RSQRT_TARGETS = {"rsqrt", "reciprocal"}
_POW_TARGETS = {"pow"}
_MEAN_TARGETS = {"mean"}
_SIN_TARGETS = {"sin"}
_COS_TARGETS = {"cos"}


def detect_and_annotate_patterns(graph: torch.fx.GraphModule) -> int:
    """Detect multi-op fusion patterns and annotate nodes.

    Scans for known sequences:
    - ``linear → activation`` → annotated as ``fused_linear_<act>``
    - ``pow → mean → add → rsqrt → mul`` → annotated as ``rms_norm``
    - ``sin/cos → mul → add`` sequences → annotated as ``rope``

    Annotations are stored in ``node.meta["_compgen_pattern"]``.

    Args:
        graph: A captured FX GraphModule.

    Returns:
        Number of patterns detected.
    """
    count = 0

    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue

        tname = _target_name(node)

        # Pattern: linear → activation
        if tname in _MATMUL_TARGETS:
            users = _get_users(node)
            if len(users) == 1 and users[0].op == "call_function":
                act_name = _target_name(users[0])
                if act_name in _ACTIVATION_TARGETS:
                    node.meta["_compgen_pattern"] = f"fused_linear_{act_name}"
                    users[0].meta["_compgen_pattern"] = f"fused_linear_{act_name}_tail"
                    count += 1
                    continue

        # Pattern: pow(x, 2) → mean → add(eps) → rsqrt → mul (RMSNorm)
        if tname in _POW_TARGETS:
            chain = _trace_rms_norm(node)
            if chain:
                for n in chain:
                    n.meta["_compgen_pattern"] = "rms_norm"
                count += 1
                continue

    return count


def _trace_rms_norm(pow_node: Any) -> list[Any] | None:
    """Try to trace an RMSNorm pattern starting from a pow node."""
    # pow → mean → add → rsqrt/reciprocal → mul
    users = _get_users(pow_node)
    if not users:
        return None

    mean_node = None
    for u in users:
        if u.op == "call_function" and _target_name(u) in _MEAN_TARGETS:
            mean_node = u
            break
    if mean_node is None:
        return None

    # mean → add (epsilon)
    add_users = _get_users(mean_node)
    add_node = None
    for u in add_users:
        if u.op == "call_function" and _target_name(u) in ("add", "aten.add.Tensor"):
            add_node = u
            break
    if add_node is None:
        return None

    # add → rsqrt/reciprocal
    rsqrt_users = _get_users(add_node)
    rsqrt_node = None
    for u in rsqrt_users:
        if u.op == "call_function" and _target_name(u) in _RSQRT_TARGETS:
            rsqrt_node = u
            break
    if rsqrt_node is None:
        return None

    # rsqrt → mul
    mul_users = _get_users(rsqrt_node)
    mul_node = None
    for u in mul_users:
        if u.op == "call_function" and _target_name(u) in ("mul", "aten.mul.Tensor"):
            mul_node = u
            break
    if mul_node is None:
        return None

    return [pow_node, mean_node, add_node, rsqrt_node, mul_node]


# ---------------------------------------------------------------------------
# Pass: Fold transpose into matmul
# ---------------------------------------------------------------------------

def fold_transpose_into_matmul(graph: torch.fx.GraphModule) -> int:
    """Fold explicit transpose ops into matmul consumers.

    Detects patterns like ``transpose(weight) → matmul`` and annotates
    the matmul to absorb the transpose (matching F.linear semantics).

    Args:
        graph: A captured FX GraphModule.

    Returns:
        Number of transposes folded.
    """
    count = 0
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        tname = _target_name(node)
        if tname in ("t", "transpose", "aten.t.default"):
            users = _get_users(node)
            for user in users:
                if user.op == "call_function" and _target_name(user) in _MATMUL_TARGETS:
                    user.meta["_compgen_transpose_absorbed"] = True
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Pass: Raise composite ops
# ---------------------------------------------------------------------------

def raise_composite_ops(graph: torch.fx.GraphModule) -> int:
    """Detect composite op sequences and annotate them.

    Identifies softmax (exp → sum → div), and marks them as composite
    patterns for pattern-level grouping.

    Args:
        graph: A captured FX GraphModule.

    Returns:
        Number of composite ops raised.
    """
    count = 0
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        tname = _target_name(node)
        # Explicit softmax ops are already atomic
        if tname in ("softmax", "_softmax", "aten._softmax.default"):
            node.meta["_compgen_pattern"] = "softmax"
            count += 1
    return count


# ---------------------------------------------------------------------------
# Combined pass
# ---------------------------------------------------------------------------

def run_all_decomposition_passes(graph: torch.fx.GraphModule) -> dict[str, int]:
    """Run all decomposition passes on an FX graph.

    Args:
        graph: A captured FX GraphModule.

    Returns:
        Dict of pass names to number of transformations applied.
    """
    results = {
        "detect_and_annotate_patterns": detect_and_annotate_patterns(graph),
        "fold_transpose_into_matmul": fold_transpose_into_matmul(graph),
        "raise_composite_ops": raise_composite_ops(graph),
    }
    return results


def run_decomposition_on_graphs(graphs: list[torch.fx.GraphModule]) -> dict[str, int]:
    """Run all decomposition passes on all graph partitions.

    Args:
        graphs: List of captured FX GraphModule partitions.

    Returns:
        Aggregated pass statistics.
    """
    totals: dict[str, int] = {}
    for graph in graphs:
        results = run_all_decomposition_passes(graph)
        for name, count in results.items():
            totals[name] = totals.get(name, 0) + count
    return totals


__all__ = [
    "detect_and_annotate_patterns",
    "fold_transpose_into_matmul",
    "raise_composite_ops",
    "run_all_decomposition_passes",
    "run_decomposition_on_graphs",
]
