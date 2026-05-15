"""FX graph analysis for target operator coverage.

Walks captured FX graph partitions and classifies every operator against
a target's op map, producing a coverage report.  Generalizable: works with
any model and any target op map (NPU, GPU, CPU, etc.).

Usage::

    from compgen.quantization.graph_analyzer import analyze_for_npu

    graphs = [gm1, gm2]  # from capture_dynamo_partitions
    analysis = analyze_for_npu(graphs)
    print(format_analysis_report(analysis))
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

from compgen.quantization.npu_op_map import (
    _OP_TABLE,
    NpuOpCategory,
    NpuQuantDecision,
)


@dataclass
class QuantizedGraphAnalysis:
    """Result of analyzing FX graph partitions against a target op map.

    Attributes:
        total_ops: Total number of ``call_function`` ops across all partitions.
        covered_ops: Op targets that have a mapping, with their count.
        uncovered_ops: Op targets missing from the map, with their count.
        ops_by_category: Ops grouped by target execution category.
        dtype_distribution: Count of ops per input dtype requirement.
        coverage_pct: Percentage of ops (by count) that are covered.
        estimated_mxu_ops: Count of MXU_FP8 ops (matmul workload).
        estimated_vpu_ops: Count of VPU_BF16 ops (vector workload).
        estimated_xlu_ops: Count of XLU_BF16 ops (reduction workload).
        partition_count: Number of graph partitions analyzed.
    """

    total_ops: int = 0
    covered_ops: dict[str, int] = field(default_factory=dict)
    uncovered_ops: dict[str, int] = field(default_factory=dict)
    ops_by_category: dict[str, list[str]] = field(default_factory=dict)
    dtype_distribution: dict[str, int] = field(default_factory=dict)
    coverage_pct: float = 0.0
    estimated_mxu_ops: int = 0
    estimated_vpu_ops: int = 0
    estimated_xlu_ops: int = 0
    partition_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "total_ops": self.total_ops,
            "coverage_pct": round(self.coverage_pct, 2),
            "partition_count": self.partition_count,
            "estimated_mxu_ops": self.estimated_mxu_ops,
            "estimated_vpu_ops": self.estimated_vpu_ops,
            "estimated_xlu_ops": self.estimated_xlu_ops,
            "covered_ops": dict(sorted(self.covered_ops.items(), key=lambda x: -x[1])),
            "uncovered_ops": dict(sorted(self.uncovered_ops.items(), key=lambda x: -x[1])),
            "ops_by_category": {k: sorted(v) for k, v in self.ops_by_category.items()},
            "dtype_distribution": self.dtype_distribution,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Map common Python function / built-in targets to ATen op strings.
_FN_TO_ATEN: dict[str, str] = {
    # nn.functional
    "linear": "aten.linear.default",
    "relu": "aten.relu.default",
    "gelu": "aten.gelu.default",
    "silu": "aten.silu.default",
    "sigmoid": "aten.sigmoid.default",
    "tanh": "aten.tanh.default",
    "softmax": "aten._softmax.default",
    "scaled_dot_product_attention": "aten.scaled_dot_product_attention.default",
    "conv2d": "aten.conv2d",
    "batch_norm": "aten.batch_norm.default",
    "layer_norm": "aten.layer_norm.default",
    "embedding": "aten.embedding.default",
    "dropout": "aten.dropout.default",
    "matmul": "aten.mm.default",
    # Python built-in operators (from dynamo captures)
    "mul": "aten.mul.Tensor",
    "add": "aten.add.Tensor",
    "sub": "aten.sub.Tensor",
    "truediv": "aten.div.Tensor",
    "pow": "aten.pow.Tensor_Scalar",
    "iadd": "aten.add.Tensor",
    "imul": "aten.mul.Tensor",
    "isub": "aten.sub.Tensor",
    "itruediv": "aten.div.Tensor",
    "getitem": "passthrough.getitem",
    "setitem": "passthrough.setitem",
    # torch methods (from dynamo captures of torch.sin, etc.)
    "sin": "aten.sin.default",
    "cos": "aten.cos.default",
    "exp": "aten.exp.default",
    "sqrt": "aten.sqrt.default",
    "rsqrt": "aten.reciprocal.default",
    "reciprocal": "aten.reciprocal.default",
    "log2": "aten.log2.default",
    "arange": "passthrough.arange",
    "empty_like": "passthrough.empty_like",
    "full": "passthrough.full",
    "zeros": "passthrough.zeros",
    "ones": "passthrough.ones",
    "cat": "aten.cat.default",
    "stack": "aten.cat.default",
    "unsqueeze": "aten.unsqueeze.default",
    "squeeze": "aten.squeeze.dim",
    "expand": "aten.expand.default",
    "reshape": "aten.reshape.default",
    "view": "aten.view.default",
    "permute": "aten.permute.default",
    "transpose": "aten.t.default",
    "contiguous": "aten.contiguous.default",
    "clone": "aten.clone.default",
    "detach": "aten.detach.default",
    "to": "aten._to_copy.default",
    "float": "aten._to_copy.default",
    "bfloat16": "aten._to_copy.default",
    "half": "aten._to_copy.default",
    "type_as": "aten._to_copy.default",
    "sum": "aten.sum.default",
    "mean": "aten.mean.dim",
    "amax": "aten.amax.default",
    "max": "aten.amax.default",
    "clamp": "aten.clamp.default",
    "abs": "aten.abs.default",
    "neg": "aten.neg.default",
    "where": "passthrough.where",
    "masked_fill": "passthrough.masked_fill",
    "index_select": "passthrough.index_select",
    "gather": "passthrough.gather",
    "scatter": "passthrough.scatter",
    "select": "aten.select.int",
    "slice": "aten.slice.Tensor",
}


def _normalize_fn_target(fn: Any) -> str:
    """Normalize a Python function target to an ATen-style string.

    Dynamo-captured graphs may use Python function objects (e.g.,
    ``torch.nn.functional.linear``) rather than ATen OpOverload objects.
    """
    name = getattr(fn, "__name__", "")
    aten_name = _FN_TO_ATEN.get(name)
    if aten_name is not None:
        return aten_name

    # Try to match torch.ops.aten.* overloads
    target_str = str(fn)
    if "aten." in target_str:
        # Extract aten.xxx.yyy from the string
        for part in target_str.split():
            if "aten." in part:
                return part.strip("()'\"<>,")
    return target_str


def analyze_fx_graphs(
    graphs: list[torch.fx.GraphModule],
    op_map: dict[str, Any],
) -> QuantizedGraphAnalysis:
    """Analyze FX graph partitions against a target op map.

    Walks every ``call_function`` node in every graph partition and checks
    whether its target string exists in the provided op map.

    Args:
        graphs: List of captured ``torch.fx.GraphModule`` partitions.
        op_map: Dict mapping ATen op target strings to any value (e.g.,
            ``NpuQuantDecision`` for the NPU, or just ``True`` for coverage
            checking).

    Returns:
        ``QuantizedGraphAnalysis`` with detailed coverage breakdown.
    """
    covered: dict[str, int] = defaultdict(int)
    uncovered: dict[str, int] = defaultdict(int)
    by_category: dict[str, list[str]] = defaultdict(list)
    dtype_dist: dict[str, int] = defaultdict(int)
    mxu = vpu = xlu = 0
    total = 0

    for graph in graphs:
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue

            total += 1
            # Normalize target to ATen string form.
            # torch.compile uses Python function objects; torch.export uses
            # OpOverload objects.  Both need to map to "aten.<name>.default".
            raw_target = node.target
            if hasattr(raw_target, "name"):
                # OpOverload: e.g., aten.linear.default
                target = f"aten.{raw_target.name()}" if callable(getattr(raw_target, "name", None)) else str(raw_target)
            elif hasattr(raw_target, "__module__") and hasattr(raw_target, "__name__"):
                # Python function: e.g., torch.nn.functional.linear -> aten.linear.default
                target = _normalize_fn_target(raw_target)
            else:
                target = str(raw_target)

            if target in op_map:
                covered[target] = covered.get(target, 0) + 1
                decision = op_map[target]

                # Extract category and dtype info if NpuQuantDecision
                if isinstance(decision, NpuQuantDecision):
                    cat_name = decision.category.value
                    if target not in by_category.get(cat_name, []):
                        by_category[cat_name].append(target)
                    dtype_dist[decision.input_dtype] = dtype_dist.get(decision.input_dtype, 0) + 1

                    if decision.category == NpuOpCategory.MXU_FP8:
                        mxu += 1
                    elif decision.category == NpuOpCategory.VPU_BF16:
                        vpu += 1
                    elif decision.category == NpuOpCategory.XLU_BF16:
                        xlu += 1
            else:
                uncovered[target] = uncovered.get(target, 0) + 1

    covered_count = sum(covered.values())
    coverage_pct = (covered_count / total * 100) if total > 0 else 100.0

    return QuantizedGraphAnalysis(
        total_ops=total,
        covered_ops=dict(covered),
        uncovered_ops=dict(uncovered),
        ops_by_category=dict(by_category),
        dtype_distribution=dict(dtype_dist),
        coverage_pct=coverage_pct,
        estimated_mxu_ops=mxu,
        estimated_vpu_ops=vpu,
        estimated_xlu_ops=xlu,
        partition_count=len(graphs),
    )


def analyze_for_npu(
    graphs: list[torch.fx.GraphModule],
) -> QuantizedGraphAnalysis:
    """Analyze FX graph partitions for NPU target op coverage.

    Convenience wrapper using the NPU op map from ``npu_op_map``.

    Args:
        graphs: List of captured graph partitions.

    Returns:
        ``QuantizedGraphAnalysis`` against the NPU op table.
    """
    return analyze_fx_graphs(graphs, _OP_TABLE)


def format_analysis_report(analysis: QuantizedGraphAnalysis) -> str:
    """Format a human-readable analysis report.

    Args:
        analysis: The graph analysis result.

    Returns:
        Multi-line report string.
    """
    lines = [
        "=" * 70,
        "  FX Graph Analysis — Target Op Coverage Report",
        "=" * 70,
        "",
        f"  Graph partitions analyzed: {analysis.partition_count}",
        f"  Total call_function ops:   {analysis.total_ops}",
        f"  Covered ops:               {sum(analysis.covered_ops.values())} ({analysis.coverage_pct:.1f}%)",
        f"  Uncovered ops:             {sum(analysis.uncovered_ops.values())}",
        "",
        "  Execution Unit Breakdown:",
        f"    MXU (FP8 matmul):   {analysis.estimated_mxu_ops} ops",
        f"    VPU (BF16 vector):  {analysis.estimated_vpu_ops} ops",
        f"    XLU (BF16 reduce):  {analysis.estimated_xlu_ops} ops",
        "",
    ]

    if analysis.covered_ops:
        lines.append("  Covered Ops (top 20):")
        for target, count in sorted(analysis.covered_ops.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"    {target}: {count}x")
        lines.append("")

    if analysis.uncovered_ops:
        lines.append("  *** Uncovered Ops (need mapping):")
        for target, count in sorted(analysis.uncovered_ops.items(), key=lambda x: -x[1]):
            lines.append(f"    {target}: {count}x")
        lines.append("")

    if analysis.ops_by_category:
        lines.append("  Ops by Category:")
        for cat, ops in sorted(analysis.ops_by_category.items()):
            lines.append(f"    {cat}: {', '.join(ops[:10])}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


__all__ = [
    "QuantizedGraphAnalysis",
    "analyze_for_npu",
    "analyze_fx_graphs",
    "format_analysis_report",
]
