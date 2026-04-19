"""``insert_reduce_scatter`` -- materialize
``compgen.collective.reduce_scatter`` where an op's sharding carries
a partial sum AND the consumer wants the result sharded.

This is the fused variant of AllReduce+Scatter that saves bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.collective import ReduceKindAttr, ReduceScatterOp, ShardingSpecAttr


@dataclass
class InsertReduceScatterStats:
    ops_seen: int = 0
    reduce_scatters_inserted: int = 0


def _groups(devices: list[int]) -> ArrayAttr:
    n = 1
    for d in devices:
        n *= d
    return ArrayAttr(
        [ArrayAttr([IntegerAttr(i, IntegerType(64)) for i in range(n)])]
    )


class _InsertReduceScatterPattern(RewritePattern):
    def __init__(self, stats: InsertReduceScatterStats) -> None:
        self.stats = stats

    def match_and_rewrite(
        self, op: Operation, rewriter: PatternRewriter
    ) -> None:
        sharding = op.attributes.get("compgen.sharding")
        scatter_axis_attr = op.attributes.get("compgen.scatter_axis")
        if not isinstance(sharding, ShardingSpecAttr):
            return
        if scatter_axis_attr is None:
            return
        self.stats.ops_seen += 1
        partial = sharding.partial.data
        if partial not in ("sum", "mean", "max", "min"):
            return
        if "compgen.reduce_scatter_inserted" in op.attributes:
            return
        if not op.results:
            return

        src = op.results[0]
        devices = [int(d.value.data) for d in sharding.devices.data]
        axis = int(scatter_axis_attr.value.data)

        rs = ReduceScatterOp.build(
            operands=[src],
            result_types=[src.type],
            properties={
                "scatter_dim": IntegerAttr(axis, IntegerType(64)),
                "reduce_kind": ReduceKindAttr(partial),
                "replica_groups": _groups(devices),
                "sharding_in": sharding,
                "sharding_out": ShardingSpecAttr(
                    devices=devices,
                    dim_map=[d.data for d in sharding.dim_map.data],
                    partial="none",
                ),
            },
        )
        src.replace_by_if(rs.result, lambda use: use.operation is not rs)
        parent = op.parent_block()
        if parent is not None:
            parent.insert_op_after(rs, op)
        op.attributes["compgen.reduce_scatter_inserted"] = scatter_axis_attr
        self.stats.reduce_scatters_inserted += 1


def run_insert_reduce_scatter(
    module: ModuleOp,
) -> InsertReduceScatterStats:
    stats = InsertReduceScatterStats()
    walker = PatternRewriteWalker(
        _InsertReduceScatterPattern(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "InsertReduceScatterStats",
    "run_insert_reduce_scatter",
]
