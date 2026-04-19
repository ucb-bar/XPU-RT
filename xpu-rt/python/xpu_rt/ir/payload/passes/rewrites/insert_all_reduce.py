"""``insert_all_reduce`` -- materialize ``compgen.collective.all_reduce``
at every op whose sharding spec is marked ``partial="sum"``.

Reconstruction of the insertion half of XLA's SPMD lowering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.collective import AllReduceOp, ReduceKindAttr, ShardingSpecAttr


@dataclass
class InsertAllReduceStats:
    ops_seen: int = 0
    all_reduces_inserted: int = 0
    skipped_no_partial: int = 0


def _replica_groups(devices: list[int]) -> ArrayAttr:
    n = 1
    for d in devices:
        n *= d
    return ArrayAttr(
        [ArrayAttr([IntegerAttr(i, IntegerType(64)) for i in range(n)])]
    )


class _InsertAllReducePattern(RewritePattern):
    def __init__(self, stats: InsertAllReduceStats) -> None:
        self.stats = stats

    def match_and_rewrite(
        self, op: Operation, rewriter: PatternRewriter
    ) -> None:
        sharding = op.attributes.get("compgen.sharding")
        if not isinstance(sharding, ShardingSpecAttr):
            return
        self.stats.ops_seen += 1
        partial = sharding.partial.data
        if partial not in ("sum", "mean", "max", "min"):
            self.stats.skipped_no_partial += 1
            return
        if "compgen.all_reduce_inserted" in op.attributes:
            return
        if not op.results:
            return

        src = op.results[0]
        devices = [int(d.value.data) for d in sharding.devices.data]
        kind = ReduceKindAttr(partial)
        groups = _replica_groups(devices)

        ar = AllReduceOp.build(
            operands=[src],
            result_types=[src.type],
            properties={
                "reduce_kind": kind,
                "replica_groups": groups,
                "sharding_in": sharding,
                # output sharding clears partial
                "sharding_out": ShardingSpecAttr(
                    devices=devices,
                    dim_map=[d.data for d in sharding.dim_map.data],
                    partial="none",
                ),
            },
        )
        # Replace uses of the op's result with the all_reduce output.
        src.replace_by_if(ar.result, lambda use: use.operation is not ar)
        parent = op.parent_block()
        if parent is not None:
            parent.insert_op_after(ar, op)
        op.attributes["compgen.all_reduce_inserted"] = sharding.partial
        self.stats.all_reduces_inserted += 1


def run_insert_all_reduce(
    module: ModuleOp,
) -> InsertAllReduceStats:
    stats = InsertAllReduceStats()
    walker = PatternRewriteWalker(
        _InsertAllReducePattern(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "InsertAllReduceStats",
    "run_insert_all_reduce",
]
