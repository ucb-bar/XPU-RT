"""``insert_all_gather`` -- materialize ``xpu_rt.collective.all_gather``
where a consumer reads a sharded tensor that it needs replicated.

Triggered by an op tagged with ``xpu_rt.gather_axis`` (the dim
along which shards live). The pass is typically paired with
``shard_tensors_spmd`` which decides the sharded axis up front.
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

from xpu_rt.ir.collective import AllGatherOp, ShardingSpecAttr


@dataclass
class InsertAllGatherStats:
    ops_seen: int = 0
    all_gathers_inserted: int = 0


def _groups(devices: list[int]) -> ArrayAttr:
    n = 1
    for d in devices:
        n *= d
    return ArrayAttr([ArrayAttr([IntegerAttr(i, IntegerType(64)) for i in range(n)])])


class _InsertAllGatherPattern(RewritePattern):
    def __init__(self, stats: InsertAllGatherStats) -> None:
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        sharding = op.attributes.get("xpu_rt.sharding")
        gather_axis_attr = op.attributes.get("xpu_rt.gather_axis")
        if not isinstance(sharding, ShardingSpecAttr):
            return
        if gather_axis_attr is None:
            return
        self.stats.ops_seen += 1
        if "xpu_rt.all_gather_inserted" in op.attributes:
            return
        if not op.results:
            return

        src = op.results[0]
        devices = [int(d.value.data) for d in sharding.devices.data]
        axis = int(gather_axis_attr.value.data)

        ag = AllGatherOp.build(
            operands=[src],
            result_types=[src.type],
            properties={
                "all_gather_dim": IntegerAttr(axis, IntegerType(64)),
                "replica_groups": _groups(devices),
                "sharding_in": sharding,
                "sharding_out": ShardingSpecAttr(
                    devices=devices,
                    dim_map=["replicated"] * len(sharding.dim_map.data),
                ),
            },
        )
        src.replace_by_if(ag.result, lambda use: use.operation is not ag)
        parent = op.parent_block()
        if parent is not None:
            parent.insert_op_after(ag, op)
        op.attributes["xpu_rt.all_gather_inserted"] = gather_axis_attr
        self.stats.all_gathers_inserted += 1


def run_insert_all_gather(module: ModuleOp) -> InsertAllGatherStats:
    stats = InsertAllGatherStats()
    walker = PatternRewriteWalker(
        _InsertAllGatherPattern(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "InsertAllGatherStats",
    "run_insert_all_gather",
]
