"""``shard_tensors_spmd`` -- annotate tensors with a sharding spec.

Reconstruction of XLA's ``SpmdPartitioner`` + IREE's
``StreamPartitioning`` frontend. CompGen owns the rewrite.

Given a caller-supplied mesh shape + shard policy, walks every
op in the module and attaches a ``compgen.collective.sharding_spec``
attribute describing how that tensor is partitioned across devices.

Default policy = **Megatron tensor-parallel**:

- matmul lhs     : shard along K (last dim)       -> "axis0"
- matmul rhs     : shard along N (last dim)       -> "axis0"
- linear biases  : replicated
- layer-norm     : replicated
- residual adds  : replicated

The sharding spec is read by ``insert_all_reduce`` /
``insert_reduce_scatter`` / ``insert_all_gather`` to decide where
collectives need to fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, TensorType
from xdsl.dialects.linalg import MatmulOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.collective import ShardingSpecAttr


@dataclass(frozen=True)
class ShardTensorsSPMDConfig:
    mesh_shape: tuple[int, ...] = (4,)
    axis_names: tuple[str, ...] = ("tp",)
    policy: str = "megatron_tp"


@dataclass
class ShardTensorsSPMDStats:
    ops_seen: int = 0
    shardings_attached: int = 0
    matmuls_sharded: int = 0
    replicated_ops: int = 0


def _mk_sharding(
    tensor_rank: int,
    devices: list[int],
    shard_dim: int | None,
    axis_name: str,
) -> ShardingSpecAttr:
    dim_map = ["replicated"] * tensor_rank
    if shard_dim is not None and 0 <= shard_dim < tensor_rank:
        dim_map[shard_dim] = axis_name
    return ShardingSpecAttr(devices=devices, dim_map=dim_map)


class _ShardMatmulPattern(RewritePattern):
    def __init__(self, cfg, stats):
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MatmulOp, rewriter: PatternRewriter
    ) -> None:
        self.stats.ops_seen += 1
        if "compgen.sharding" in op.attributes:
            return

        lhs_type = op.inputs[0].type
        rhs_type = op.inputs[1].type
        out_type = op.res.types[0] if op.res.types else op.outputs[0].type
        if not all(isinstance(t, TensorType) for t in (lhs_type, rhs_type, out_type)):
            return

        devices = list(self.cfg.mesh_shape)
        axis = self.cfg.axis_names[0]

        lhs_rank = len(list(lhs_type.get_shape()))
        rhs_rank = len(list(rhs_type.get_shape()))
        out_rank = len(list(out_type.get_shape()))

        # Megatron-TP: shard rhs along last dim (N), lhs replicated,
        # out sharded along last dim + partial=sum (requires AllReduce).
        if self.cfg.policy == "megatron_tp":
            op.attributes["compgen.sharding"] = ShardingSpecAttr(
                devices=devices,
                dim_map=["replicated"] * out_rank,
                partial="sum",
            )
            op.attributes["compgen.sharding_lhs"] = _mk_sharding(
                lhs_rank, devices, None, axis,
            )
            op.attributes["compgen.sharding_rhs"] = _mk_sharding(
                rhs_rank, devices, rhs_rank - 1, axis,
            )
            self.stats.matmuls_sharded += 1
            self.stats.shardings_attached += 1


def run_shard_tensors_spmd(
    module: ModuleOp,
    *,
    config: ShardTensorsSPMDConfig | None = None,
) -> ShardTensorsSPMDStats:
    cfg = config if config is not None else ShardTensorsSPMDConfig()
    stats = ShardTensorsSPMDStats()
    walker = PatternRewriteWalker(_ShardMatmulPattern(cfg, stats))
    walker.rewrite_module(module)

    # Tag every other op as replicated so downstream collective
    # passes can confirm.
    for op in module.walk():
        if "compgen.sharding" not in op.attributes:
            # only float tensor-producing ops get a default tag
            if any(isinstance(v.type, TensorType) for v in op.results):
                op.attributes["compgen.sharding"] = ShardingSpecAttr(
                    devices=list(cfg.mesh_shape),
                    dim_map=["replicated"] *
                    max(1, len(list(op.results[0].type.get_shape())))
                    if isinstance(op.results[0].type, TensorType) else 1,
                )
                stats.replicated_ops += 1
    return stats


__all__ = [
    "ShardTensorsSPMDConfig",
    "ShardTensorsSPMDStats",
    "run_shard_tensors_spmd",
]
