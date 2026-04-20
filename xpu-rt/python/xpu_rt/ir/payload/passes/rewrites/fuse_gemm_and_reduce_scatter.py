"""``fuse_gemm_and_reduce_scatter`` -- fuse a matmul with the
immediately-following ``reduce_scatter``.

Mirrors XLA's ``AsyncCollectiveCreator`` / hexagon-mlir's static
GEMM+RS overlap. When a matmul produces a partial-sum tensor and
the next op is ``compgen.collective.reduce_scatter``, merge them
into one tagged op so the Triton/codegen emitter can emit the
overlapped ring-reduce kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.linalg import MatmulOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.collective import ReduceScatterOp


@dataclass
class FuseGemmReduceScatterStats:
    matmuls_seen: int = 0
    fusions_applied: int = 0


class _FuseGEMMRS(RewritePattern):
    def __init__(self, stats: FuseGemmReduceScatterStats) -> None:
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        self.stats.matmuls_seen += 1
        if "compgen.gemm_rs_fused" in op.attributes:
            return
        if not op.res.types:
            return
        # Find an RS consumer of op's result.
        result = op.res[0]
        for use in result.uses:
            consumer = use.operation
            if isinstance(consumer, ReduceScatterOp):
                op.attributes["compgen.gemm_rs_fused"] = StringAttr("true")
                op.attributes["compgen.gemm_rs_scatter_dim"] = consumer.scatter_dim
                op.attributes["compgen.gemm_rs_reduce_kind"] = consumer.reduce_kind.kind
                self.stats.fusions_applied += 1
                return


def run_fuse_gemm_and_reduce_scatter(
    module: ModuleOp,
) -> FuseGemmReduceScatterStats:
    stats = FuseGemmReduceScatterStats()
    walker = PatternRewriteWalker(
        _FuseGEMMRS(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "FuseGemmReduceScatterStats",
    "run_fuse_gemm_and_reduce_scatter",
]
