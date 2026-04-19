"""``hoist_encoding_ops`` -- move compgen.quant dequantize ops closer
to their producer so downstream fusion has a shorter use-def chain.

Mirror of IREE's ``HoistEncodingOps`` + ``MaterializeEncodings``.
When a dequantize op sits between a producer and a single
consumer, hoisting it up to the producer creates a clear
fuse-dequant-into-producer opportunity.

The pass is conservative: it only tags the dequantize with
``compgen.hoist_candidate`` when:
- the dequantize has a single use, AND
- its input (the quantized tensor) has a single use (the dequantize).

Structural op movement lands alongside the Wave 6 memory-planning
pass — at this layer we surface the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.quant import (
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
)


_DEQUANT_OPS = (DequantizePerTensorOp, DequantizePerChannelOp, DequantizePerGroupOp)


@dataclass
class HoistEncodingOpsStats:
    dequants_seen: int = 0
    candidates_tagged: int = 0


def _single_use(v: SSAValue) -> bool:
    c = 0
    for _ in v.uses:
        c += 1
        if c > 1:
            return False
    return c == 1


class _HoistPattern(RewritePattern):
    def __init__(self, stats: HoistEncodingOpsStats) -> None:
        self.stats = stats

    def match_and_rewrite(
        self, op: Operation, rewriter: PatternRewriter
    ) -> None:
        if not isinstance(op, _DEQUANT_OPS):
            return
        self.stats.dequants_seen += 1
        if "compgen.hoist_candidate" in op.attributes:
            return
        # single-use on the dequant result AND on the quant input
        if not _single_use(op.results[0]):
            return
        q_input = op.operands[0]
        if not _single_use(q_input):
            return
        op.attributes["compgen.hoist_candidate"] = StringAttr("true")
        op.attributes["compgen.hoist_dequant_kind"] = StringAttr(
            type(op).__name__
        )
        self.stats.candidates_tagged += 1


def run_hoist_encoding_ops(
    module: ModuleOp,
) -> HoistEncodingOpsStats:
    stats = HoistEncodingOpsStats()
    walker = PatternRewriteWalker(
        _HoistPattern(stats), apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "HoistEncodingOpsStats",
    "run_hoist_encoding_ops",
]
