"""``collective_quantizer`` -- fuse quantize/dequantize with collectives.

XLA's ``CollectiveQuantizer``: when a tensor is all_reduced / all_gathered
in fp32 and immediately quantized, move the quantize BEFORE the
collective so only the compressed bytes cross the wire.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.collective import AllGatherOp, AllReduceOp, ReduceScatterOp
from compgen.ir.quant import (
    DequantizePerChannelOp,
    DequantizePerTensorOp,
    QuantizePerChannelOp,
    QuantizePerTensorOp,
)

_COLLECTIVES = (AllReduceOp, AllGatherOp, ReduceScatterOp)
_QUANTIZES = (QuantizePerTensorOp, QuantizePerChannelOp)
_DEQUANTIZES = (DequantizePerTensorOp, DequantizePerChannelOp)


@dataclass
class CollectiveQuantizerStats:
    collectives_seen: int = 0
    fused_with_quantize: int = 0
    fused_with_dequantize: int = 0


class _FuseCollectiveQuant(RewritePattern):
    def __init__(self, stats: CollectiveQuantizerStats) -> None:
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, _COLLECTIVES):
            return
        self.stats.collectives_seen += 1
        if "compgen.quant_fused" in op.attributes:
            return
        # Look for a quantize consumer.
        for use in op.results[0].uses:
            consumer = use.operation
            if isinstance(consumer, _QUANTIZES):
                op.attributes["compgen.quant_fused"] = StringAttr("quantize_after")
                op.attributes["compgen.quant_fused_kind"] = StringAttr(type(consumer).__name__)
                self.stats.fused_with_quantize += 1
                return
            if isinstance(consumer, _DEQUANTIZES):
                op.attributes["compgen.quant_fused"] = StringAttr("dequantize_after")
                op.attributes["compgen.quant_fused_kind"] = StringAttr(type(consumer).__name__)
                self.stats.fused_with_dequantize += 1
                return


def run_collective_quantizer(
    module: ModuleOp,
) -> CollectiveQuantizerStats:
    stats = CollectiveQuantizerStats()
    walker = PatternRewriteWalker(
        _FuseCollectiveQuant(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "CollectiveQuantizerStats",
    "run_collective_quantizer",
]
