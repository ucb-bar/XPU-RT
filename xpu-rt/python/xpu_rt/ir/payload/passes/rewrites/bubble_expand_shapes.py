"""``bubble_expand_shapes`` -- move reshape / expand ops downward
past elementwise ops to expose fusion opportunities.

Mirror of IREE's ``BubbleExpandShapes``. When a view/reshape sits
between a producer and an elementwise generic, the generic can
often absorb the reshape via indexing-map composition. This pass
tags the reshape-elementwise pair for a later fusion pass.

The pass looks for:

    %v = func.call @aten_view(%x)   // reshape
    %y = func.call @aten_<elementwise>(%v)   // add / mul / gelu / ...

When the elementwise op has exactly one input and that input is a
reshape, tag the elementwise with
``compgen.bubble_reshape_through=true`` + the original input tensor
type so a later pass can compose.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import CallOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

_RESHAPE_HINTS = frozenset({"view", "expand", "reshape", "unsqueeze", "squeeze"})
_ELEMENTWISE_HINTS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "gelu",
        "silu",
        "neg",
        "sigmoid",
        "relu",
        "tanh",
    }
)


def _hint(op: Operation) -> str | None:
    attr = op.attributes.get("compgen._pattern_hint")
    return attr.data if isinstance(attr, StringAttr) else None


@dataclass
class BubbleExpandShapesStats:
    elementwise_seen: int = 0
    reshape_pairs_tagged: int = 0


class _BubbleExpandPattern(RewritePattern):
    def __init__(self, stats: BubbleExpandShapesStats) -> None:
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        hint = _hint(op)
        if hint not in _ELEMENTWISE_HINTS:
            return
        self.stats.elementwise_seen += 1
        if not op.operands:
            return
        producer = op.operands[0].owner
        if not isinstance(producer, CallOp):
            return
        prod_hint = _hint(producer)
        if prod_hint not in _RESHAPE_HINTS:
            return
        if "compgen.bubble_reshape_through" in op.attributes:
            return
        op.attributes["compgen.bubble_reshape_through"] = StringAttr(prod_hint)
        self.stats.reshape_pairs_tagged += 1


def run_bubble_expand_shapes(
    module: ModuleOp,
) -> BubbleExpandShapesStats:
    stats = BubbleExpandShapesStats()
    walker = PatternRewriteWalker(
        _BubbleExpandPattern(stats),
        apply_recursively=False,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "BubbleExpandShapesStats",
    "run_bubble_expand_shapes",
]
