"""``decompose_concat`` -- lower ``compgen.tensor_ext.concat`` into
``tensor.empty`` + a chain of ``tensor.insert_slice``.

Reconstruction of IREE's ``DecomposeConcatPass`` as a CompGen
PatternRewriter. Zero references to IREE at runtime; this module
owns the rewrite.

Semantics:

    %r = compgen.tensor_ext.concat dim(d) %a, %b, %c
        : (tensor<L1xMxNxf32>, tensor<L2xMxNxf32>, tensor<L3xMxNxf32>)
        -> tensor<L1+L2+L3 x M x N xf32>

lowers to:

    %d0 = tensor.empty()  : tensor<L1+L2+L3 x M x N xf32>
    %d1 = tensor.insert_slice %a into %d0[0,   0, 0][L1, M, N][1, 1, 1]
    %d2 = tensor.insert_slice %b into %d1[L1,  0, 0][L2, M, N][1, 1, 1]
    %r  = tensor.insert_slice %c into %d2[L1+L2, 0, 0][L3, M, N][1, 1, 1]

This works for any concat dim (outer and non-outer alike) because
``tensor.insert_slice`` accepts offsets on any axis. The
"transpose-to-outer" trick IREE uses is unnecessary.

Static-shape gate: the pass bails out when any input or the result
has a dynamic dim. Dynamic-shape support will land in a follow-up
(we'd need ``tensor.dim`` + dynamic-sized slices). The bail leaves
the concat op in place; it's a no-op, not a failure.

LLM-tool signature (for registration by the agent layer):

    tool_name="decompose_concat"
    wraps_pass="CompGen:DecomposeConcat"
    invent_slot="pattern_library/structural_lowering"
    policy="ReplaceEveryStaticConcat"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.tensor import EmptyOp, InsertSliceOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.payload.passes._shape_utils import static_shape_or_none
from compgen.ir.tensor_ext import ConcatOp


@dataclass
class DecomposeConcatStats:
    """Small stats bag returned to callers for observability / testing."""

    concat_ops_seen: int = 0
    concat_ops_rewritten: int = 0
    concat_ops_skipped_dynamic: int = 0

    @property
    def rewrite_rate(self) -> float:
        if self.concat_ops_seen == 0:
            return 1.0
        return self.concat_ops_rewritten / self.concat_ops_seen


class DecomposeConcatPattern(RewritePattern):
    """Match one ``compgen.tensor_ext.concat`` and lower it."""

    def __init__(self, stats: DecomposeConcatStats | None = None) -> None:
        self.stats = stats if stats is not None else DecomposeConcatStats()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConcatOp, rewriter: PatternRewriter) -> None:
        self.stats.concat_ops_seen += 1

        # Static-shape gate.
        result_shape = static_shape_or_none(op.result)
        if result_shape is None:
            self.stats.concat_ops_skipped_dynamic += 1
            return

        input_shapes: list[tuple[int, ...]] = []
        for v in op.inputs:
            s = static_shape_or_none(v)
            if s is None:
                self.stats.concat_ops_skipped_dynamic += 1
                return
            input_shapes.append(s)

        dim = op.dim.value.data
        if dim < 0 or dim >= len(result_shape):
            # Shouldn't happen (ConcatOp.verify_ catches this), but
            # defensive bail keeps the rewriter safe under malformed IR.
            return

        rank = len(result_shape)
        inserts: list[Operation] = []

        # Allocate destination.
        empty_op = EmptyOp([], op.result.type)
        inserts.append(empty_op)
        current_dst: SSAValue = empty_op.results[0]

        # Chain insert_slice ops.
        offset_on_dim = 0
        for inp_value, inp_shape in zip(op.inputs, input_shapes, strict=True):
            offsets = [0] * rank
            offsets[dim] = offset_on_dim
            sizes = list(inp_shape)
            strides = [1] * rank

            slice_op = InsertSliceOp.from_static_parameters(
                source=inp_value,
                dest=current_dst,
                offsets=offsets,
                sizes=sizes,
                strides=strides,
            )
            inserts.append(slice_op)
            current_dst = slice_op.result
            offset_on_dim += inp_shape[dim]

        # Sanity: accumulated offset should equal the result extent on
        # the concat dim. This is an invariant already enforced by
        # ConcatOp.verify_ but asserting here catches regressions from
        # future dynamic-shape changes.
        assert offset_on_dim == result_shape[dim], (
            f"decompose_concat: accumulated offset {offset_on_dim} != result extent {result_shape[dim]} on dim {dim}"
        )

        rewriter.replace_matched_op(inserts, new_results=[current_dst])
        self.stats.concat_ops_rewritten += 1


def run_decompose_concat(
    module: ModuleOp,
    *,
    apply_recursively: bool = True,
) -> DecomposeConcatStats:
    """Apply the ``decompose_concat`` pattern to a module in place.

    Returns the :class:`DecomposeConcatStats` so callers can assert
    how many ops rewrote vs were skipped.
    """
    stats = DecomposeConcatStats()
    pattern = DecomposeConcatPattern(stats=stats)
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([pattern]),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "DecomposeConcatPattern",
    "DecomposeConcatStats",
    "run_decompose_concat",
]
