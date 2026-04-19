"""``fold_transposes_into_dots`` -- absorb transposes feeding matmul.

Reconstruction of XLA's ``TransposeFolding`` pass as a CompGen
PatternRewriter. Zero external references; this module owns the rewrite.

Two folds are implemented:

1. ``matmul(transpose(a), b)`` and ``matmul(a, transpose(b))`` --
   absorb the 2D transpose into the matmul by rewriting its
   ``indexing_maps`` property. The new matmul reads the operand
   directly without materializing the transposed tensor.

2. Double-transpose elimination: ``transpose(transpose(x, p1), p2)``
   where ``p2 @ p1 == identity`` collapses to ``x``.

The two patterns run in the same ``GreedyRewritePatternApplier`` so
chains like ``matmul(transpose(transpose(a)), b)`` simplify in one
pass.

Only 2D transposes with permutation ``[1, 0]`` participate in the
matmul fold. Higher-rank transposes or non-``[1, 0]`` perms produce
shapes ``linalg.matmul`` can't consume without a generic lowering.
When such cases appear in practice they will be handled by the
Wave 3 ``propagate_transposes`` pass.

LLM-tool signature:

    tool_name="fold_transposes_into_dots"
    wraps_pass="CompGen:TransposeFolding"
    invent_slot="pattern_library/algebraic_fold"
    policy="AbsorbEvery2DTransposeIntoDot"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    ModuleOp,
)
from xdsl.dialects.linalg import MatmulOp, TransposeOp
from xdsl.ir import SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class FoldTransposesStats:
    """Observability: how many folds fired of each kind."""

    matmuls_seen: int = 0
    transpose_a_folds: int = 0
    transpose_b_folds: int = 0
    transpose_both_folds: int = 0
    double_transpose_eliminations: int = 0


# --- helpers -----------------------------------------------------------------


def _use_count(value: SSAValue) -> int:
    """Count ``value.uses`` without relying on ``__len__`` support.

    xDSL's ``IRUses`` is iterable but not sized, so we materialize it.
    """
    count = 0
    for _ in value.uses:
        count += 1
    return count


def _defining_transpose(value: SSAValue) -> TransposeOp | None:
    """Return the ``linalg.transpose`` that produces ``value``, else ``None``.

    The transpose must have exactly one use (the matmul) -- otherwise
    absorbing the transpose would leave a dangling producer for other
    consumers.
    """
    defining = value.owner if hasattr(value, "owner") else None
    if not isinstance(defining, TransposeOp):
        return None
    if _use_count(value) != 1:
        return None
    return defining


def _permutation_list(op: TransposeOp) -> list[int]:
    perm_attr = op.permutation
    # TransposeOp.permutation is a DenseArrayBase of i64.
    if isinstance(perm_attr, DenseArrayBase):
        return [int(v) for v in perm_attr.get_values()]
    return []


def _is_2d_reverse_perm(perm: list[int]) -> bool:
    return perm == [1, 0]


def _matmul_default_maps() -> list[AffineMapAttr]:
    """The three indexing maps for a plain ``linalg.matmul``.

    Matches ``MatmulOp.indexing_maps``'s default:
        lhs: (i, j, k) -> (i, k)
        rhs: (i, j, k) -> (k, j)
        out: (i, j, k) -> (i, j)
    """
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    return [
        AffineMapAttr(AffineMap(3, 0, (i, k))),
        AffineMapAttr(AffineMap(3, 0, (k, j))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]


def _matmul_transpose_a_maps() -> list[AffineMapAttr]:
    """lhs read as ``a[k, i]`` (transposed), rhs + out unchanged."""
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    return [
        AffineMapAttr(AffineMap(3, 0, (k, i))),
        AffineMapAttr(AffineMap(3, 0, (k, j))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]


def _matmul_transpose_b_maps() -> list[AffineMapAttr]:
    """rhs read as ``b[j, k]`` (transposed), lhs + out unchanged."""
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    return [
        AffineMapAttr(AffineMap(3, 0, (i, k))),
        AffineMapAttr(AffineMap(3, 0, (j, k))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]


def _matmul_transpose_both_maps() -> list[AffineMapAttr]:
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    return [
        AffineMapAttr(AffineMap(3, 0, (k, i))),
        AffineMapAttr(AffineMap(3, 0, (j, k))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]


# --- patterns ----------------------------------------------------------------


class FoldMatmulTransposePattern(RewritePattern):
    """Absorb a 2D transpose on either matmul input."""

    def __init__(self, stats: FoldTransposesStats | None = None) -> None:
        self.stats = stats if stats is not None else FoldTransposesStats()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: MatmulOp, rewriter: PatternRewriter
    ) -> None:
        self.stats.matmuls_seen += 1

        lhs = op.inputs[0]
        rhs = op.inputs[1]

        lhs_transpose = _defining_transpose(lhs)
        rhs_transpose = _defining_transpose(rhs)

        lhs_foldable = (
            lhs_transpose is not None
            and _is_2d_reverse_perm(_permutation_list(lhs_transpose))
        )
        rhs_foldable = (
            rhs_transpose is not None
            and _is_2d_reverse_perm(_permutation_list(rhs_transpose))
        )

        if not lhs_foldable and not rhs_foldable:
            return

        # Assemble the new matmul. Keep outs + result type, swap inputs
        # to the transpose source(s), install new indexing maps.
        new_lhs = lhs_transpose.input if lhs_foldable else lhs
        new_rhs = rhs_transpose.input if rhs_foldable else rhs

        if lhs_foldable and rhs_foldable:
            new_maps = _matmul_transpose_both_maps()
            self.stats.transpose_both_folds += 1
        elif lhs_foldable:
            new_maps = _matmul_transpose_a_maps()
            self.stats.transpose_a_folds += 1
        else:
            new_maps = _matmul_transpose_b_maps()
            self.stats.transpose_b_folds += 1

        # Build the replacement matmul. linalg.matmul accepts a
        # property-level ``indexing_maps`` that we set here.
        new_matmul = MatmulOp(
            inputs=[new_lhs, new_rhs],
            outputs=list(op.outputs),
            res=list(op.res.types),
            attributes=dict(op.attributes),
        )
        new_matmul.properties["indexing_maps"] = ArrayAttr(new_maps)

        # Preserve region-id / pattern-hint attributes if present.
        for key in ("compgen.region_id", "compgen._pattern_hint"):
            if key in op.attributes and key not in new_matmul.attributes:
                new_matmul.attributes[key] = op.attributes[key]

        rewriter.replace_matched_op(new_matmul)


class EliminateDoubleTransposePattern(RewritePattern):
    """``transpose(transpose(x, p1), p2)`` with ``p2 ∘ p1 == identity`` -> ``x``."""

    def __init__(self, stats: FoldTransposesStats | None = None) -> None:
        self.stats = stats if stats is not None else FoldTransposesStats()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: TransposeOp, rewriter: PatternRewriter
    ) -> None:
        # Look at the producer of ``op.input``.
        source = op.input
        producer = source.owner if hasattr(source, "owner") else None
        if not isinstance(producer, TransposeOp):
            return
        # Only safe when the inner transpose has no other uses.
        if _use_count(source) != 1:
            return

        inner_perm = _permutation_list(producer)
        outer_perm = _permutation_list(op)
        if not inner_perm or not outer_perm:
            return
        if len(inner_perm) != len(outer_perm):
            return

        # Compose: outer takes dim `outer_perm[i]` of the inner result,
        # which is dim `inner_perm[outer_perm[i]]` of the original.
        composed = [inner_perm[outer_perm[i]] for i in range(len(outer_perm))]
        if composed != list(range(len(composed))):
            return

        # Replace the outer transpose with the inner's input directly.
        rewriter.replace_matched_op([], new_results=[producer.input])
        self.stats.double_transpose_eliminations += 1


# --- entry point --------------------------------------------------------------


def run_fold_transposes_into_dots(
    module: ModuleOp,
    *,
    apply_recursively: bool = True,
) -> FoldTransposesStats:
    """Apply both folds to ``module`` in place."""
    stats = FoldTransposesStats()
    patterns = [
        FoldMatmulTransposePattern(stats=stats),
        EliminateDoubleTransposePattern(stats=stats),
    ]
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "EliminateDoubleTransposePattern",
    "FoldMatmulTransposePattern",
    "FoldTransposesStats",
    "run_fold_transposes_into_dots",
]
