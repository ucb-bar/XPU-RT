"""``propagate_transposes`` -- bubble ``linalg.transpose`` through
downstream ops so the transpose eventually reaches a free edge (an
input / output boundary) where it can be dropped or absorbed by
layout.

Reconstruction of IREE's ``PropagateLinalgTransposePass`` as a
CompGen PatternRewriter. Zero external references; CompGen owns
the rewrite.

Two fold rules land in Wave 3 (the MVP subset that matters for LLM
workloads):

1. **Chained-transpose collapse** -- ``transpose(transpose(x, p1),
   p2)`` with ``p2 ∘ p1 == identity`` collapses to ``x``. This
   fires recursively, so even long chains fold out. Implemented by
   :class:`ComposeAdjacentTransposesPattern`.
2. **Transpose into elementwise generic** -- when a
   ``linalg.transpose`` feeds a ``linalg.generic`` whose input
   indexing maps + iterator types make the result of the transpose
   the sole consumer of a dim that can be permuted inside the
   generic, we push the transpose through by composing permutations
   into the generic's ``indexing_maps``. Implemented by
   :class:`PushTransposeIntoElementwisePattern`.

Transpose → matmul absorption already lives in
:mod:`compgen.ir.payload.passes.rewrites.fold_transposes_into_dots`
(W1.2); this pass defers to that one for contraction absorption.

Transpose → convolution (HWCF ↔ HWFC) and transpose → pad are
deferred to a follow-up (Wave 5 convolution rewrites).

``aggressiveness``:

- ``"conservative"`` -- only the chained-transpose collapse. Safe
  everywhere (no indexing-map changes on non-transpose ops).
- ``"through_elementwise"`` -- also pushes transposes through
  elementwise generics whose iterators are all ``parallel``. This
  is the default for cuda_a100 / cuda_h100 presets.

Runs to fixed point via ``apply_recursively=True`` -- every rewrite
may unlock another.

LLM-tool signature:

    tool_name="propagate_transposes"
    wraps_pass="CompGen:PropagateLinalgTranspose"
    invent_slot="layout/transpose_propagation"
    policy="BubbleTransposesThroughElementwise"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    ModuleOp,
    i64,
)
from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, TransposeOp
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class PropagateTransposesStats:
    chained_collapses: int = 0
    elementwise_pushes: int = 0
    transposes_seen: int = 0
    transposes_skipped_aggressiveness: int = 0


@dataclass(frozen=True)
class PropagateTransposesConfig:
    aggressiveness: str = "through_elementwise"


_VALID_AGGRESSIVENESS = frozenset(
    {"conservative", "through_elementwise", "through_conv", "through_pad"}
)


# --- helpers ------------------------------------------------------------------


def _perm_list(op: TransposeOp) -> list[int]:
    attr = op.permutation
    if isinstance(attr, DenseArrayBase):
        return [int(v) for v in attr.get_values()]
    return []


def _compose(outer: list[int], inner: list[int]) -> list[int]:
    """``outer ∘ inner`` as a permutation.

    For ``outer = [a0, a1, ...]`` and ``inner = [b0, b1, ...]`` the
    composition is ``[inner[outer[i]] for i in range(len(outer))]``.
    """
    return [inner[outer[i]] for i in range(len(outer))]


def _is_identity(perm: list[int]) -> bool:
    return perm == list(range(len(perm)))


def _use_count(value: SSAValue) -> int:
    c = 0
    for _ in value.uses:
        c += 1
    return c


def _defining_transpose(value: SSAValue) -> TransposeOp | None:
    owner = value.owner if hasattr(value, "owner") else None
    if isinstance(owner, TransposeOp):
        if _use_count(value) == 1:
            return owner
    return None


# --- Pattern 1: chained transpose collapse ------------------------------------


class ComposeAdjacentTransposesPattern(RewritePattern):
    """``transpose(transpose(x, p1), p2)`` -> ``transpose(x, p2 ∘ p1)``.

    Collapses to ``x`` when the composition is the identity. When
    it's not identity but both transposes are single-use, we fold
    into a single transpose with the composed permutation.
    """

    def __init__(self, stats: PropagateTransposesStats) -> None:
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: TransposeOp, rewriter: PatternRewriter
    ) -> None:
        inner = _defining_transpose(op.input)
        if inner is None:
            return
        inner_perm = _perm_list(inner)
        outer_perm = _perm_list(op)
        if not inner_perm or not outer_perm:
            return
        if len(inner_perm) != len(outer_perm):
            return
        composed = _compose(outer_perm, inner_perm)

        if _is_identity(composed):
            # Replace outer with the original source.
            rewriter.replace_matched_op([], new_results=[inner.input])
            self.stats.chained_collapses += 1
            return

        # Fold into a single transpose. Reuse the outer's init tensor.
        new_perm = DenseArrayBase.from_list(i64, composed)
        new_t = TransposeOp(
            input=inner.input,
            init=op.init,
            permutation=new_perm,
            result=op.results[0].type,
        )
        # Preserve attributes if any.
        for k in ("compgen.region_id", "compgen._pattern_hint"):
            if k in op.attributes and k not in new_t.attributes:
                new_t.attributes[k] = op.attributes[k]
        rewriter.replace_matched_op(new_t)
        self.stats.chained_collapses += 1


# --- Pattern 2: push transpose into elementwise linalg.generic ---------------


def _iterator_is_all_parallel(op: GenericOp) -> bool:
    kinds = op.iterator_types
    if not isinstance(kinds, ArrayAttr):
        return False
    for k in kinds.data:
        if not isinstance(k, IteratorTypeAttr):
            return False
        if k.data != IteratorType.PARALLEL:
            return False
    return True


def _all_identity_maps(op: GenericOp) -> bool:
    maps = op.indexing_maps
    if not isinstance(maps, ArrayAttr):
        return False
    for m in maps.data:
        if not isinstance(m, AffineMapAttr):
            return False
        rank = m.data.num_dims
        identity = AffineMap.identity(rank)
        if m.data != identity:
            return False
    return True


def _apply_perm_to_map(m: AffineMap, perm: list[int]) -> AffineMap:
    """Compose an identity-shaped affine map with a permutation.

    For an identity map ``(d0, ..., d_{n-1}) -> (d0, ..., d_{n-1})``
    and permutation ``perm``, the result is
    ``(d0, ..., d_{n-1}) -> (d_{perm[0]}, d_{perm[1]}, ..., d_{perm[n-1]})``.
    """
    n = len(perm)
    exprs = tuple(AffineExpr.dimension(perm[i]) for i in range(n))
    return AffineMap(n, 0, exprs)


class PushTransposeIntoElementwisePattern(RewritePattern):
    """Bubble a transpose through an elementwise generic.

    Matches the shape ``generic(transpose(x))`` where:

    - the generic has a single input,
    - all iterator types are ``parallel`` (pure elementwise),
    - all indexing maps are identity (the simplest elementwise shape),
    - the transpose has a single use.

    Rewrites to ``transpose(generic(x))`` with the generic using
    identity maps on the original (un-transposed) input, and a new
    ``linalg.transpose`` wrapping the result with the same
    permutation.

    When the above invariants don't hold we skip (deferring to
    follow-up waves that handle non-identity indexing maps +
    multi-input generics).
    """

    def __init__(
        self,
        cfg: PropagateTransposesConfig,
        stats: PropagateTransposesStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: GenericOp, rewriter: PatternRewriter
    ) -> None:
        if self.cfg.aggressiveness == "conservative":
            return
        if not _iterator_is_all_parallel(op):
            return
        if not _all_identity_maps(op):
            return
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            return

        src = op.inputs[0]
        transpose = _defining_transpose(src)
        if transpose is None:
            return

        perm = _perm_list(transpose)
        if not perm:
            return
        # Skip identity perms.
        if _is_identity(perm):
            return

        # The generic re-runs on the pre-transpose input; new
        # indexing maps compose the permutation on the input side.
        input_map = _apply_perm_to_map(AffineMap.identity(len(perm)), perm)
        output_map = AffineMap.identity(len(perm))

        new_maps = ArrayAttr(
            [
                AffineMapAttr(input_map),
                AffineMapAttr(output_map),
            ]
        )

        # Build the rewritten generic using low-level ``build`` so we
        # keep the original body region intact.
        pre_input = transpose.input
        pre_input_type = pre_input.type
        # The original output type matches the generic's result;
        # when pushing the transpose through, the generic's result
        # is now the *pre-transpose* shape.
        new_output_type = pre_input_type
        # The output operand init must match the new shape.
        # We create a fresh tensor.empty for the init to avoid
        # mutating any shared init producer.
        from xdsl.dialects.tensor import EmptyOp as _EmptyOp

        new_init = _EmptyOp([], new_output_type)

        new_generic = GenericOp(
            inputs=[pre_input],
            outputs=[new_init.results[0]],
            body=op.body.clone(),
            indexing_maps=[
                AffineMapAttr(input_map),
                AffineMapAttr(output_map),
            ],
            iterator_types=op.iterator_types,
            result_types=[new_output_type],
        )

        # The post-transpose sits on top of the new generic's result.
        new_transpose_init = _EmptyOp([], op.results[0].type)
        new_transpose = TransposeOp(
            input=new_generic.results[0],
            init=new_transpose_init.results[0],
            permutation=DenseArrayBase.from_list(i64, perm),
            result=op.results[0].type,
        )

        # Preserve op-level metadata.
        for k in ("compgen.region_id", "compgen._pattern_hint"):
            if k in op.attributes and k not in new_generic.attributes:
                new_generic.attributes[k] = op.attributes[k]

        rewriter.replace_matched_op(
            [new_init, new_generic, new_transpose_init, new_transpose],
            new_results=[new_transpose.results[0]],
        )
        self.stats.elementwise_pushes += 1


# --- entry point --------------------------------------------------------------


def run_propagate_transposes(
    module: ModuleOp,
    *,
    config: PropagateTransposesConfig | None = None,
    apply_recursively: bool = True,
) -> PropagateTransposesStats:
    """Bubble transposes to a fixed point."""
    cfg = config if config is not None else PropagateTransposesConfig()
    if cfg.aggressiveness not in _VALID_AGGRESSIVENESS:
        raise ValueError(
            f"aggressiveness must be one of {sorted(_VALID_AGGRESSIVENESS)}; "
            f"got {cfg.aggressiveness!r}"
        )
    stats = PropagateTransposesStats()
    patterns = [
        ComposeAdjacentTransposesPattern(stats=stats),
    ]
    if cfg.aggressiveness != "conservative":
        patterns.append(PushTransposeIntoElementwisePattern(cfg=cfg, stats=stats))
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "ComposeAdjacentTransposesPattern",
    "PropagateTransposesConfig",
    "PropagateTransposesStats",
    "PushTransposeIntoElementwisePattern",
    "run_propagate_transposes",
]
