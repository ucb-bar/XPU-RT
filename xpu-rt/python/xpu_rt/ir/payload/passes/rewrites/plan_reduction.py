"""``plan_reduction`` -- choose the best reduction strategy per op.

Reconstruction of XLA's ``ReductionDimensionGrouper`` +
``TreeReductionRewriter`` as a CompGen PatternRewriter. Zero external
references; CompGen owns the rewrite.

Three sub-strategies:

- ``group`` -- move all reduction iteration dims to the end of the
  iteration space so they're contiguous. On ``linalg.generic`` ops
  this is a **real structural rewrite**: iterator_types and
  indexing_maps get permuted, operand shapes stay the same (affine
  maps absorb the permutation). Lets downstream tiling treat the
  trailing reduction dims as one fused reduction nest.
- ``split`` -- split one large reduce dim into an outer reduction
  plus an inner parallel reduction (``outer_size`` rows each
  accumulating ``inner_size`` elements). Reduces register pressure
  and improves cache locality for long reductions. Threshold: a
  reduce dim whose extent is ``> large_reduction_threshold``
  becomes ``split``. Annotational in Wave 3; the structural body
  change lands in a follow-up tiling pass.
- ``tree_reduce`` -- cascade the reduction into a balanced tree of
  halving adds. Annotational today.

Beyond choosing a strategy, this pass tags every matching reduction
op with ``compgen.reduction_strategy`` + ``compgen.reduction_extent``
so downstream tiling / codegen can pick the right implementation.
The ``group`` path goes further: it physically reorders the
iteration domain.

Split + tree_reduce remain annotational because the structural
loop-nest transforms they need haven't landed yet. The annotation
contract is however *mandatory*
-- without it, the kernel generator has no signal to pick the
strategy, which makes Wave 7 end-to-end tests fail on large-reduce
workloads like Qwen-MoE's expert combine.

The rewrite also walks ``compgen.linalg_ext.softmax`` /
``rms_norm`` / ``layer_norm`` ops (all of which carry an implicit
reduction) and tags them the same way. This is the fast path for
the real-workload tests.

Configuration:

- ``policy`` -- force a specific strategy (``"group"`` / ``"split"``
  / ``"tree_reduce"``). Default is ``"auto"`` which chooses per
  shape.
- ``large_reduction_threshold`` -- extent above which ``auto`` picks
  ``split`` instead of ``group``. Default 512 (matches XLA's
  default).
- ``tree_reduce_threshold`` -- extent above which ``auto`` prefers
  ``tree_reduce`` over ``split``. Default 8192.

LLM-tool signature:

    tool_name="plan_reduction"
    wraps_pass="CompGen:TreeReductionRewriter"
    invent_slot="layout/reduction_planning"
    policy="AutoByExtent"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
)
from xdsl.ir import Operation
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

from compgen.ir.linalg_ext import LayerNormOp, RMSNormOp, SoftmaxOp

_VALID_POLICIES = frozenset({"auto", "group", "split", "tree_reduce"})
_VALID_STRATEGIES = frozenset({"group", "split", "tree_reduce"})


@dataclass(frozen=True)
class PlanReductionConfig:
    policy: str = "auto"
    large_reduction_threshold: int = 512
    tree_reduce_threshold: int = 8192

    def __post_init__(self) -> None:
        if self.policy not in _VALID_POLICIES:
            raise ValueError(f"policy must be one of {sorted(_VALID_POLICIES)}; got {self.policy!r}")
        if self.large_reduction_threshold <= 0:
            raise ValueError("large_reduction_threshold must be positive")
        if self.tree_reduce_threshold <= self.large_reduction_threshold:
            raise ValueError("tree_reduce_threshold must exceed large_reduction_threshold")


@dataclass
class PlanReductionStats:
    ops_seen: int = 0
    ops_annotated: int = 0
    chosen_group: int = 0
    chosen_split: int = 0
    chosen_tree_reduce: int = 0
    skipped_non_reduction: int = 0
    skipped_already_annotated: int = 0
    # Number of ops whose iteration space was structurally permuted
    # by the ``group`` strategy. A subset of ``chosen_group``.
    iteration_permutations_applied: int = 0


# --- helpers -----------------------------------------------------------------


def _reduction_iter_indices(op: GenericOp) -> list[int]:
    kinds = op.iterator_types
    if not isinstance(kinds, ArrayAttr):
        return []
    result: list[int] = []
    for i, k in enumerate(kinds.data):
        if isinstance(k, IteratorTypeAttr) and k.data == IteratorType.REDUCTION:
            result.append(i)
    return result


def _shape_of(op: Operation) -> tuple[int, ...] | None:
    # Use the first tensor input as the shape donor.
    for v in op.operands:
        t = v.type
        if isinstance(t, TensorType):
            return tuple(t.get_shape())
    return None


def _choose_auto(
    extent: int,
    cfg: PlanReductionConfig,
) -> str:
    if extent < 0:
        # Dynamic dim -- conservative default.
        return "group"
    if extent > cfg.tree_reduce_threshold:
        return "tree_reduce"
    if extent > cfg.large_reduction_threshold:
        return "split"
    return "group"


def _already_annotated(op: Operation) -> bool:
    return "compgen.reduction_strategy" in op.attributes


def _annotate(op: Operation, strategy: str, extent: int) -> None:
    op.attributes["compgen.reduction_strategy"] = StringAttr(strategy)
    op.attributes["compgen.reduction_extent"] = IntegerAttr(extent, IntegerType(64))


def _permute_affine_map(m: AffineMap, perm_inv: list[int]) -> AffineMap:
    """Rewrite ``m`` so each reference to iteration dim ``i`` becomes
    ``perm_inv[i]``.

    ``perm_inv`` is the inverse of the dim permutation we're applying:
    if ``perm`` is "new_order[i] = old_order[perm[i]]", then
    ``perm_inv[j] = position of j in perm``.

    Only supports identity-style dim references (each expr must be a
    plain ``AffineDimExpr``). Non-dim expressions (constants,
    additions, etc.) are left untouched.
    """
    new_exprs = []
    for expr in m.results:
        dim_pos = expr.position if hasattr(expr, "position") else None
        if dim_pos is not None and isinstance(dim_pos, int):
            new_exprs.append(AffineExpr.dimension(perm_inv[dim_pos]))
        else:
            # Non-trivial expr; leave it as-is. (In practice linalg
            # generics we emit only use AffineDimExpr.)
            new_exprs.append(expr)
    return AffineMap(m.num_dims, m.num_symbols, tuple(new_exprs))


def _try_group_structural_rewrite(op: GenericOp) -> bool:
    """Move reduction iterator dims to the end of the iteration space.

    Returns ``True`` when the rewrite fired (iteration order was
    actually changed). Returns ``False`` when the op was already
    "grouped" (all reductions trailing) or the rewrite is unsafe.

    Safety gates:
    - All iterator kinds must be either PARALLEL or REDUCTION.
      WINDOW / other kinds bail out.
    - All indexing_maps must be pure AffineDimExpr references (no
      constants, no compound exprs). This holds for ops emitted by
      our own pass pipeline; hand-rolled ops with weird maps are
      skipped.
    """
    kinds = op.iterator_types
    if not isinstance(kinds, ArrayAttr):
        return False
    kind_list = []
    for k in kinds.data:
        if not isinstance(k, IteratorTypeAttr):
            return False
        if k.data not in (IteratorType.PARALLEL, IteratorType.REDUCTION):
            return False
        kind_list.append(k.data)

    n = len(kind_list)
    parallel_idx = [i for i, k in enumerate(kind_list) if k == IteratorType.PARALLEL]
    reduction_idx = [i for i, k in enumerate(kind_list) if k == IteratorType.REDUCTION]
    if not reduction_idx:
        return False
    perm = parallel_idx + reduction_idx
    if perm == list(range(n)):
        return False  # already in grouped order

    # Verify every AffineMap uses only plain dim references.
    maps = op.indexing_maps
    if not isinstance(maps, ArrayAttr):
        return False
    new_maps: list[AffineMapAttr] = []
    perm_inv = [0] * n
    for new_pos, old_pos in enumerate(perm):
        perm_inv[old_pos] = new_pos
    for m_attr in maps.data:
        if not isinstance(m_attr, AffineMapAttr):
            return False
        m = m_attr.data
        for expr in m.results:
            if not hasattr(expr, "position") or not isinstance(expr.position, int):
                return False
        new_maps.append(AffineMapAttr(_permute_affine_map(m, perm_inv)))

    new_iter_kinds = [kind_list[old] for old in perm]
    op.properties["iterator_types"] = ArrayAttr([IteratorTypeAttr(k) for k in new_iter_kinds])
    op.properties["indexing_maps"] = ArrayAttr(new_maps)
    return True


# --- patterns ----------------------------------------------------------------


class _GenericReductionAnnotator(RewritePattern):
    def __init__(
        self,
        cfg: PlanReductionConfig,
        stats: PlanReductionStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, GenericOp):
            return
        red_idx = _reduction_iter_indices(op)
        if not red_idx:
            self.stats.skipped_non_reduction += 1
            return
        self.stats.ops_seen += 1
        if _already_annotated(op):
            self.stats.skipped_already_annotated += 1
            return

        shape = _shape_of(op)
        if shape is None:
            return

        # For reduction extents we multiply all reduction-dim extents.
        extent = 1
        for d in red_idx:
            if d < len(shape):
                ext = shape[d]
                if ext < 0:
                    extent = -1
                    break
                extent *= ext

        strategy = self.cfg.policy if self.cfg.policy != "auto" else _choose_auto(extent, self.cfg)
        _annotate(op, strategy, extent)
        self.stats.ops_annotated += 1
        if strategy == "group":
            self.stats.chosen_group += 1
            # Real structural rewrite: permute iteration dims so
            # all reduction dims are contiguous at the end.
            if _try_group_structural_rewrite(op):
                self.stats.iteration_permutations_applied += 1
        elif strategy == "split":
            self.stats.chosen_split += 1
        else:
            self.stats.chosen_tree_reduce += 1


class _LinalgExtReductionAnnotator(RewritePattern):
    """Tag ``compgen.linalg_ext.{softmax,rms_norm,layer_norm}``.

    Each of those ops carries an implicit last-axis reduction; the
    extent is the last dim of the input.
    """

    def __init__(
        self,
        cfg: PlanReductionConfig,
        stats: PlanReductionStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, (SoftmaxOp, RMSNormOp, LayerNormOp)):
            return
        self.stats.ops_seen += 1
        if _already_annotated(op):
            self.stats.skipped_already_annotated += 1
            return
        src_type = op.input.type
        if not isinstance(src_type, TensorType):
            return
        shape = list(src_type.get_shape())
        extent = shape[-1] if shape else -1

        strategy = self.cfg.policy if self.cfg.policy != "auto" else _choose_auto(extent, self.cfg)
        _annotate(op, strategy, extent)
        self.stats.ops_annotated += 1
        if strategy == "group":
            self.stats.chosen_group += 1
        elif strategy == "split":
            self.stats.chosen_split += 1
        else:
            self.stats.chosen_tree_reduce += 1


# --- entry point -------------------------------------------------------------


def run_plan_reduction(
    module: ModuleOp,
    *,
    config: PlanReductionConfig | None = None,
    apply_recursively: bool = False,
) -> PlanReductionStats:
    """Annotate reduction-carrying ops with a chosen strategy."""
    cfg = config if config is not None else PlanReductionConfig()
    stats = PlanReductionStats()
    patterns = [
        _GenericReductionAnnotator(cfg, stats),
        _LinalgExtReductionAnnotator(cfg, stats),
    ]
    for p in patterns:
        walker = PatternRewriteWalker(
            p,
            apply_recursively=apply_recursively,
        )
        walker.rewrite_module(module)
    return stats


__all__ = [
    "PlanReductionConfig",
    "PlanReductionStats",
    "run_plan_reduction",
]
