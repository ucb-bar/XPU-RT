"""``demote_contraction_inputs`` -- truncate f32 contraction operands to
a lower precision while keeping the accumulator in f32.

Reconstruction of IREE's ``DemoteContractionInputsToBF16Pass`` as a
CompGen PatternRewriter. Zero external references; this module owns
the rewrite.

Semantics (for ``linalg.matmul`` on f32 inputs + f32 accumulator):

    %lhs_lo = linalg.generic <truncf elementwise>   # f32 -> target (bf16/f16)
    %rhs_lo = linalg.generic <truncf elementwise>
    %out    = linalg.generic <mixed-precision matmul body>
        ins(%lhs_lo, %rhs_lo) outs(%acc)
        iterator_types = [parallel, parallel, reduction]
        body: extf(lhs_lo), extf(rhs_lo), mulf, addf(acc)

The truncf + matmul body are built with the W0.5
``linalg_generic_*`` helpers so the pass stays small and the indexing
maps are centrally reviewed.

Configuration:

- ``target_type`` -- the low-precision float type (default
  ``BFloat16Type``; set ``Float16Type`` for H100 FP16).
- ``restrict_to_region_ids`` -- optional allowlist of
  ``compgen.region_id`` values. Lets the LLM policy layer opt in
  per region.

Covered contraction ops: ``linalg.matmul`` only in Wave 1. The
quantized-weight variants
(``compgen.quant.weight_int{4,8}pack_{mm,qm}``) already carry integer
weights; the activation input is demoted through the generic
``linalg.generic`` elementwise path in Wave 4 (``lower_quantized_matmul``
covers the joint rewrite).

LLM-tool signature:

    tool_name="demote_contraction_inputs"
    wraps_pass="CompGen:DemoteContractionInputsToBF16"
    invent_slot="numerics/precision_policy"
    policy="DemoteActivationsOnMatrixEngineTargets"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.arith import AddfOp, ExtFOp, MulfOp, TruncFOp
from xdsl.dialects.builtin import (
    BFloat16Type,
    FixedBitwidthType,
    Float32Type,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.linalg import MatmulOp, YieldOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.payload.passes._builders import (
    linalg_generic_elementwise,
    linalg_generic_matmul_like,
)


@dataclass
class DemoteContractionStats:
    contractions_seen: int = 0
    contractions_rewritten: int = 0
    operands_truncated: int = 0
    contractions_skipped_wrong_dtype: int = 0
    contractions_skipped_region_filter: int = 0


# --- configuration -----------------------------------------------------------


@dataclass
class DemoteContractionInputsConfig:
    target_type: Attribute = field(default_factory=BFloat16Type)
    restrict_to_region_ids: frozenset[str] | None = None

    def is_allowed_region(self, region_id: str | None) -> bool:
        if self.restrict_to_region_ids is None:
            return True
        if region_id is None:
            return False
        return region_id in self.restrict_to_region_ids


# --- helpers ------------------------------------------------------------------


def _tensor_elem(value: SSAValue) -> Attribute | None:
    t = value.type
    if isinstance(t, TensorType):
        return t.get_element_type()
    return None


def _bitwidth(attr: Attribute) -> int | None:
    if isinstance(attr, FixedBitwidthType):
        return attr.bitwidth
    return None


def _region_id(op: Operation) -> str | None:
    attr = op.attributes.get("compgen.region_id")
    if attr is None:
        return None
    return attr.data


def _build_trunc_generic(
    src: SSAValue,
    target_elem: Attribute,
) -> tuple[list[Operation], SSAValue]:
    """Elementwise ``truncf`` via ``linalg.generic``.

    Returns ``(pre_ops, out_value)`` where ``pre_ops`` is
    ``[tensor.empty, linalg.generic]`` and ``out_value`` is the
    generic's result.
    """
    src_type = src.type
    assert isinstance(src_type, TensorType)
    shape = list(src_type.get_shape())
    out_type = TensorType(target_elem, shape)
    init = EmptyOp([], out_type)

    def body(args, block):
        trunc = TruncFOp(args[0], target_elem)
        block.add_op(trunc)
        block.add_op(YieldOp(trunc.result))

    generic = linalg_generic_elementwise(
        inputs=[src],
        init=init.results[0],
        result_type=out_type,
        body=body,
    )
    return [init, generic], generic.results[0]


def _build_mixed_matmul(
    lhs: SSAValue,
    rhs: SSAValue,
    acc: SSAValue,
    acc_elem: Attribute,
    result_type: Attribute,
) -> Operation:
    """Mixed-precision matmul body using ``linalg_generic_matmul_like``.

    The body reads ``lhs`` and ``rhs`` at ``target_type`` precision,
    widens via ``arith.extf`` to the accumulator's type, and
    multiplies + accumulates in that higher precision.
    """
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    lhs_map = AffineMap(3, 0, (i, k))
    rhs_map = AffineMap(3, 0, (k, j))
    out_map = AffineMap(3, 0, (i, j))

    def body(args, block):
        # args = [lhs_low, rhs_low, acc_hi]
        l_ext = ExtFOp(args[0], acc_elem)
        r_ext = ExtFOp(args[1], acc_elem)
        block.add_op(l_ext)
        block.add_op(r_ext)
        mul = MulfOp(l_ext.result, r_ext.result)
        block.add_op(mul)
        add = AddfOp(mul.result, args[2])
        block.add_op(add)
        block.add_op(YieldOp(add.result))

    return linalg_generic_matmul_like(
        lhs=lhs,
        rhs=rhs,
        init=acc,
        result_type=result_type,
        lhs_map=lhs_map,
        rhs_map=rhs_map,
        output_map=out_map,
        body=body,
    )


# --- pattern ------------------------------------------------------------------


class _LinalgMatmulDemote(RewritePattern):
    """Demote f32 ``linalg.matmul`` inputs to ``target_type``."""

    def __init__(
        self,
        cfg: DemoteContractionInputsConfig,
        stats: DemoteContractionStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        self.stats.contractions_seen += 1
        if not self.cfg.is_allowed_region(_region_id(op)):
            self.stats.contractions_skipped_region_filter += 1
            return

        lhs, rhs = op.inputs[0], op.inputs[1]
        acc = op.outputs[0]

        lhs_elem = _tensor_elem(lhs)
        rhs_elem = _tensor_elem(rhs)
        acc_elem = _tensor_elem(acc)
        if lhs_elem is None or rhs_elem is None or acc_elem is None:
            return

        target = self.cfg.target_type
        target_bits = _bitwidth(target)
        acc_bits = _bitwidth(acc_elem)
        if target_bits is None or acc_bits is None:
            return

        # Gate: accumulator must be strictly wider than target, and
        # both inputs must be at the accumulator's precision (i.e. f32).
        if acc_bits <= target_bits:
            self.stats.contractions_skipped_wrong_dtype += 1
            return
        if not isinstance(acc_elem, Float32Type):
            # Support only f32-accumulator matmuls in Wave 1.
            self.stats.contractions_skipped_wrong_dtype += 1
            return
        if lhs_elem != acc_elem or rhs_elem != acc_elem:
            self.stats.contractions_skipped_wrong_dtype += 1
            return
        if lhs_elem == target and rhs_elem == target:
            # Already demoted -- idempotent.
            return

        # Build trunc generics for each input.
        pre_ops: list[Operation] = []
        lhs_ops, lhs_new = _build_trunc_generic(lhs, target)
        rhs_ops, rhs_new = _build_trunc_generic(rhs, target)
        pre_ops.extend(lhs_ops)
        pre_ops.extend(rhs_ops)
        self.stats.operands_truncated += 2

        # Build the replacement mixed-precision matmul.
        result_type = op.res.types[0]
        mixed = _build_mixed_matmul(
            lhs=lhs_new,
            rhs=rhs_new,
            acc=acc,
            acc_elem=acc_elem,
            result_type=result_type,
        )

        # Preserve region-id / pattern-hint on the replacement.
        for key in ("compgen.region_id", "compgen._pattern_hint"):
            if key in op.attributes and key not in mixed.attributes:
                mixed.attributes[key] = op.attributes[key]

        rewriter.replace_matched_op(pre_ops + [mixed], new_results=[mixed.results[0]])
        self.stats.contractions_rewritten += 1


# --- entry point -------------------------------------------------------------


def run_demote_contraction_inputs(
    module: ModuleOp,
    *,
    config: DemoteContractionInputsConfig | None = None,
    apply_recursively: bool = False,
) -> DemoteContractionStats:
    """Apply the matmul-input demotion to ``module`` in place."""
    cfg = config if config is not None else DemoteContractionInputsConfig()
    stats = DemoteContractionStats()
    pattern = _LinalgMatmulDemote(cfg, stats)
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([pattern]),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "DemoteContractionInputsConfig",
    "DemoteContractionStats",
    "run_demote_contraction_inputs",
]
