"""``set_numerics_policy`` -- insert float casts around ops the target
kernel family cannot consume natively.

Reconstruction of XLA's ``FloatNormalizationPass`` as a CompGen
PatternRewriter. Zero external references; this module owns the rewrite.

XLA's pass walks HLO ops and, for each op whose operand or result type
isn't in the target's supported-float set, inserts a pair of
``convert`` ops: ``lo -> hi`` around the operand and ``hi -> lo``
around the result. The op itself runs at the promoted precision
while the surrounding graph keeps its storage type.

CompGen's version operates on xDSL linalg + math + arith ops. The
input is a **target numerics policy** describing:

- ``storage_types`` -- float types legally flowing through untouched
  ops (e.g. ``{bf16, f16, f32}``).
- ``supported_compute_types`` -- types each family of op can consume
  natively (e.g. ``f32`` for targets without a bf16 ALU).
- ``promotion_type`` -- when an op is illegal for its current type,
  promote to this type (default ``f32``).

The rewrite:

1. For each op in the module, read its ``kind`` (mapped from the op's
   MLIR name) and its effective operand element type.
2. If the element type is in ``supported_compute_types[kind]``, no
   change.
3. Otherwise: insert an ``arith.extf`` on each operand casting to
   ``promotion_type``, rebuild the op at ``promotion_type``, then
   emit an ``arith.truncf`` on the result back to the original
   storage type so consumers see the unchanged type.

This is applied idempotently: after the first pass, operands already
match ``promotion_type`` and the gate skips.

Applies only to **elementwise** math/arith ops in . Contraction
ops (matmul/conv) go through ``demote_contraction_inputs``
or the  ``propagate_transposes`` pass instead, which has the
structural awareness to handle mixed-precision GEMM bodies correctly.

LLM-tool signature:

    tool_name="set_numerics_policy"
    wraps_pass="CompGen:FloatNormalization"
    invent_slot="numerics/precision_policy"
    policy="PromoteElementwiseToF32OnUnsupportedKernels"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.arith import (
    AddfOp,
    DivfOp,
    ExtFOp,
    MaximumfOp,
    MinimumfOp,
    MulfOp,
    SubfOp,
    TruncFOp,
)
from xdsl.dialects.builtin import (
    BFloat16Type,
    FixedBitwidthType,
    Float16Type,
    Float32Type,
    Float64Type,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

# --- configuration ------------------------------------------------------------


@dataclass
class NumericsPolicy:
    """Describes the target's legal float type combinations.

    Attributes:
        storage_types: float types legal to pass through untouched ops.
        supported_per_kind: per-op-kind set of natively-consumable
            float types.  Keys are op-kind strings ("elementwise_add",
            "elementwise_div", ...).  Entries not in the map default
            to ``storage_types`` (i.e. "everything legal").
        promotion_type: the higher-precision float to promote to when
            the current type is not supported.
    """

    storage_types: frozenset[type[Attribute]] = field(
        default_factory=lambda: frozenset({Float16Type, BFloat16Type, Float32Type, Float64Type})
    )
    supported_per_kind: dict[str, frozenset[type[Attribute]]] = field(default_factory=dict)
    promotion_type_cls: type[Attribute] = Float32Type

    def is_supported(self, kind: str, elem_type: Attribute) -> bool:
        allowed = self.supported_per_kind.get(kind)
        if allowed is None:
            # Unrestricted for this kind.
            return True
        return type(elem_type) in allowed

    def make_promotion_type(self) -> Attribute:
        return self.promotion_type_cls()


@dataclass
class SetNumericsPolicyStats:
    ops_seen: int = 0
    ops_promoted: int = 0
    operands_extf_inserted: int = 0
    results_truncf_inserted: int = 0
    ops_skipped_legal: int = 0


# --- op kind map --------------------------------------------------------------


# We only touch the elementwise float ops below; contraction/reduction ops
# have their own dedicated numerics passes.
_KINDED_ARITH: dict[type[Operation], str] = {
    AddfOp: "elementwise_add",
    SubfOp: "elementwise_sub",
    MulfOp: "elementwise_mul",
    DivfOp: "elementwise_div",
    MaximumfOp: "elementwise_max",
    MinimumfOp: "elementwise_min",
}


# --- helpers ------------------------------------------------------------------


def _element_type(value: SSAValue) -> Attribute | None:
    t = value.type
    if isinstance(t, TensorType):
        return t.get_element_type()
    if isinstance(t, FixedBitwidthType):
        return t
    return None


def _needs_tensor_type(value: SSAValue, elem: Attribute) -> Attribute:
    t = value.type
    if isinstance(t, TensorType):
        return TensorType(elem, list(t.get_shape()))
    return elem


def _build_cast(src: SSAValue, target_elem: Attribute, *, widen: bool) -> Operation:
    """Build the right scalar cast op for ``src``'s type.

    When both src and target are scalar floats (bitwidth-driven
    ``ExtFOp`` / ``TruncFOp``), emit directly. This matches XLA's
    ``convert`` that works on scalars without needing ``linalg.generic``
    for tensors -- we apply the same shortcut as long as the source
    is a scalar float. For tensor operands we bail and let the
    demote / kernel pass handle broadcast.
    """
    new_type = _needs_tensor_type(src, target_elem)
    op_cls = ExtFOp if widen else TruncFOp
    return op_cls(src, new_type)


def _is_tensor_operand(value: SSAValue) -> bool:
    return isinstance(value.type, TensorType)


# --- pattern ------------------------------------------------------------------


class _NumericsElementwisePattern(RewritePattern):
    """Promote elementwise arith float ops whose current type is illegal."""

    def __init__(
        self,
        policy: NumericsPolicy,
        stats: SetNumericsPolicyStats,
    ) -> None:
        self.policy = policy
        self.stats = stats

    def _handle(self, op: Operation, rewriter: PatternRewriter) -> None:
        kind = _KINDED_ARITH.get(type(op))
        if kind is None:
            return
        self.stats.ops_seen += 1

        if not op.operands:
            return
        src0 = op.operands[0]
        elem = _element_type(src0)
        if elem is None:
            return
        if self.policy.is_supported(kind, elem):
            self.stats.ops_skipped_legal += 1
            return

        promotion_type = self.policy.make_promotion_type()
        if type(elem) is type(promotion_type):
            # Already at the promotion precision but policy rejected
            # it -> can't promote further; bail.
            return

        # The scalar-only arith ops we touch here don't take tensor
        # operands anyway (linalg.generic owns the tensor broadcast
        # path). So the cast ops stay scalar.
        if any(_is_tensor_operand(v) for v in op.operands):
            # Skip tensor-shaped elementwise; those live in
            # linalg.generic bodies where this pass's extf/truncf
            # would be illegal. A dedicated linalg-body pass handles
            # them in .
            return

        # Build ext casts, rebuild op at promotion type, truncf back.
        ext_ops: list[Operation] = []
        new_operands: list[SSAValue] = []
        for v in op.operands:
            cast = _build_cast(v, promotion_type, widen=True)
            ext_ops.append(cast)
            new_operands.append(cast.result)
            self.stats.operands_extf_inserted += 1

        new_op = type(op).__call__(*new_operands) if False else type(op)(*new_operands)
        trunc = _build_cast(new_op.result, elem, widen=False)
        self.stats.results_truncf_inserted += 1

        # Preserve pattern-hint/region-id.
        for key in ("compgen.region_id", "compgen._pattern_hint"):
            if key in op.attributes and key not in new_op.attributes:
                new_op.attributes[key] = op.attributes[key]

        rewriter.replace_matched_op(
            ext_ops + [new_op, trunc],
            new_results=[trunc.result],
        )
        self.stats.ops_promoted += 1


class _AddfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


class _SubfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SubfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


class _MulfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MulfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


class _DivfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DivfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


class _MaxfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MaximumfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


class _MinfRW(_NumericsElementwisePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MinimumfOp, rewriter: PatternRewriter) -> None:
        self._handle(op, rewriter)


# --- entry point --------------------------------------------------------------


def run_set_numerics_policy(
    module: ModuleOp,
    *,
    policy: NumericsPolicy | None = None,
    apply_recursively: bool = False,
) -> SetNumericsPolicyStats:
    """Apply the numerics policy to ``module`` in place."""
    p = policy if policy is not None else NumericsPolicy()
    stats = SetNumericsPolicyStats()
    patterns = [
        _AddfRW(p, stats),
        _SubfRW(p, stats),
        _MulfRW(p, stats),
        _DivfRW(p, stats),
        _MaxfRW(p, stats),
        _MinfRW(p, stats),
    ]
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier(patterns),
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "NumericsPolicy",
    "SetNumericsPolicyStats",
    "run_set_numerics_policy",
]
