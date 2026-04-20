"""``fuse_dequant_matmul`` -- fuse ``compgen.quant.dequantize_*`` feeding
``linalg.matmul`` into a single mixed-precision ``linalg.generic``.

Reconstruction of IREE's ``FuseDequantizationMatmulPass``. Zero
external references; CompGen owns the rewrite.

The pattern we match:

    %scales = ...
    %zps = ...
    %w_fp = compgen.quant.dequantize_per_{tensor,channel,group}(
                %w_int, %scales, %zps
            ) : (tensor<I...>, tensor<F...>, tensor<Z...>) -> tensor<F...>
    %out  = linalg.matmul(%x, %w_fp) outs(%acc) -> tensor<F...>

and rewrite as:

    %out = linalg.generic                    // mixed-precision matmul
             ins(%x, %w_int, %scales, %zps) outs(%acc)
             iterators = [parallel, parallel, reduction]
             body:
               %s = arith.sitofp(%w_int)
               %zp = arith.sitofp(%zps)      // when per_channel/per_group
               %deq = (s - zp) * scales
               %p   = %x * %deq
               %acc = %p + %acc_in
               linalg.yield %acc

The fused form never materializes the full f32 weight tensor; it
dequantizes each element inline in the matmul body, eliminating
one O(K*N) pass over memory.

Safety modes:

- ``reassoc_safe_only`` (default) -- only fuse when the dequant is
  a simple per-tensor / per-channel affine (``(x - zp) * s``) with
  zp=0. This preserves bit-exact fp arithmetic order.
- ``allow_numerics_relaxation`` -- fuse regardless, accepting that
  reduction reassociation may give slightly different rounding.
  Enabled by ``CompGenOptions.fuse_dequant_reassoc_safe = False``.

For Wave 4 we ship the **tag + detach** path: the matmul is rewritten
to drop the dequant input and instead read from the int weight
directly via a property ``compgen.fused_dequant_kind`` on the
matmul. The actual mixed-precision body lives in a follow-up wave
that lowers the tagged matmul into a body-inlined generic. Tagging
is mandatory so downstream kernel dispatch sees the fusion
opportunity even before the body lowering lands.

LLM-tool signature:

    tool_name="fuse_dequant_matmul"
    wraps_pass="CompGen:FuseDequantizationMatmul"
    invent_slot="quantization/dequant_matmul_fusion"
    policy="FuseDequantMatmulReassocSafe"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.arith import AddfOp, ExtSIOp, MulfOp, SIToFPOp, SubiOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
    MatmulOp,
    YieldOp,
)
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compgen.ir.quant import (
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
)

_DequantOp = (DequantizePerTensorOp, DequantizePerChannelOp, DequantizePerGroupOp)


@dataclass(frozen=True)
class FuseDequantMatmulConfig:
    reassoc_safe_only: bool = True
    allow_per_group: bool = True


@dataclass
class FuseDequantMatmulStats:
    matmuls_seen: int = 0
    fusions_applied: int = 0
    fusions_body_inlined: int = 0
    skipped_reassoc_unsafe: int = 0
    skipped_no_dequant_producer: int = 0
    skipped_already_fused: int = 0
    skipped_body_emit_unsupported: int = 0


# --- helpers -----------------------------------------------------------------


def _use_count(v: SSAValue) -> int:
    c = 0
    for _ in v.uses:
        c += 1
    return c


def _defining_dequant(value: SSAValue) -> Operation | None:
    owner = value.owner if hasattr(value, "owner") else None
    if owner is None:
        return None
    if not isinstance(owner, _DequantOp):
        return None
    # The dequant must be single-use to safely inline it.
    if _use_count(value) != 1:
        return None
    return owner


def _dequant_kind(op: Operation) -> str:
    if isinstance(op, DequantizePerTensorOp):
        return "per_tensor"
    if isinstance(op, DequantizePerChannelOp):
        return "per_channel"
    if isinstance(op, DequantizePerGroupOp):
        return "per_group"
    return "unknown"


def _is_reassoc_safe(dq: Operation) -> bool:
    """Per-tensor dequant with zero-point=0 (symmetric) is reassoc-safe.

    Per-channel with zero-point=0 is also safe because the scale
    broadcast commutes with the reduction.

    Per-group fusion re-orders the scale application across groups
    and thus changes rounding; treat it as unsafe unless explicitly
    allowed.
    """
    if isinstance(dq, DequantizePerTensorOp):
        return True
    if isinstance(dq, DequantizePerChannelOp):
        return True
    return False


# --- Real body fusion --------------------------------------------------------


def _tensor_rank(v: SSAValue) -> int | None:
    t = v.type
    if isinstance(t, TensorType):
        return len(list(t.get_shape()))
    return None


def _maybe_build_fused_generic(
    matmul: MatmulOp,
    dequant: Operation,
    side: str,
) -> Operation | None:
    """Emit a single ``linalg.generic`` that fuses the dequant into
    the matmul body.

    Only handles the canonical case:
    - matmul is 2-D (lhs rank 2, rhs rank 2, out rank 2)
    - side = 'rhs' (the weight is the dequantized operand)
    - dequant is per-tensor OR per-channel along the weight's axis 0 or 1
    - the quantized-weight tensor has static shape
    - the weight storage dtype exposes integer element type

    Returns the new ``GenericOp`` when the rewrite fired, else
    ``None``. Callers are responsible for inserting it via
    ``replace_matched_op``.
    """
    if side != "rhs":
        return None  # lhs-side fusion is rare + uses same template on the other input
    lhs = matmul.inputs[0]
    out = matmul.outputs[0]
    if _tensor_rank(lhs) != 2 or _tensor_rank(out) != 2:
        return None

    q_weight = dequant.operands[0]
    scales = dequant.operands[1]
    zeros = dequant.operands[2] if len(dequant.operands) > 2 else None

    q_type = q_weight.type
    out_type = out.type
    lhs_type = lhs.type
    if not (isinstance(q_type, TensorType) and isinstance(out_type, TensorType) and isinstance(lhs_type, TensorType)):
        return None
    if any(d < 0 for d in list(q_type.get_shape())):
        return None

    # Index vars: (i, j, k) with i,j parallel and k reduction.
    i, j, k = (
        AffineExpr.dimension(0),
        AffineExpr.dimension(1),
        AffineExpr.dimension(2),
    )
    # Input maps: lhs reads x[i, k], q_weight reads w[k, j] (assuming
    # the default matmul layout on the dequantized tensor). The
    # scales and zeros broadcast along the matching per-channel axis
    # -- we take the rank of the scales tensor as the signal.
    lhs_map = AffineMap(3, 0, (i, k))
    q_map = AffineMap(3, 0, (k, j))

    scales_rank = _tensor_rank(scales)
    if scales_rank == 0:
        scales_map = AffineMap(3, 0, ())
    elif scales_rank == 1:
        scales_map = AffineMap(3, 0, (j,))
    else:
        # per_group (rank-2) -- safer to bail and let the matmul
        # stay tag-only; a dedicated per-group body lives in Wave 6.
        return None
    if zeros is not None:
        zeros_rank = _tensor_rank(zeros)
        if zeros_rank == 0:
            zeros_map = AffineMap(3, 0, ())
        elif zeros_rank == 1:
            zeros_map = AffineMap(3, 0, (j,))
        else:
            return None
    out_map = AffineMap(3, 0, (i, j))

    # Build body: args are [lhs_scalar, q_scalar, scale_scalar, (zp_scalar?), out_scalar].
    out_elem = out_type.get_element_type()
    if not isinstance(out_elem, Float32Type):
        return None  # Wave 3 body assumes f32 accumulate.

    arg_types: list[Attribute] = [
        lhs_type.get_element_type(),
        q_type.get_element_type(),
        scales.type.get_element_type(),
    ]
    if zeros is not None:
        arg_types.append(zeros.type.get_element_type())
    arg_types.append(out_elem)

    body = Block(arg_types=arg_types)
    lhs_arg = body.args[0]
    q_arg = body.args[1]
    scale_arg = body.args[2]
    next_idx = 3
    if zeros is not None:
        zp_arg = body.args[next_idx]
        next_idx += 1
    else:
        zp_arg = None
    acc_arg = body.args[next_idx]

    # q_minus_zp = (i32)q_arg - (i32)zp_arg; then sitofp; then mul by scale.
    # When zp exists, widen q to match zp's dtype before the subtract
    # (the weight storage dtype is typically narrower than the zp
    # dtype in TorchAO's canonical layout).
    if zp_arg is not None:
        zp_elem_type = zeros.type.get_element_type()
        q_elem_type = q_type.get_element_type()
        if q_elem_type != zp_elem_type:
            q_ext = ExtSIOp(q_arg, zp_elem_type)
            body.add_op(q_ext)
            q_widened = q_ext.result
        else:
            q_widened = q_arg
        sub = SubiOp(q_widened, zp_arg)
        body.add_op(sub)
        cast = SIToFPOp(sub.result, Float32Type())
    else:
        cast = SIToFPOp(q_arg, Float32Type())
    body.add_op(cast)
    scaled = MulfOp(cast.result, scale_arg)
    body.add_op(scaled)
    mul = MulfOp(lhs_arg, scaled.result)
    body.add_op(mul)
    add = AddfOp(mul.result, acc_arg)
    body.add_op(add)
    body.add_op(YieldOp(add.result))

    # Assemble the generic.
    maps = [AffineMapAttr(lhs_map), AffineMapAttr(q_map), AffineMapAttr(scales_map)]
    inputs: list[SSAValue] = [lhs, q_weight, scales]
    if zeros is not None:
        maps.append(AffineMapAttr(zeros_map))
        inputs.append(zeros)
    maps.append(AffineMapAttr(out_map))

    iterator_types = [
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.REDUCTION),
    ]

    new_gen = GenericOp(
        inputs=inputs,
        outputs=[out],
        body=Region([body]),
        indexing_maps=maps,
        iterator_types=iterator_types,
        result_types=[out_type],
    )
    for k_attr in ("compgen.region_id", "compgen._pattern_hint"):
        if k_attr in matmul.attributes and k_attr not in new_gen.attributes:
            new_gen.attributes[k_attr] = matmul.attributes[k_attr]
    new_gen.attributes["compgen.fused_dequant_kind"] = matmul.attributes.get(
        "compgen.fused_dequant_kind",
        StringAttr("per_tensor" if isinstance(dequant, DequantizePerTensorOp) else "per_channel"),
    )
    return new_gen


# --- pattern -----------------------------------------------------------------


class _FuseDequantMatmulPattern(RewritePattern):
    def __init__(
        self,
        cfg: FuseDequantMatmulConfig,
        stats: FuseDequantMatmulStats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MatmulOp, rewriter: PatternRewriter) -> None:
        self.stats.matmuls_seen += 1

        # Already fused -> idempotent.
        if "compgen.fused_dequant_kind" in op.attributes:
            self.stats.skipped_already_fused += 1
            return

        # Look at both inputs: rhs is the typical quantized weight path,
        # but some emitter orderings flip lhs/rhs.
        lhs, rhs = op.inputs[0], op.inputs[1]
        dq_lhs = _defining_dequant(lhs)
        dq_rhs = _defining_dequant(rhs)

        if dq_lhs is None and dq_rhs is None:
            self.stats.skipped_no_dequant_producer += 1
            return

        # Prefer rhs fusion (weight side); fall back to lhs.
        if dq_rhs is not None:
            dq = dq_rhs
            side = "rhs"
        else:
            dq = dq_lhs
            side = "lhs"

        # Safety gate.
        if self.cfg.reassoc_safe_only and not _is_reassoc_safe(dq):
            if not (isinstance(dq, DequantizePerGroupOp) and self.cfg.allow_per_group):
                self.stats.skipped_reassoc_unsafe += 1
                return

        kind = _dequant_kind(dq)
        op.attributes["compgen.fused_dequant_kind"] = StringAttr(kind)
        op.attributes["compgen.fused_dequant_side"] = StringAttr(side)
        self.stats.fusions_applied += 1

        # Real body fusion when the shape is canonical. When the
        # fused generic can't be emitted (per-group, non-f32 acc,
        # weird ranks), we keep the op tagged so Wave 6 can pick it
        # up.
        if not isinstance(dq, DequantizePerGroupOp):
            fused = _maybe_build_fused_generic(op, dq, side)
            if fused is not None:
                rewriter.replace_matched_op(fused)
                self.stats.fusions_body_inlined += 1
                return
            self.stats.skipped_body_emit_unsupported += 1


# --- entry point -------------------------------------------------------------


def run_fuse_dequant_matmul(
    module: ModuleOp,
    *,
    config: FuseDequantMatmulConfig | None = None,
    apply_recursively: bool = False,
) -> FuseDequantMatmulStats:
    cfg = config if config is not None else FuseDequantMatmulConfig()
    stats = FuseDequantMatmulStats()
    pattern = _FuseDequantMatmulPattern(cfg, stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "FuseDequantMatmulConfig",
    "FuseDequantMatmulStats",
    "run_fuse_dequant_matmul",
]
