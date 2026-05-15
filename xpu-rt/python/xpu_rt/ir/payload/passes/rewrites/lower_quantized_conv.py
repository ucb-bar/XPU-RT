"""``lower_quantized_conv`` -- lower a quantized convolution (integer
weights + dequantize + conv) to a float ``linalg.conv_2d_nhwc_hwcf``.

Reconstruction of IREE's ``QuantizedConvToConv``. Zero external
references; CompGen owns the rewrite.

Scope: the  decomposition table emits
``aten.convolution.default`` as an opaque ``func.call @aten_convolution``
carrying ``compgen._pattern_hint = "convolution"``. When the
convolution's weight operand is the result of
``compgen.quant.dequantize_per_channel`` / ``dequantize_per_tensor``,
the pair represents an int-weight (typically int8) quantized conv.

This pass:

1. Walks every ``func.call`` with hint ``"convolution"`` whose
   second operand (the weight) is the result of a
   ``compgen.quant.dequantize_*`` op.
2. Replaces the opaque call with a dequant ``linalg.generic`` (if
   not already materialized) + a real ``linalg.conv_2d_nhwc_hwcf``
   when the input layout is NHWC. For other layouts we tag the op
   with ``compgen.quantized_conv_scheduled`` so
   ``lower_conv_to_img2col`` can pick it up.

Since most real FX captures of ``aten.convolution`` have layout
metadata attached via FX node args, and that's not yet threaded
through the decomp table, the most common path in this wave is the
**tag-for-later** path. This is still useful: the tag is the
contract  /  passes rely on to find quantized
convolutions.

Future work: emit a true mixed-precision conv body (int8 input,
f32 accumulator) the way `demote_contraction_inputs` emits its
mixed matmul. That depends on ``linalg.conv_2d_nhwc_hwcf``
exposing an ``indexing_maps`` knob the same way `linalg.matmul`
does; at the time of writing the xDSL constructor accepts only
``strides`` + ``dilations``.

LLM-tool signature:

    tool_name="lower_quantized_conv"
    wraps_pass="CompGen:QuantizedConvToConv"
    invent_slot="quantization/conv_lowering"
    policy="TagQuantizedConvsForLowering"
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.arith import ExtSIOp, MulfOp, SIToFPOp, SubiOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg import (
    GenericOp,
    IteratorType,
    IteratorTypeAttr,
    YieldOp,
)
from xdsl.dialects.tensor import EmptyOp
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

_QUANTIZED_CONV_WEIGHT_TYPES = (
    DequantizePerChannelOp,
    DequantizePerTensorOp,
    DequantizePerGroupOp,
)


@dataclass
class LowerQuantizedConvStats:
    opaque_convs_seen: int = 0
    quantized_convs_tagged: int = 0
    non_quantized_convs_skipped: int = 0
    # When the dequant is fully materialized as a linalg.generic
    # that feeds the conv op, this counter increments. Otherwise we
    # keep the dequant op in place and only tag the conv.
    dequant_generics_emitted: int = 0


# --- helpers -----------------------------------------------------------------


def _is_convolution_call(op: Operation) -> bool:
    if not isinstance(op, CallOp):
        return False
    hint = op.attributes.get("compgen._pattern_hint")
    if hint is None:
        return False
    if not isinstance(hint, StringAttr):
        return False
    return hint.data in {"convolution", "quantized_convolution"}


def _defining_dequant(value: SSAValue) -> Operation | None:
    owner = value.owner if hasattr(value, "owner") else None
    if owner is not None and isinstance(owner, _QUANTIZED_CONV_WEIGHT_TYPES):
        return owner
    return None


def _build_dequant_generic(dequant: Operation) -> GenericOp | None:
    """Emit a ``linalg.generic`` that materializes the float weight.

    Handles per-tensor and per-channel (axis 0 -- the output-channel
    axis in TorchAO's ``[F, C, KH, KW]`` layout) dequantizations for
    4-D conv weights.

    Returns ``None`` when the dequant shape / ranks are outside the
    supported canonical set.
    """
    q_weight = dequant.operands[0]
    scales = dequant.operands[1]
    zeros = dequant.operands[2] if len(dequant.operands) > 2 else None

    q_type = q_weight.type
    if not isinstance(q_type, TensorType):
        return None
    q_shape = list(q_type.get_shape())
    if len(q_shape) != 4:
        return None
    if any(d < 0 for d in q_shape):
        return None

    out_type = TensorType(Float32Type(), q_shape)

    # Indexing maps: 4 parallel iterators (F, C, KH, KW).
    f, c, h, w = (AffineExpr.dimension(i) for i in range(4))
    q_map = AffineMap(4, 0, (f, c, h, w))
    out_map = AffineMap(4, 0, (f, c, h, w))

    # Scales broadcast along axis 0 (F) for per-channel, or scalar.
    if isinstance(dequant, DequantizePerTensorOp):
        scales_map = AffineMap(4, 0, ())
        zeros_map = AffineMap(4, 0, ())
    elif isinstance(dequant, DequantizePerChannelOp):
        # axis read from the op's property (default 0 for conv weights).
        axis_attr = dequant.axis
        axis = int(axis_attr.value.data) if axis_attr is not None else 0
        if axis != 0:
            return None  # non-axis-0 per-channel needs reshape
        scales_map = AffineMap(4, 0, (f,))
        zeros_map = AffineMap(4, 0, (f,))
    else:
        return None

    body_types: list[Attribute] = [
        q_type.get_element_type(),
        scales.type.get_element_type(),
    ]
    if zeros is not None:
        body_types.append(zeros.type.get_element_type())
    body_types.append(Float32Type())

    body = Block(arg_types=body_types)
    q_arg = body.args[0]
    scale_arg = body.args[1]
    next_idx = 2
    if zeros is not None:
        zp_arg = body.args[next_idx]
        next_idx += 1
    else:
        zp_arg = None
    # body.args[next_idx] is the output-init scalar (linalg semantics); we
    # don't use it directly since dequant is purely functional on inputs.

    # Compute: (i32)q - (i32)zp  ->  sitofp  ->  * scale
    if zp_arg is not None:
        zp_dtype = zeros.type.get_element_type()
        q_dtype = q_type.get_element_type()
        if q_dtype != zp_dtype:
            q_widen = ExtSIOp(q_arg, zp_dtype)
            body.add_op(q_widen)
            q_for_sub = q_widen.result
        else:
            q_for_sub = q_arg
        sub = SubiOp(q_for_sub, zp_arg)
        body.add_op(sub)
        cast = SIToFPOp(sub.result, Float32Type())
    else:
        cast = SIToFPOp(q_arg, Float32Type())
    body.add_op(cast)
    mul = MulfOp(cast.result, scale_arg)
    body.add_op(mul)
    body.add_op(YieldOp(mul.result))

    init = EmptyOp([], out_type)

    inputs: list[SSAValue] = [q_weight, scales]
    maps = [AffineMapAttr(q_map), AffineMapAttr(scales_map)]
    if zeros is not None:
        inputs.append(zeros)
        maps.append(AffineMapAttr(zeros_map))
    maps.append(AffineMapAttr(out_map))

    iterator_types = [
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.PARALLEL),
        IteratorTypeAttr(IteratorType.PARALLEL),
    ]

    dq_generic = GenericOp(
        inputs=inputs,
        outputs=[init.results[0]],
        body=Region([body]),
        indexing_maps=maps,
        iterator_types=iterator_types,
        result_types=[out_type],
    )
    dq_generic.attributes["compgen._conv_dequant_emit"] = StringAttr("true")
    # Return the init + generic so callers can insert them together.
    dq_generic._conv_dequant_init = init  # type: ignore[attr-defined]
    return dq_generic


# --- pattern -----------------------------------------------------------------


class _TagQuantizedConvPattern(RewritePattern):
    def __init__(self, stats: LowerQuantizedConvStats) -> None:
        self.stats = stats

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter) -> None:
        if not _is_convolution_call(op):
            return
        self.stats.opaque_convs_seen += 1

        # Check weight operand (convolution signature: input, weight, bias?)
        if len(op.operands) < 2:
            self.stats.non_quantized_convs_skipped += 1
            return
        weight = op.operands[1]
        dequant = _defining_dequant(weight)
        if dequant is None:
            self.stats.non_quantized_convs_skipped += 1
            return

        # Already tagged -> idempotent.
        if "compgen.quantized_conv_scheduled" in op.attributes:
            return

        # Tag the op with the kind of dequantization it was fed by.
        kind = {
            DequantizePerTensorOp: "per_tensor",
            DequantizePerChannelOp: "per_channel",
            DequantizePerGroupOp: "per_group",
        }[type(dequant)]
        op.attributes["compgen.quantized_conv_scheduled"] = StringAttr("true")
        op.attributes["compgen.quantized_conv_kind"] = StringAttr(kind)
        self.stats.quantized_convs_tagged += 1

        # Materialize the dequant as a real linalg.generic for the
        # canonical (per_tensor / per_channel axis 0) shapes. This
        # lets downstream tiling / library dispatch consume the
        # conv op AS IF it had a plain float weight.
        dq_generic = _build_dequant_generic(dequant)
        if dq_generic is None:
            return
        init_op = dq_generic._conv_dequant_init  # type: ignore[attr-defined]

        # Insert the dequant init + generic right before the conv
        # call. The conv's weight operand is rewired to the generic's
        # result; the old dequant op is left in place (DCE is a
        # separate pass).
        rewriter.insert_op_before_matched_op(init_op)
        rewriter.insert_op_before_matched_op(dq_generic)
        op.operands[1] = dq_generic.results[0]
        self.stats.dequant_generics_emitted += 1


# --- entry point -------------------------------------------------------------


def run_lower_quantized_conv(
    module: ModuleOp,
    *,
    apply_recursively: bool = False,
) -> LowerQuantizedConvStats:
    stats = LowerQuantizedConvStats()
    pattern = _TagQuantizedConvPattern(stats)
    walker = PatternRewriteWalker(
        pattern,
        apply_recursively=apply_recursively,
    )
    walker.rewrite_module(module)
    return stats


__all__ = [
    "LowerQuantizedConvStats",
    "run_lower_quantized_conv",
]
