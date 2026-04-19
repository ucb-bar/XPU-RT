"""Operations for the ``compgen.quant`` dialect.

Each op mirrors a TorchAO / PyTorch ``quantized_decomposed`` op exactly
so that the FX→xDSL importer can emit them directly instead of opaque
``func.call``s.

All ops are ``Pure`` (no side effects, their result is a pure function
of their operands + properties); buffer-planning passes must honour
that when deciding in-place aliasing.

Key design choices:

- The *result type* is an ordinary ``TensorType``. Quantization-specific
  metadata rides as properties on the op (``granularity``, ``axis``,
  ``group_size``, ``quant_min``, ``quant_max``, ``output_dtype``).
  Passes that need a richer type description attach an
  ``AffineQuantizedTensorType`` via the optional
  ``qtype`` property (defined in :mod:`compgen.ir.quant.types`).
- Scale + zero_point + scales_and_zeros operands stay as ordinary
  tensor SSA values. This matches how TorchAO's ``AffineQuantizedTensor``
  stores them (subclass-level attributes of the tensor instance, which
  in the compiled graph become first-class SSA values).
- The per-group ops carry an explicit ``group_size`` property (last-dim
  blocks), matching ``torch.ops.quantized_decomposed.
  quantize_per_group_along_last_dim``. If you need a non-last-dim
  grouping, use ``AffineQuantizedTensorType.block_size``.
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.quant.types import AffineQuantizedTensorType


# -- small shared validators ----------------------------------------------------


def _verify_quant_range(op: Operation, quant_min_attr, quant_max_attr) -> None:
    """Ensure quant_min < quant_max when both are set."""
    if quant_min_attr is None or quant_max_attr is None:
        return
    lo = quant_min_attr.value.data
    hi = quant_max_attr.value.data
    if lo >= hi:
        raise VerifyException(
            f"{op.name}: quant_min ({lo}) must be strictly less than "
            f"quant_max ({hi})"
        )


# -- Per-tensor affine quantize / dequantize -----------------------------------


@irdl_op_definition
class QuantizePerTensorOp(IRDLOperation):
    """Affine quantize of a float tensor with a single scalar scale and zero_point.

    Matches ``torch.ops.quantized_decomposed.quantize_per_tensor.default``::

        out = clamp(round(input / scale) + zero_point, quant_min, quant_max)

    Operands:
        input: the float tensor to quantize.
        scale: a scalar float tensor (``tensor<f32>``).
        zero_point: a scalar integer tensor (``tensor<i32>``).

    Properties:
        quant_min, quant_max: integer clamp range (typically -128, 127
            for int8; 0, 255 for uint8).
        output_dtype: string tag for the storage dtype ("int8",
            "uint8", "int4", ...). Informative only; the actual
            storage type is the result's element type.
        qtype: optional ``AffineQuantizedTensorType`` describing
            granularity / layout for downstream passes.

    Result: a tensor with integer (or sub-byte) element type.
    """

    name = "compgen.quant.quantize_per_tensor"

    input = operand_def(Attribute)
    scale = operand_def(Attribute)
    zero_point = operand_def(Attribute)
    result = result_def(Attribute)

    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    output_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)


@irdl_op_definition
class DequantizePerTensorOp(IRDLOperation):
    """Affine dequantize of an integer tensor back to floats.

    Matches ``torch.ops.quantized_decomposed.dequantize_per_tensor.default``::

        out = (input - zero_point) * scale
    """

    name = "compgen.quant.dequantize_per_tensor"

    input = operand_def(Attribute)
    scale = operand_def(Attribute)
    zero_point = operand_def(Attribute)
    result = result_def(Attribute)

    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    input_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)


# -- Per-channel affine quantize / dequantize ----------------------------------


@irdl_op_definition
class QuantizePerChannelOp(IRDLOperation):
    """Per-channel affine quantize.

    Matches ``torch.ops.quantized_decomposed.quantize_per_channel.default``.
    Scales and zero_points are vectors along ``axis``.
    """

    name = "compgen.quant.quantize_per_channel"

    input = operand_def(Attribute)
    scales = operand_def(Attribute)
    zero_points = operand_def(Attribute)
    result = result_def(Attribute)

    axis = prop_def(IntegerAttr)
    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    output_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)


@irdl_op_definition
class DequantizePerChannelOp(IRDLOperation):
    """Per-channel affine dequantize."""

    name = "compgen.quant.dequantize_per_channel"

    input = operand_def(Attribute)
    scales = operand_def(Attribute)
    zero_points = operand_def(Attribute)
    result = result_def(Attribute)

    axis = prop_def(IntegerAttr)
    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    input_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)


# -- Per-group affine quantize / dequantize (GPTQ/AWQ style) -------------------


@irdl_op_definition
class QuantizePerGroupOp(IRDLOperation):
    """Per-group affine quantize.

    Matches ``torch.ops.quantized_decomposed.
    quantize_per_group_along_last_dim.default``. Scales + zero_points
    are shaped ``[..., K / group_size]``.
    """

    name = "compgen.quant.quantize_per_group"

    input = operand_def(Attribute)
    scales = operand_def(Attribute)
    zero_points = operand_def(Attribute)
    result = result_def(Attribute)

    group_size = prop_def(IntegerAttr)
    axis = opt_prop_def(IntegerAttr)
    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    output_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)
        if self.group_size.value.data <= 0:
            raise VerifyException(
                f"{self.name}: group_size must be positive, "
                f"got {self.group_size.value.data}"
            )


@irdl_op_definition
class DequantizePerGroupOp(IRDLOperation):
    """Per-group affine dequantize."""

    name = "compgen.quant.dequantize_per_group"

    input = operand_def(Attribute)
    scales = operand_def(Attribute)
    zero_points = operand_def(Attribute)
    result = result_def(Attribute)

    group_size = prop_def(IntegerAttr)
    axis = opt_prop_def(IntegerAttr)
    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    input_dtype = opt_prop_def(StringAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)
        if self.group_size.value.data <= 0:
            raise VerifyException(
                f"{self.name}: group_size must be positive, "
                f"got {self.group_size.value.data}"
            )


# -- Packed-weight GEMMs (TorchAO + aten) --------------------------------------


@irdl_op_definition
class WeightInt8PackMMOp(IRDLOperation):
    """Packed int8 weight GEMM.

    Mirrors ``aten._weight_int8pack_mm.default(input, weight_i8, scales)``.

    The weight is stored as int8, with per-output-channel scales.
    """

    name = "compgen.quant.weight_int8pack_mm"

    input = operand_def(Attribute)
    weight = operand_def(Attribute)
    scales = operand_def(Attribute)
    result = result_def(Attribute)

    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())


@irdl_op_definition
class WeightInt4PackMMOp(IRDLOperation):
    """Packed int4 weight GEMM.

    Mirrors ``aten._weight_int4pack_mm.default(input, weight_i4,
    group_size, scales_and_zeros)`` from TorchAO's int4 weight-only
    quantization path.

    Operands:
        input: the float or bfloat16 activation tensor.
        weight: the int4-packed weight tensor (stored as int32 or int8
            depending on backend).
        scales_and_zeros: a ``[num_groups, out_channels, 2]`` or similar
            tensor packing per-group scale+zp pairs.

    Properties:
        group_size: the last-dim group size (e.g. 32, 64, 128).
    """

    name = "compgen.quant.weight_int4pack_mm"

    input = operand_def(Attribute)
    weight = operand_def(Attribute)
    scales_and_zeros = operand_def(Attribute)
    result = result_def(Attribute)

    group_size = prop_def(IntegerAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    _VALID_GROUP_SIZES: ClassVar[frozenset[int]] = frozenset({32, 64, 128, 256})

    def verify_(self) -> None:
        gs = self.group_size.value.data
        if gs not in self._VALID_GROUP_SIZES:
            raise VerifyException(
                f"{self.name}: group_size must be one of "
                f"{sorted(self._VALID_GROUP_SIZES)}, got {gs}"
            )


@irdl_op_definition
class WeightInt4PackQMOp(IRDLOperation):
    """Batched packed int4 weight GEMM.

    Mirrors ``aten._weight_int4pack_qm.default`` -- the batched
    (quantized-matmul, "qm") variant of int4 pack-mm used for LLM
    attention paths.
    """

    name = "compgen.quant.weight_int4pack_qm"

    input = operand_def(Attribute)
    weight = operand_def(Attribute)
    scales_and_zeros = operand_def(Attribute)
    result = result_def(Attribute)

    group_size = prop_def(IntegerAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())


# -- QParam selection ----------------------------------------------------------


@irdl_op_definition
class ChooseQParamsPerTensorOp(IRDLOperation):
    """Compute per-tensor scale + zero_point from an input tensor.

    Mirrors ``aten._choose_qparams_per_tensor.default``. Yields two
    scalar tensors.
    """

    name = "compgen.quant.choose_qparams_per_tensor"

    input = operand_def(Attribute)
    scale = result_def(Attribute)
    zero_point = result_def(Attribute)

    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    reduce_range = opt_prop_def(IntegerAttr)  # 0 / 1 boolean

    traits = traits_def(Pure())


@irdl_op_definition
class ChooseQParamsPerChannelOp(IRDLOperation):
    """Compute per-channel scale + zero_point vectors.

    Mirrors ``aten._choose_qparams_per_channel.default``.
    """

    name = "compgen.quant.choose_qparams_per_channel"

    input = operand_def(Attribute)
    scales = result_def(Attribute)
    zero_points = result_def(Attribute)

    axis = prop_def(IntegerAttr)
    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())


# -- QAT fake-quantize ---------------------------------------------------------


@irdl_op_definition
class FakeQuantOp(IRDLOperation):
    """Straight-through fake-quantize for QAT.

    Emits ``q = clamp(round(x/scale) + zp, lo, hi)`` followed by
    ``y = (q - zp) * scale`` in a single op, with the STE gradient
    semantics preserved by the downstream lowering. Mirrors
    ``torchao.quantization.qat.affine_fake_quantize``.
    """

    name = "compgen.quant.fake_quantize"

    input = operand_def(Attribute)
    scale = operand_def(Attribute)
    zero_point = operand_def(Attribute)
    result = result_def(Attribute)

    quant_min = opt_prop_def(IntegerAttr)
    quant_max = opt_prop_def(IntegerAttr)
    granularity = opt_prop_def(StringAttr)
    group_size = opt_prop_def(IntegerAttr)
    axis = opt_prop_def(IntegerAttr)
    qtype = opt_prop_def(AffineQuantizedTensorType)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        _verify_quant_range(self, self.quant_min, self.quant_max)


# -- Top-level registry --------------------------------------------------------


QUANT_OPS: list[type[IRDLOperation]] = [
    QuantizePerTensorOp,
    DequantizePerTensorOp,
    QuantizePerChannelOp,
    DequantizePerChannelOp,
    QuantizePerGroupOp,
    DequantizePerGroupOp,
    WeightInt8PackMMOp,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    ChooseQParamsPerTensorOp,
    ChooseQParamsPerChannelOp,
    FakeQuantOp,
]


__all__ = [
    "QUANT_OPS",
    "ChooseQParamsPerChannelOp",
    "ChooseQParamsPerTensorOp",
    "DequantizePerChannelOp",
    "DequantizePerGroupOp",
    "DequantizePerTensorOp",
    "FakeQuantOp",
    "QuantizePerChannelOp",
    "QuantizePerGroupOp",
    "QuantizePerTensorOp",
    "WeightInt4PackMMOp",
    "WeightInt4PackQMOp",
    "WeightInt8PackMMOp",
]
