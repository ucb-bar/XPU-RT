"""CompGen quantization dialect.

``compgen.quant`` mirrors TorchAO's quantized-tensor model directly so
that Payload IR can represent quantized computations without opaque
calls. The dialect owns:

- **Types** -- ``AffineQuantizedTensorType``, ``PackedIntTensorType``,
  ``MXQuantizedTensorType``, ``NVFP4TensorType``. These are
  supplementary type attributes that carry TorchAO-style quantization
  metadata (granularity, block size, scale dtype) alongside ordinary
  tensor types.
- **Ops** -- ``QuantizePerTensorOp`` / ``DequantizePerTensorOp``,
  per-channel + per-group variants, ``WeightInt8PackMMOp`` and
  ``WeightInt4PackMMOp`` / ``WeightInt4PackQMOp``,
  ``ChooseQParamsPerTensorOp`` / ``ChooseQParamsPerChannelOp``, and
  ``FakeQuantOp`` (the QAT fake-quantize).

Register the dialect on a ``Context`` via::

    ctx.register_dialect("compgen.quant", lambda: Quant)
"""

from __future__ import annotations

from compgen.ir.quant.dialect import ALL_ATTRS, ALL_OPS, Quant
from compgen.ir.quant.ops import (
    ChooseQParamsPerChannelOp,
    ChooseQParamsPerTensorOp,
    DequantizePerChannelOp,
    DequantizePerGroupOp,
    DequantizePerTensorOp,
    FakeQuantOp,
    QuantizePerChannelOp,
    QuantizePerGroupOp,
    QuantizePerTensorOp,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)
from compgen.ir.quant.types import (
    AffineQuantizedTensorType,
    MXQuantizedTensorType,
    NVFP4TensorType,
    PackedIntTensorType,
)

__all__ = [
    "ALL_ATTRS",
    "ALL_OPS",
    "AffineQuantizedTensorType",
    "ChooseQParamsPerChannelOp",
    "ChooseQParamsPerTensorOp",
    "DequantizePerChannelOp",
    "DequantizePerGroupOp",
    "DequantizePerTensorOp",
    "FakeQuantOp",
    "MXQuantizedTensorType",
    "NVFP4TensorType",
    "PackedIntTensorType",
    "Quant",
    "QuantizePerChannelOp",
    "QuantizePerGroupOp",
    "QuantizePerTensorOp",
    "WeightInt4PackMMOp",
    "WeightInt4PackQMOp",
    "WeightInt8PackMMOp",
]
