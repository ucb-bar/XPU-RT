"""Registration for the ``xpu_rt.quant`` dialect."""

from __future__ import annotations

from xdsl.ir import Dialect

from xpu_rt.ir.quant.ops import QUANT_OPS
from xpu_rt.ir.quant.types import (
    AffineQuantizedTensorType,
    MXQuantizedTensorType,
    NVFP4TensorType,
    PackedIntTensorType,
)

ALL_OPS = list(QUANT_OPS)

ALL_ATTRS = [
    AffineQuantizedTensorType,
    PackedIntTensorType,
    MXQuantizedTensorType,
    NVFP4TensorType,
]

Quant = Dialect("xpu_rt.quant", ALL_OPS, ALL_ATTRS)
"""The quantization dialect.

Register on a ``Context`` with::

    ctx.register_dialect("xpu_rt.quant", lambda: Quant)
"""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Quant"]
