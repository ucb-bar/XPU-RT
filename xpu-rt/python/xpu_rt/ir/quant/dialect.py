"""Registration for the ``compgen.quant`` dialect."""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.quant.ops import QUANT_OPS
from compgen.ir.quant.types import (
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

Quant = Dialect("compgen.quant", ALL_OPS, ALL_ATTRS)
"""The quantization dialect.

Register on a ``Context`` with::

    ctx.register_dialect("compgen.quant", lambda: Quant)
"""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Quant"]
