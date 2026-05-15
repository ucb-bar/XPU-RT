"""Registration of the ``xpu_rt.tensor_ext`` dialect."""

from __future__ import annotations

from xdsl.ir import Dialect

from xpu_rt.ir.tensor_ext.ops import TENSOR_EXT_OPS

ALL_OPS = list(TENSOR_EXT_OPS)
ALL_ATTRS: list = []

TensorExt = Dialect("xpu_rt.tensor_ext", ALL_OPS, ALL_ATTRS)
"""The tensor-ext dialect.

Register with ``ctx.register_dialect("xpu_rt.tensor_ext", lambda: TensorExt)``.
"""


__all__ = ["ALL_ATTRS", "ALL_OPS", "TensorExt"]
