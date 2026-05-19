"""Registration of the ``xpu_rt.linalg_ext`` dialect."""

from __future__ import annotations

from xdsl.ir import Dialect

from xpu_rt.ir.linalg_ext.ops import LINALG_EXT_OPS

ALL_OPS = list(LINALG_EXT_OPS)
ALL_ATTRS: list = []

LinalgExt = Dialect("xpu_rt.linalg_ext", ALL_OPS, ALL_ATTRS)
"""Register on a ``Context`` with
``ctx.register_dialect('xpu_rt.linalg_ext', lambda: LinalgExt)``."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "LinalgExt"]
