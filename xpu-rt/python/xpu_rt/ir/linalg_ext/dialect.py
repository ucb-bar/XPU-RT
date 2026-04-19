"""Registration of the ``compgen.linalg_ext`` dialect."""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.linalg_ext.ops import LINALG_EXT_OPS

ALL_OPS = list(LINALG_EXT_OPS)
ALL_ATTRS: list = []

LinalgExt = Dialect("compgen.linalg_ext", ALL_OPS, ALL_ATTRS)
"""Register on a ``Context`` with
``ctx.register_dialect('compgen.linalg_ext', lambda: LinalgExt)``."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "LinalgExt"]
