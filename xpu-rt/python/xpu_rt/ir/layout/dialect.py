"""Layout IR dialect registration.

Registers all Layout IR operations and attributes with xDSL.
"""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr
from compgen.ir.layout.ops import (
    PackOp,
    SetLayoutOp,
    UnpackOp,
    UnsetLayoutOp,
)

ALL_OPS = [
    SetLayoutOp,
    UnsetLayoutOp,
    PackOp,
    UnpackOp,
]

ALL_ATTRS = [
    LayoutEncodingAttr,
    PackSpecAttr,
]

Layout = Dialect("layout", ALL_OPS, ALL_ATTRS)
"""The Layout IR dialect -- register with ``ctx.register_dialect("layout", lambda: Layout)``."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Layout"]
