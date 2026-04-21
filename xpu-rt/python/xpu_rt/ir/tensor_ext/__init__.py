"""``compgen.tensor_ext`` -- supplement to xDSL's tensor dialect.

xDSL's ``tensor`` dialect is missing three ops we need for +
reconstruction: ``concat``, ``pack``, ``unpack``. MLIR upstream ships
them (``tensor.concat``, ``tensor.pack``, ``tensor.unpack``); this
dialect mirrors their semantics so the  ``decompose_concat``
and  ``normalize_subbyte`` passes have a real destination.

Register on a ``Context`` with::

    ctx.register_dialect("compgen.tensor_ext", lambda: TensorExt)
"""

from __future__ import annotations

from compgen.ir.tensor_ext.dialect import ALL_ATTRS, ALL_OPS, TensorExt
from compgen.ir.tensor_ext.ops import ConcatOp, PackOp, UnpackOp

__all__ = [
    "ALL_ATTRS",
    "ALL_OPS",
    "ConcatOp",
    "PackOp",
    "TensorExt",
    "UnpackOp",
]
