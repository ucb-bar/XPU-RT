"""Tile IR dialect registration.

Registers all Tile IR operations and attributes with xDSL.
"""

from __future__ import annotations

from xdsl.ir import Dialect

from compgen.ir.tile.attrs import FragmentLayoutAttr, MemoryClassAttr, TileShapeAttr
from compgen.ir.tile.ops import (
    TileAsyncCopyOp,
    TileBarrierOp,
    TileElementwiseOp,
    TileLoadOp,
    TileMMAOp,
    TileReduceOp,
    TileStoreOp,
)

ALL_OPS = [
    TileLoadOp,
    TileStoreOp,
    TileMMAOp,
    TileElementwiseOp,
    TileReduceOp,
    TileBarrierOp,
    TileAsyncCopyOp,
]

ALL_ATTRS = [
    MemoryClassAttr,
    FragmentLayoutAttr,
    TileShapeAttr,
]

Tile = Dialect("tile", ALL_OPS, ALL_ATTRS)
"""The Tile IR dialect -- register with ``ctx.register_dialect("tile", lambda: Tile)``."""


__all__ = ["ALL_ATTRS", "ALL_OPS", "Tile"]
