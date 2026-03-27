"""Tile IR operations.

Seven operations representing a backend-plural tile virtual ISA:
    - TileLoadOp: load tile from memory to fragment
    - TileStoreOp: store fragment to memory
    - TileMMAOp: matrix multiply-accumulate
    - TileElementwiseOp: elementwise operation on fragment
    - TileReduceOp: reduction along an axis
    - TileBarrierOp: synchronization barrier
    - TileAsyncCopyOp: asynchronous memory copy
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

# Reuse ProvenanceAttr from Recipe IR for lineage tracking
from compgen.ir.recipe.attrs import ProvenanceAttr
from compgen.ir.tile.attrs import FragmentLayoutAttr, MemoryClassAttr, TileShapeAttr


@irdl_op_definition
class TileLoadOp(IRDLOperation):
    """Load a tile from memory into a fragment."""

    name = "tile.load"

    src_memref = prop_def(SymbolRefAttr)
    memory_class = prop_def(MemoryClassAttr)
    shape = prop_def(TileShapeAttr)
    layout = opt_prop_def(FragmentLayoutAttr)
    is_async = opt_prop_def(IntegerAttr)  # 0 or 1
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class TileStoreOp(IRDLOperation):
    """Store a fragment back to memory."""

    name = "tile.store"

    dst_memref = prop_def(SymbolRefAttr)
    fragment_ref = prop_def(SymbolRefAttr)
    memory_class = prop_def(MemoryClassAttr)
    shape = prop_def(TileShapeAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class TileMMAOp(IRDLOperation):
    """Matrix multiply-accumulate on tile fragments.

    C += A @ B, where shape specifies [M, N, K].
    """

    name = "tile.mma"

    a_ref = prop_def(SymbolRefAttr)
    b_ref = prop_def(SymbolRefAttr)
    c_ref = prop_def(SymbolRefAttr)
    shape = prop_def(TileShapeAttr)  # [M, N, K]
    dtype = opt_prop_def(StringAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        dims = [a.value.data for a in self.shape.dims.data if isinstance(a, IntegerAttr)]
        if len(dims) < 2:
            raise VerifyException("tile.mma shape must have at least M, N dimensions")
        if any(d <= 0 for d in dims):
            raise VerifyException("tile.mma shape dimensions must be positive")


@irdl_op_definition
class TileElementwiseOp(IRDLOperation):
    """Apply an elementwise operation on a tile fragment.

    Supported ops: relu, gelu, sigmoid, tanh, add, mul, sub, div,
                   exp, log, sqrt, rsqrt, abs, neg, max, min.
    """

    name = "tile.elementwise"

    fragment_ref = prop_def(SymbolRefAttr)
    op_kind = prop_def(StringAttr)
    shape = prop_def(TileShapeAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    _VALID_OPS: ClassVar[frozenset[str]] = frozenset({
        "relu", "gelu", "sigmoid", "tanh", "add", "mul", "sub", "div",
        "exp", "log", "sqrt", "rsqrt", "abs", "neg", "max", "min",
    })

    def verify_(self) -> None:
        if self.op_kind.data not in self._VALID_OPS:
            raise VerifyException(
                f"Invalid elementwise op '{self.op_kind.data}', "
                f"expected one of {sorted(self._VALID_OPS)}"
            )


@irdl_op_definition
class TileReduceOp(IRDLOperation):
    """Reduce a tile fragment along an axis.

    Supported kinds: sum, max, min, mean.
    """

    name = "tile.reduce"

    fragment_ref = prop_def(SymbolRefAttr)
    reduce_kind = prop_def(StringAttr)
    axis = prop_def(IntegerAttr)
    shape = prop_def(TileShapeAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset({"sum", "max", "min", "mean"})

    def verify_(self) -> None:
        if self.reduce_kind.data not in self._VALID_KINDS:
            raise VerifyException(
                f"Invalid reduce kind '{self.reduce_kind.data}', "
                f"expected one of {sorted(self._VALID_KINDS)}"
            )


@irdl_op_definition
class TileBarrierOp(IRDLOperation):
    """Synchronization barrier at tile level.

    Scopes: workgroup, device, system.
    """

    name = "tile.barrier"

    scope = prop_def(StringAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    _VALID_SCOPES: ClassVar[frozenset[str]] = frozenset({"workgroup", "device", "system"})

    def verify_(self) -> None:
        if self.scope.data not in self._VALID_SCOPES:
            raise VerifyException(
                f"Invalid barrier scope '{self.scope.data}', "
                f"expected one of {sorted(self._VALID_SCOPES)}"
            )


@irdl_op_definition
class TileAsyncCopyOp(IRDLOperation):
    """Asynchronous memory copy between memory classes."""

    name = "tile.async_copy"

    src_ref = prop_def(SymbolRefAttr)
    dst_ref = prop_def(SymbolRefAttr)
    src_memory_class = prop_def(MemoryClassAttr)
    dst_memory_class = prop_def(MemoryClassAttr)
    shape = prop_def(TileShapeAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


__all__ = [
    "TileAsyncCopyOp",
    "TileBarrierOp",
    "TileElementwiseOp",
    "TileLoadOp",
    "TileMMAOp",
    "TileReduceOp",
    "TileStoreOp",
]
