"""Accelerator dialect operations.

Defines hardware-specific ops for custom accelerators. Each op models
a real hardware concept (not a 1:1 intrinsic mapping).

Invariants:
    - Ops have explicit memory effect annotations.
    - Ops have explicit shape and dtype contracts.
    - Async ops (DMA, matrix engine) have explicit start/wait semantics.

Two representations are provided:
    1. Frozen dataclass ops -- lightweight Python-side descriptors.
    2. xDSL IRDL operations -- for dialect registration and IR manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

# ---------------------------------------------------------------------------
# Frozen dataclass ops (lightweight descriptors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileLoadOp:
    """Load a tile from global memory to local/scratchpad memory.

    Attributes:
        src_memref: Source memory reference.
        dst_memref: Destination (local) memory reference.
        shape: Tile shape.
        dtype: Data type.
    """

    src_memref: str
    dst_memref: str
    shape: tuple[int, ...] = ()
    dtype: str = "float32"


@dataclass(frozen=True)
class TileStoreOp:
    """Store a tile from local memory to global memory.

    Attributes:
        src_memref: Source (local) memory reference.
        dst_memref: Destination memory reference.
        shape: Tile shape.
        dtype: Data type.
    """

    src_memref: str
    dst_memref: str
    shape: tuple[int, ...] = ()
    dtype: str = "float32"


@dataclass(frozen=True)
class DMAStartOp:
    """Start an asynchronous DMA transfer.

    Attributes:
        src: Source address/memref.
        dst: Destination address/memref.
        size_bytes: Transfer size.
        event: Event/token for synchronization.
    """

    src: str
    dst: str
    size_bytes: int = 0
    event: str = ""


@dataclass(frozen=True)
class DMAWaitOp:
    """Wait for a DMA transfer to complete.

    Attributes:
        event: Event/token to wait on.
    """

    event: str


@dataclass(frozen=True)
class MatrixEngineOp:
    """Launch a matrix/tensor engine computation.

    Attributes:
        op_kind: Operation kind ("matmul", "conv", "mma", "outer_product").
        a_ref: Operand A reference.
        b_ref: Operand B reference.
        c_ref: Accumulator/output reference.
        config: Engine-specific configuration.
    """

    op_kind: str
    a_ref: str
    b_ref: str
    c_ref: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarrierOp:
    """Synchronization barrier.

    Attributes:
        scope: Barrier scope ("workgroup", "device", "system").
    """

    scope: str = "workgroup"


# ---------------------------------------------------------------------------
# xDSL IRDL operations
# ---------------------------------------------------------------------------


@irdl_op_definition
class AccelTileLoadIROp(IRDLOperation):
    """Load a tile from global memory to local/scratchpad memory (xDSL op)."""

    name = "compgen.accel.tile_load"

    src_memref = prop_def(StringAttr)
    dst_memref = prop_def(StringAttr)
    shape = opt_prop_def(ArrayAttr)
    dtype = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class AccelTileStoreIROp(IRDLOperation):
    """Store a tile from local memory to global memory (xDSL op)."""

    name = "compgen.accel.tile_store"

    src_memref = prop_def(StringAttr)
    dst_memref = prop_def(StringAttr)
    shape = opt_prop_def(ArrayAttr)
    dtype = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class AccelDMAStartIROp(IRDLOperation):
    """Start an asynchronous DMA transfer (xDSL op)."""

    name = "compgen.accel.dma_start"

    src = prop_def(StringAttr)
    dst = prop_def(StringAttr)
    size_bytes = prop_def(IntegerAttr)
    event = prop_def(StringAttr)


@irdl_op_definition
class AccelDMAWaitIROp(IRDLOperation):
    """Wait for a DMA transfer to complete (xDSL op)."""

    name = "compgen.accel.dma_wait"

    event = prop_def(StringAttr)


@irdl_op_definition
class AccelMatrixEngineIROp(IRDLOperation):
    """Launch a matrix/tensor engine computation (xDSL op)."""

    name = "compgen.accel.matrix_engine"

    op_kind = prop_def(StringAttr)
    a_ref = prop_def(StringAttr)
    b_ref = prop_def(StringAttr)
    c_ref = prop_def(StringAttr)

    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "matmul",
            "conv",
            "mma",
            "outer_product",
        }
    )

    def verify_(self) -> None:
        if self.op_kind.data not in self._VALID_KINDS:
            raise VerifyException(
                f"Invalid matrix engine op_kind '{self.op_kind.data}', expected one of {sorted(self._VALID_KINDS)}"
            )


@irdl_op_definition
class AccelBarrierIROp(IRDLOperation):
    """Synchronization barrier (xDSL op)."""

    name = "compgen.accel.barrier"

    scope = opt_prop_def(StringAttr)

    _VALID_SCOPES: ClassVar[frozenset[str]] = frozenset({"workgroup", "device", "system"})

    def verify_(self) -> None:
        if self.scope is not None and self.scope.data not in self._VALID_SCOPES:
            raise VerifyException(
                f"Invalid barrier scope '{self.scope.data}', expected one of {sorted(self._VALID_SCOPES)}"
            )


# ---------------------------------------------------------------------------
# Wave 9: HMX tile primitives (hexagon-mlir-inspired)
# ---------------------------------------------------------------------------


@irdl_op_definition
class HMXTileLoadIROp(IRDLOperation):
    """Load a 32x32 (or configurable) tile from DRAM to VTCM with format xform.

    Mirrors hexagon's ``micro_hmx_copy_submatrix_to_f16`` pattern: load
    a submatrix + apply a row-major-to-activation-horizontal format
    transform in a single op.
    """

    name = "compgen.accel.hmx_tile_load"

    src_memref = prop_def(StringAttr)
    dst_memref = prop_def(StringAttr)
    tile_shape = prop_def(ArrayAttr)
    format_xform = prop_def(StringAttr)
    dtype = prop_def(StringAttr)

    _VALID_XFORMS: ClassVar[frozenset[str]] = frozenset(
        {
            "rm_to_ah",
            "rm_to_av",
            "identity",
        }
    )

    def verify_(self) -> None:
        if self.format_xform.data not in self._VALID_XFORMS:
            raise VerifyException(
                f"Invalid format_xform '{self.format_xform.data}', expected one of {sorted(self._VALID_XFORMS)}"
            )
        for d in self.tile_shape.data:
            if isinstance(d, IntegerAttr) and d.value.data <= 0:
                raise VerifyException(f"hmx_tile_load tile_shape entries must be positive, got {d.value.data}")


@irdl_op_definition
class HMXMatrixEngineIROp(IRDLOperation):
    """Invoke the HMX matrix engine: ``C = op_kind(A, B, C_init)`` on tiles."""

    name = "compgen.accel.hmx_matrix_engine"

    a_tile = prop_def(StringAttr)
    b_tile = prop_def(StringAttr)
    c_tile = prop_def(StringAttr)
    op_kind = prop_def(StringAttr)
    shape = prop_def(ArrayAttr)
    dtype = prop_def(StringAttr)
    accumulate = opt_prop_def(StringAttr)

    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "matmul",
            "matmul_accumulate",
            "outer_product",
        }
    )

    def verify_(self) -> None:
        if self.op_kind.data not in self._VALID_KINDS:
            raise VerifyException(
                f"Invalid hmx_matrix_engine op_kind '{self.op_kind.data}', expected one of {sorted(self._VALID_KINDS)}"
            )
        dims = [int(d.value.data) for d in self.shape.data if isinstance(d, IntegerAttr)]
        if len(dims) != 3:
            raise VerifyException(f"hmx_matrix_engine shape must have 3 entries [M, N, K], got {len(dims)}")
        if any(d <= 0 for d in dims):
            raise VerifyException(f"hmx_matrix_engine shape entries must be positive, got {dims}")


@irdl_op_definition
class HMXAccumulatorClearIROp(IRDLOperation):
    """Clear an HMX accumulator tile to zero."""

    name = "compgen.accel.hmx_accumulator_clear"

    c_tile = prop_def(StringAttr)
    dtype = prop_def(StringAttr)
    shape = prop_def(ArrayAttr)


@irdl_op_definition
class HMXDMAOverlapIROp(IRDLOperation):
    """Pipeline marker for the hexagon-mlir double-buffer (S1/S2) lowering."""

    name = "compgen.accel.hmx_dma_overlap"

    producer_tile = prop_def(StringAttr)
    consumer_tile = prop_def(StringAttr)
    line_bytes = prop_def(IntegerAttr)
    depth = prop_def(IntegerAttr)

    def verify_(self) -> None:
        if self.line_bytes.value.data <= 0:
            raise VerifyException(f"hmx_dma_overlap line_bytes must be positive, got {self.line_bytes.value.data}")
        if self.depth.value.data < 2:
            raise VerifyException(f"hmx_dma_overlap depth must be >= 2, got {self.depth.value.data}")


ACCEL_IR_OPS: list[type[IRDLOperation]] = [
    AccelTileLoadIROp,
    AccelTileStoreIROp,
    AccelDMAStartIROp,
    AccelDMAWaitIROp,
    AccelMatrixEngineIROp,
    AccelBarrierIROp,
    HMXTileLoadIROp,
    HMXMatrixEngineIROp,
    HMXAccumulatorClearIROp,
    HMXDMAOverlapIROp,
]
"""All xDSL IRDL operations in the accelerator dialect."""

__all__ = [
    "AccelBarrierIROp",
    "AccelDMAStartIROp",
    "AccelDMAWaitIROp",
    "AccelMatrixEngineIROp",
    "AccelTileLoadIROp",
    "AccelTileStoreIROp",
    "HMXTileLoadIROp",
    "HMXMatrixEngineIROp",
    "HMXAccumulatorClearIROp",
    "HMXDMAOverlapIROp",
    "ACCEL_IR_OPS",
    "BarrierOp",
    "DMAStartOp",
    "DMAWaitOp",
    "MatrixEngineOp",
    "TileLoadOp",
    "TileStoreOp",
]
