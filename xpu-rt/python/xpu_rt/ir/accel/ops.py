"""Accelerator dialect operations.

Defines hardware-specific ops for custom accelerators. Each op models
a real hardware concept (not a 1:1 intrinsic mapping).

Invariants:
    - Ops have explicit memory effect annotations.
    - Ops have explicit shape and dtype contracts.
    - Async ops (DMA, matrix engine) have explicit start/wait semantics.

TODO: Implement as xDSL Operation subclasses.
TODO: Add verifier rules per op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


__all__ = ["BarrierOp", "DMAStartOp", "DMAWaitOp", "MatrixEngineOp", "TileLoadOp", "TileStoreOp"]
