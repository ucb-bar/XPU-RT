"""Triton operation → target semantic mapping.

Defines the universal mapping between Triton's tile-level operations and
target-independent compute semantics.  Each target backend provides its
own hardware-specific realization of these semantics.

This is the bridge that makes Triton a **portable kernel language**:
write once in Triton, compile to any target via the semantic mapping.

Inspired by Hexagon-MLIR's approach where Triton ops flow through
``triton-to-linalg`` and then to target-specific lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ComputeUnit(str, Enum):
    """Abstract compute unit classification (target-independent)."""

    MATRIX = "matrix"        # Tiled matrix multiply (MXU, tensor core, HMX)
    VECTOR = "vector"        # Vector/SIMD elementwise (VPU, HVX, NEON)
    SCALAR = "scalar"        # Scalar ALU
    REDUCTION = "reduction"  # Reduction hardware (tree, shuffle)
    MEMORY = "memory"        # Load/store/DMA
    CONTROL = "control"      # Control flow, synchronization


class MemoryLevel(str, Enum):
    """Abstract memory hierarchy level (target-independent)."""

    REGISTER = "register"        # Register file / accumulator
    SCRATCHPAD = "scratchpad"    # On-chip fast memory (VTCM, VMEM, shared memory)
    GLOBAL = "global"            # Main memory (DDR, HBM)


@dataclass(frozen=True)
class TritonOpSemantic:
    """Semantic description of a Triton operation.

    Describes what a Triton op does in target-independent terms.
    Each target backend maps these semantics to its own hardware ops.

    Attributes:
        triton_op: Triton API name (``"tl.load"``, ``"tl.dot"``, etc.).
        semantic: Abstract operation type.
        compute_unit: Which type of compute unit handles this.
        memory_level: Which memory level is involved (for load/store).
        description: Human-readable description.
        linalg_equivalent: What this maps to in linalg IR.
    """

    triton_op: str
    semantic: str
    compute_unit: ComputeUnit
    memory_level: MemoryLevel | None = None
    description: str = ""
    linalg_equivalent: str = ""


# ---------------------------------------------------------------------------
# Universal Triton op → semantic mapping
# ---------------------------------------------------------------------------

TRITON_OP_SEMANTICS: list[TritonOpSemantic] = [
    # --- Memory operations ---
    TritonOpSemantic(
        triton_op="tl.load",
        semantic="tile_load",
        compute_unit=ComputeUnit.MEMORY,
        memory_level=MemoryLevel.GLOBAL,
        description="Load a tile of data from global memory into registers",
        linalg_equivalent="memref.load / tensor.extract_slice",
    ),
    TritonOpSemantic(
        triton_op="tl.store",
        semantic="tile_store",
        compute_unit=ComputeUnit.MEMORY,
        memory_level=MemoryLevel.GLOBAL,
        description="Store a tile of data from registers to global memory",
        linalg_equivalent="memref.store / tensor.insert_slice",
    ),

    # --- Matrix operations ---
    TritonOpSemantic(
        triton_op="tl.dot",
        semantic="tile_matmul",
        compute_unit=ComputeUnit.MATRIX,
        description="Tile-level matrix multiply-accumulate",
        linalg_equivalent="linalg.matmul / linalg.batch_matmul",
    ),

    # --- Elementwise operations ---
    TritonOpSemantic(
        triton_op="tl.exp",
        semantic="elementwise_exp",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise exponential",
        linalg_equivalent="math.exp (in linalg.generic)",
    ),
    TritonOpSemantic(
        triton_op="tl.log2",
        semantic="elementwise_log2",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise log base 2",
        linalg_equivalent="math.log2 (in linalg.generic)",
    ),
    TritonOpSemantic(
        triton_op="tl.sqrt",
        semantic="elementwise_sqrt",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise square root",
        linalg_equivalent="math.sqrt (in linalg.generic)",
    ),
    TritonOpSemantic(
        triton_op="tl.sigmoid",
        semantic="elementwise_sigmoid",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise sigmoid activation",
        linalg_equivalent="linalg.generic with sigmoid body",
    ),
    TritonOpSemantic(
        triton_op="tl.where",
        semantic="elementwise_select",
        compute_unit=ComputeUnit.VECTOR,
        description="Conditional element selection",
        linalg_equivalent="arith.select (in linalg.generic)",
    ),
    TritonOpSemantic(
        triton_op="tl.maximum",
        semantic="elementwise_max",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise maximum",
        linalg_equivalent="arith.maximumf (in linalg.generic)",
    ),
    TritonOpSemantic(
        triton_op="tl.abs",
        semantic="elementwise_abs",
        compute_unit=ComputeUnit.VECTOR,
        description="Element-wise absolute value",
        linalg_equivalent="math.absf (in linalg.generic)",
    ),

    # --- Reduction operations ---
    TritonOpSemantic(
        triton_op="tl.sum",
        semantic="reduction_sum",
        compute_unit=ComputeUnit.REDUCTION,
        description="Sum reduction along an axis",
        linalg_equivalent="linalg.generic with add accumulation",
    ),
    TritonOpSemantic(
        triton_op="tl.max",
        semantic="reduction_max",
        compute_unit=ComputeUnit.REDUCTION,
        description="Max reduction along an axis",
        linalg_equivalent="linalg.generic with max accumulation",
    ),
    TritonOpSemantic(
        triton_op="tl.min",
        semantic="reduction_min",
        compute_unit=ComputeUnit.REDUCTION,
        description="Min reduction along an axis",
        linalg_equivalent="linalg.generic with min accumulation",
    ),

    # --- Initialization ---
    TritonOpSemantic(
        triton_op="tl.zeros",
        semantic="tile_init_zeros",
        compute_unit=ComputeUnit.MEMORY,
        memory_level=MemoryLevel.REGISTER,
        description="Initialize a tile with zeros",
        linalg_equivalent="linalg.fill with 0",
    ),
    TritonOpSemantic(
        triton_op="tl.full",
        semantic="tile_init_full",
        compute_unit=ComputeUnit.MEMORY,
        memory_level=MemoryLevel.REGISTER,
        description="Initialize a tile with a constant",
        linalg_equivalent="linalg.fill",
    ),

    # --- Control flow ---
    TritonOpSemantic(
        triton_op="tl.program_id",
        semantic="grid_index",
        compute_unit=ComputeUnit.CONTROL,
        description="Get the current program/block/tile index",
        linalg_equivalent="N/A (maps to loop iteration variable)",
    ),
    TritonOpSemantic(
        triton_op="tl.arange",
        semantic="index_range",
        compute_unit=ComputeUnit.SCALAR,
        description="Generate sequential indices within a block",
        linalg_equivalent="affine.for / scf.for indices",
    ),
    TritonOpSemantic(
        triton_op="tl.num_programs",
        semantic="grid_size",
        compute_unit=ComputeUnit.CONTROL,
        description="Get the total number of programs/blocks",
        linalg_equivalent="N/A (maps to loop bound)",
    ),

    # --- Type conversion ---
    TritonOpSemantic(
        triton_op="tl.cast",
        semantic="type_convert",
        compute_unit=ComputeUnit.VECTOR,
        description="Convert between data types",
        linalg_equivalent="arith.truncf / arith.extf / arith.fptosi",
    ),

    # --- Atomic operations ---
    TritonOpSemantic(
        triton_op="tl.atomic_add",
        semantic="atomic_add",
        compute_unit=ComputeUnit.MEMORY,
        memory_level=MemoryLevel.GLOBAL,
        description="Atomic addition to global memory",
        linalg_equivalent="memref.atomic_rmw",
    ),
]

# Build lookup dict
_SEMANTIC_BY_OP: dict[str, TritonOpSemantic] = {
    s.triton_op: s for s in TRITON_OP_SEMANTICS
}


def get_triton_semantic(triton_op: str) -> TritonOpSemantic | None:
    """Look up the semantic description for a Triton operation.

    Args:
        triton_op: Triton API name (e.g., ``"tl.dot"``).

    Returns:
        Semantic description, or None if unknown.
    """
    return _SEMANTIC_BY_OP.get(triton_op)


def get_semantics_by_compute_unit(unit: ComputeUnit) -> list[TritonOpSemantic]:
    """Get all Triton ops that use a given compute unit type."""
    return [s for s in TRITON_OP_SEMANTICS if s.compute_unit == unit]


__all__ = [
    "ComputeUnit",
    "MemoryLevel",
    "TRITON_OP_SEMANTICS",
    "TritonOpSemantic",
    "get_semantics_by_compute_unit",
    "get_triton_semantic",
]
