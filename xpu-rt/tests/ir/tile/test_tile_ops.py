"""Tests for Tile IR operations.

Covers all 7 ops: construction, property access, and verify_()
positive/negative cases.
"""

from __future__ import annotations

import pytest
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
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr
from xdsl.utils.exceptions import VerifyException


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


# -- TileLoadOp ----------------------------------------------------------------


def test_tile_load_build_minimal() -> None:
    """TileLoadOp can be built with required properties only."""
    op = TileLoadOp.build(
        properties={
            "src_memref": SymbolRefAttr("buf_A"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        }
    )
    assert op.src_memref.root_reference.data == "buf_A"
    assert op.memory_class.kind.data == "global"
    assert op.layout is None
    assert op.is_async is None


def test_tile_load_with_layout() -> None:
    """TileLoadOp can include an optional layout."""
    op = TileLoadOp.build(
        properties={
            "src_memref": SymbolRefAttr("buf_A"),
            "memory_class": MemoryClassAttr("shared"),
            "shape": TileShapeAttr([32, 32]),
            "layout": FragmentLayoutAttr("row_major"),
        }
    )
    assert op.layout.layout.data == "row_major"


def test_tile_load_with_async() -> None:
    """TileLoadOp can be marked async."""
    op = TileLoadOp.build(
        properties={
            "src_memref": SymbolRefAttr("buf_A"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
            "is_async": _i64(1),
        }
    )
    assert op.is_async.value.data == 1


def test_tile_load_name() -> None:
    assert TileLoadOp.name == "tile.load"


# -- TileStoreOp ---------------------------------------------------------------


def test_tile_store_build() -> None:
    """TileStoreOp can be built with required properties."""
    op = TileStoreOp.build(
        properties={
            "dst_memref": SymbolRefAttr("buf_C"),
            "fragment_ref": SymbolRefAttr("frag_C"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        }
    )
    assert op.dst_memref.root_reference.data == "buf_C"
    assert op.fragment_ref.root_reference.data == "frag_C"


def test_tile_store_name() -> None:
    assert TileStoreOp.name == "tile.store"


# -- TileMMAOp ----------------------------------------------------------------


def test_tile_mma_build() -> None:
    """TileMMAOp can be built with A, B, C refs and shape."""
    op = TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("frag_A"),
            "b_ref": SymbolRefAttr("frag_B"),
            "c_ref": SymbolRefAttr("frag_C"),
            "shape": TileShapeAttr([16, 16, 8]),
        }
    )
    assert op.a_ref.root_reference.data == "frag_A"
    assert op.c_ref.root_reference.data == "frag_C"


def test_tile_mma_verify_ok() -> None:
    """TileMMAOp verifies with valid M, N, K dimensions."""
    op = TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16, 16, 8]),
        }
    )
    op.verify()  # should not raise


def test_tile_mma_verify_2d_ok() -> None:
    """TileMMAOp verifies with 2D shape (M, N only)."""
    op = TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16, 16]),
        }
    )
    op.verify()  # should not raise


def test_tile_mma_verify_1d_fails() -> None:
    """TileMMAOp rejects shape with fewer than 2 dimensions."""
    op = TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16]),
        }
    )
    with pytest.raises(VerifyException, match="at least M, N"):
        op.verify()


def test_tile_mma_verify_zero_dim_fails() -> None:
    """TileMMAOp rejects zero dimensions."""
    op = TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16, 0]),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_tile_mma_name() -> None:
    assert TileMMAOp.name == "tile.mma"


# -- TileElementwiseOp --------------------------------------------------------


def test_tile_elementwise_build() -> None:
    """TileElementwiseOp can be built with a valid op kind."""
    op = TileElementwiseOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("frag"),
            "op_kind": StringAttr("relu"),
            "shape": TileShapeAttr([128]),
        }
    )
    assert op.op_kind.data == "relu"


def test_tile_elementwise_verify_ok() -> None:
    """TileElementwiseOp verifies for all valid op kinds."""
    for kind in ("relu", "gelu", "sigmoid", "tanh", "add", "mul", "exp", "neg"):
        op = TileElementwiseOp.build(
            properties={
                "fragment_ref": SymbolRefAttr("frag"),
                "op_kind": StringAttr(kind),
                "shape": TileShapeAttr([64]),
            }
        )
        op.verify()


def test_tile_elementwise_verify_invalid_fails() -> None:
    """TileElementwiseOp rejects invalid op kinds."""
    op = TileElementwiseOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("frag"),
            "op_kind": StringAttr("invalid_op"),
            "shape": TileShapeAttr([64]),
        }
    )
    with pytest.raises(VerifyException, match="Invalid elementwise op"):
        op.verify()


def test_tile_elementwise_name() -> None:
    assert TileElementwiseOp.name == "tile.elementwise"


# -- TileReduceOp -------------------------------------------------------------


def test_tile_reduce_build() -> None:
    """TileReduceOp can be built with valid properties."""
    op = TileReduceOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("frag"),
            "reduce_kind": StringAttr("sum"),
            "axis": _i64(0),
            "shape": TileShapeAttr([128, 64]),
        }
    )
    assert op.reduce_kind.data == "sum"
    assert op.axis.value.data == 0


def test_tile_reduce_verify_ok() -> None:
    """TileReduceOp verifies for all valid reduce kinds."""
    for kind in ("sum", "max", "min", "mean"):
        op = TileReduceOp.build(
            properties={
                "fragment_ref": SymbolRefAttr("frag"),
                "reduce_kind": StringAttr(kind),
                "axis": _i64(0),
                "shape": TileShapeAttr([64]),
            }
        )
        op.verify()


def test_tile_reduce_verify_invalid_fails() -> None:
    """TileReduceOp rejects invalid reduce kinds."""
    op = TileReduceOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("frag"),
            "reduce_kind": StringAttr("product"),
            "axis": _i64(0),
            "shape": TileShapeAttr([64]),
        }
    )
    with pytest.raises(VerifyException, match="Invalid reduce kind"):
        op.verify()


# -- TileBarrierOp ------------------------------------------------------------


def test_tile_barrier_build() -> None:
    """TileBarrierOp can be built with a valid scope."""
    op = TileBarrierOp.build(
        properties={
            "scope": StringAttr("workgroup"),
        }
    )
    assert op.scope.data == "workgroup"


def test_tile_barrier_verify_ok() -> None:
    """TileBarrierOp verifies for all valid scopes."""
    for scope in ("workgroup", "device", "system"):
        op = TileBarrierOp.build(
            properties={
                "scope": StringAttr(scope),
            }
        )
        op.verify()


def test_tile_barrier_verify_invalid_fails() -> None:
    """TileBarrierOp rejects invalid scopes."""
    op = TileBarrierOp.build(
        properties={
            "scope": StringAttr("invalid"),
        }
    )
    with pytest.raises(VerifyException, match="Invalid barrier scope"):
        op.verify()


# -- TileAsyncCopyOp ----------------------------------------------------------


def test_tile_async_copy_build() -> None:
    """TileAsyncCopyOp can be built with required properties."""
    op = TileAsyncCopyOp.build(
        properties={
            "src_ref": SymbolRefAttr("buf_A"),
            "dst_ref": SymbolRefAttr("smem_A"),
            "src_memory_class": MemoryClassAttr("global"),
            "dst_memory_class": MemoryClassAttr("shared"),
            "shape": TileShapeAttr([32, 32]),
        }
    )
    assert op.src_ref.root_reference.data == "buf_A"
    assert op.dst_ref.root_reference.data == "smem_A"
    assert op.src_memory_class.kind.data == "global"
    assert op.dst_memory_class.kind.data == "shared"


def test_tile_async_copy_name() -> None:
    assert TileAsyncCopyOp.name == "tile.async_copy"
