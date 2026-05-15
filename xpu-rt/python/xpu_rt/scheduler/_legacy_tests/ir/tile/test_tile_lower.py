"""Tests for Tile IR lowering to Triton and Exo.

Covers representative ops lowered through both backends.
"""

from __future__ import annotations

from compgen.ir.tile.attrs import MemoryClassAttr, TileShapeAttr
from compgen.ir.tile.lower_exo import ExoLoweringResult, lower_tile_to_exo
from compgen.ir.tile.lower_triton import TritonLoweringResult, lower_tile_to_triton
from compgen.ir.tile.ops import (
    TileBarrierOp,
    TileElementwiseOp,
    TileLoadOp,
    TileMMAOp,
    TileReduceOp,
    TileStoreOp,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _make_load() -> TileLoadOp:
    return TileLoadOp.build(
        properties={
            "src_memref": SymbolRefAttr("A"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        }
    )


def _make_store() -> TileStoreOp:
    return TileStoreOp.build(
        properties={
            "dst_memref": SymbolRefAttr("C"),
            "fragment_ref": SymbolRefAttr("frag_C"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        }
    )


def _make_mma() -> TileMMAOp:
    return TileMMAOp.build(
        properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16, 16, 8]),
        }
    )


def _make_elementwise() -> TileElementwiseOp:
    return TileElementwiseOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("frag"),
            "op_kind": StringAttr("relu"),
            "shape": TileShapeAttr([128]),
        }
    )


def _make_reduce() -> TileReduceOp:
    return TileReduceOp.build(
        properties={
            "fragment_ref": SymbolRefAttr("x"),
            "reduce_kind": StringAttr("sum"),
            "axis": _i64(0),
            "shape": TileShapeAttr([128, 64]),
        }
    )


def _make_barrier() -> TileBarrierOp:
    return TileBarrierOp.build(
        properties={
            "scope": StringAttr("workgroup"),
        }
    )


# -- Triton lowering -----------------------------------------------------------


def test_triton_lower_load() -> None:
    """TileLoadOp lowers to tl.load."""
    result = lower_tile_to_triton([_make_load()])
    assert "tl.load" in result.kernel_code
    assert "A_frag" in result.kernel_code
    assert not result.diagnostics


def test_triton_lower_store() -> None:
    """TileStoreOp lowers to tl.store."""
    result = lower_tile_to_triton([_make_store()])
    assert "tl.store" in result.kernel_code
    assert "frag_C_frag" in result.kernel_code


def test_triton_lower_mma() -> None:
    """TileMMAOp lowers to tl.dot."""
    result = lower_tile_to_triton([_make_mma()])
    assert "tl.dot" in result.kernel_code
    assert "C_frag" in result.kernel_code


def test_triton_lower_elementwise() -> None:
    """TileElementwiseOp lowers to Triton elementwise code."""
    result = lower_tile_to_triton([_make_elementwise()])
    assert "tl.maximum" in result.kernel_code
    assert "frag_frag" in result.kernel_code


def test_triton_lower_reduce() -> None:
    """TileReduceOp lowers to tl.sum."""
    result = lower_tile_to_triton([_make_reduce()])
    assert "tl.sum" in result.kernel_code
    assert "x_reduced" in result.kernel_code


def test_triton_lower_empty() -> None:
    """Empty op list produces empty result."""
    result = lower_tile_to_triton([])
    assert result.kernel_code == ""
    assert not result.diagnostics


def test_triton_result_dataclass() -> None:
    """TritonLoweringResult has expected default fields."""
    result = TritonLoweringResult(kernel_code="test")
    assert result.kernel_code == "test"
    assert result.launch_config == {}
    assert result.diagnostics == []


# -- Exo lowering --------------------------------------------------------------


def test_exo_lower_mma() -> None:
    """TileMMAOp lowers to Exo loop nest."""
    result = lower_tile_to_exo([_make_mma()])
    assert "tile.mma" in result.proc_source
    assert "seq(0," in result.proc_source
    assert len(result.schedule_hints) > 0


def test_exo_lower_elementwise() -> None:
    """TileElementwiseOp lowers to Exo loop."""
    result = lower_tile_to_exo([_make_elementwise()])
    assert "relu" in result.proc_source
    assert "seq(0," in result.proc_source


def test_exo_lower_with_target_kit() -> None:
    """Exo lowering with non-generic target kit adds instruction mapping hints."""
    result = lower_tile_to_exo([_make_mma()], target_kit_name="neon_v8")
    assert any("neon_v8" in h for h in result.schedule_hints)


def test_exo_result_dataclass() -> None:
    """ExoLoweringResult has expected default fields."""
    result = ExoLoweringResult(proc_source="test")
    assert result.proc_source == "test"
    assert result.schedule_hints == []
    assert result.diagnostics == []
