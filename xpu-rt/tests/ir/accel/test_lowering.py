"""Tests for accelerator dialect lowering."""

from __future__ import annotations

from compgen.ir.accel.lowering import LoweringOutput, lower_accel_to_llvm
from compgen.ir.accel.ops import (
    BarrierOp,
    DMAStartOp,
    DMAWaitOp,
    MatrixEngineOp,
    TileLoadOp,
    TileStoreOp,
)


def test_lower_accel_to_llvm_exists() -> None:
    assert callable(lower_accel_to_llvm)


# -- Single op lowering ------------------------------------------------------


def test_lower_tile_load() -> None:
    """TileLoadOp should lower to a memcpy descriptor."""
    op = TileLoadOp(src_memref="global", dst_memref="local", shape=(128, 128), dtype="float16")
    result = lower_accel_to_llvm(op)
    assert isinstance(result, LoweringOutput)
    assert len(result.lowered_ops) == 1
    lowered = result.lowered_ops[0]
    assert lowered["type"] == "memcpy"
    assert lowered["src"] == "global"
    assert lowered["dst"] == "local"
    assert lowered["shape"] == [128, 128]
    assert lowered["dtype"] == "float16"


def test_lower_tile_store() -> None:
    """TileStoreOp should lower to a memcpy descriptor."""
    op = TileStoreOp(src_memref="local", dst_memref="global", shape=(64,), dtype="float32")
    result = lower_accel_to_llvm(op)
    assert len(result.lowered_ops) == 1
    assert result.lowered_ops[0]["type"] == "memcpy"
    assert result.lowered_ops[0]["src"] == "local"


def test_lower_dma_start() -> None:
    """DMAStartOp should lower to a dma_start descriptor."""
    op = DMAStartOp(src="hbm", dst="sram", size_bytes=65536, event="ev0")
    result = lower_accel_to_llvm(op)
    assert len(result.lowered_ops) == 1
    lowered = result.lowered_ops[0]
    assert lowered["type"] == "dma_start"
    assert lowered["size_bytes"] == 65536
    assert lowered["event"] == "ev0"


def test_lower_dma_wait() -> None:
    """DMAWaitOp should lower to a dma_wait descriptor."""
    op = DMAWaitOp(event="ev0")
    result = lower_accel_to_llvm(op)
    assert len(result.lowered_ops) == 1
    assert result.lowered_ops[0]["type"] == "dma_wait"
    assert result.lowered_ops[0]["event"] == "ev0"


def test_lower_matrix_engine() -> None:
    """MatrixEngineOp should lower to a matrix_engine descriptor."""
    op = MatrixEngineOp(op_kind="matmul", a_ref="a", b_ref="b", c_ref="c")
    result = lower_accel_to_llvm(op)
    assert len(result.lowered_ops) == 1
    lowered = result.lowered_ops[0]
    assert lowered["type"] == "matrix_engine"
    assert lowered["op_kind"] == "matmul"
    assert lowered["a_ref"] == "a"


def test_lower_barrier() -> None:
    """BarrierOp should lower to a barrier descriptor."""
    op = BarrierOp(scope="device")
    result = lower_accel_to_llvm(op)
    assert len(result.lowered_ops) == 1
    assert result.lowered_ops[0]["type"] == "barrier"
    assert result.lowered_ops[0]["scope"] == "device"


# -- Target triple -----------------------------------------------------------


def test_lower_with_target_triple() -> None:
    """Target triple should propagate to all lowered ops."""
    triple = "x86_64-unknown-linux-gnu"
    ops = [
        TileLoadOp(src_memref="g", dst_memref="l", shape=(32,), dtype="f32"),
        BarrierOp(scope="workgroup"),
    ]
    result = lower_accel_to_llvm(ops, target_triple=triple)
    assert len(result.lowered_ops) == 2
    for lowered in result.lowered_ops:
        assert lowered["target_triple"] == triple


# -- Multiple ops ------------------------------------------------------------


def test_lower_multiple_ops() -> None:
    """Lowering a list of ops should produce one descriptor per op."""
    ops = [
        TileLoadOp(src_memref="g", dst_memref="l", shape=(32, 32), dtype="f16"),
        DMAStartOp(src="a", dst="b", size_bytes=1024, event="e1"),
        DMAWaitOp(event="e1"),
        MatrixEngineOp(op_kind="mma", a_ref="x", b_ref="y", c_ref="z"),
        BarrierOp(scope="system"),
        TileStoreOp(src_memref="l", dst_memref="g", shape=(32, 32), dtype="f16"),
    ]
    result = lower_accel_to_llvm(ops)
    assert len(result.lowered_ops) == 6
    assert result.diagnostics == []


# -- Diagnostics -------------------------------------------------------------


def test_lower_unknown_op_produces_diagnostic() -> None:
    """An unrecognized object should produce a diagnostic, not crash."""
    result = lower_accel_to_llvm(["not_an_op"])
    assert len(result.lowered_ops) == 0
    assert len(result.diagnostics) == 1
    assert "Unsupported" in result.diagnostics[0]


# -- LoweringOutput dataclass ------------------------------------------------


def test_lowering_output_defaults() -> None:
    """LoweringOutput should default to empty lists."""
    out = LoweringOutput()
    assert out.lowered_ops == []
    assert out.diagnostics == []
