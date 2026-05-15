"""Tests for accelerator dialect ops."""

from __future__ import annotations

from compgen.ir.accel.ops import BarrierOp, DMAStartOp, DMAWaitOp, MatrixEngineOp, TileLoadOp


def test_tile_load() -> None:
    op = TileLoadOp(src_memref="global", dst_memref="local", shape=(128, 128), dtype="float16")
    assert op.shape == (128, 128)
    assert op.dtype == "float16"


def test_dma_start_wait() -> None:
    start = DMAStartOp(src="hbm", dst="sram", size_bytes=65536, event="ev0")
    wait = DMAWaitOp(event="ev0")
    assert start.event == wait.event


def test_matrix_engine() -> None:
    op = MatrixEngineOp(op_kind="matmul", a_ref="a", b_ref="b", c_ref="c")
    assert op.op_kind == "matmul"


def test_barrier() -> None:
    op = BarrierOp(scope="device")
    assert op.scope == "device"
