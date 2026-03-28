"""Tests for accelerator dialect verification."""

from __future__ import annotations

from compgen.ir.accel.ops import (
    BarrierOp,
    DMAStartOp,
    DMAWaitOp,
    MatrixEngineOp,
    TileLoadOp,
    TileStoreOp,
)
from compgen.ir.accel.verify import AccelVerificationResult, verify_accel_ops


# -- Valid ops pass ----------------------------------------------------------


def test_valid_tile_load_passes() -> None:
    """Well-formed TileLoadOp should pass verification."""
    op = TileLoadOp(src_memref="global", dst_memref="local", shape=(128, 128), dtype="float16")
    result = verify_accel_ops(op)
    assert result.valid
    assert result.errors == []


def test_valid_tile_store_passes() -> None:
    """Well-formed TileStoreOp should pass verification."""
    op = TileStoreOp(src_memref="local", dst_memref="global", shape=(64,), dtype="float32")
    result = verify_accel_ops(op)
    assert result.valid


def test_valid_dma_pair_passes() -> None:
    """DMAStart + matching DMAWait should pass verification."""
    ops = [
        DMAStartOp(src="hbm", dst="sram", size_bytes=65536, event="ev0"),
        DMAWaitOp(event="ev0"),
    ]
    result = verify_accel_ops(ops)
    assert result.valid
    assert result.errors == []


def test_valid_matrix_engine_passes() -> None:
    """Well-formed MatrixEngineOp should pass verification."""
    op = MatrixEngineOp(op_kind="matmul", a_ref="a", b_ref="b", c_ref="c")
    result = verify_accel_ops(op)
    assert result.valid


def test_valid_barrier_passes() -> None:
    """Well-formed BarrierOp should pass verification."""
    for scope in ("workgroup", "device", "system"):
        result = verify_accel_ops(BarrierOp(scope=scope))
        assert result.valid


def test_valid_mixed_ops_pass() -> None:
    """A list of well-formed ops should all pass."""
    ops = [
        TileLoadOp(src_memref="g", dst_memref="l", shape=(32, 32), dtype="float16"),
        DMAStartOp(src="a", dst="b", size_bytes=1024, event="e1"),
        DMAWaitOp(event="e1"),
        MatrixEngineOp(op_kind="mma", a_ref="x", b_ref="y", c_ref="z"),
        BarrierOp(scope="workgroup"),
    ]
    result = verify_accel_ops(ops)
    assert result.valid


# -- Invalid ops fail --------------------------------------------------------


def test_tile_load_empty_shape_fails() -> None:
    """TileLoadOp with empty shape should fail."""
    op = TileLoadOp(src_memref="g", dst_memref="l", shape=(), dtype="float32")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("shape" in msg for _, msg in result.errors)


def test_tile_load_zero_dim_fails() -> None:
    """TileLoadOp with zero dimension should fail."""
    op = TileLoadOp(src_memref="g", dst_memref="l", shape=(128, 0), dtype="float32")
    result = verify_accel_ops(op)
    assert not result.valid


def test_tile_load_empty_dtype_fails() -> None:
    """TileLoadOp with empty dtype should fail."""
    op = TileLoadOp(src_memref="g", dst_memref="l", shape=(32,), dtype="")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("dtype" in msg for _, msg in result.errors)


def test_dma_start_zero_size_fails() -> None:
    """DMAStartOp with size_bytes=0 should fail."""
    op = DMAStartOp(src="a", dst="b", size_bytes=0, event="ev")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("size_bytes" in msg for _, msg in result.errors)


def test_dma_start_empty_event_fails() -> None:
    """DMAStartOp with empty event should fail."""
    op = DMAStartOp(src="a", dst="b", size_bytes=1024, event="")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("event" in msg for _, msg in result.errors)


def test_mismatched_event_fails() -> None:
    """DMAWait on non-existent event should fail."""
    ops = [
        DMAStartOp(src="a", dst="b", size_bytes=1024, event="ev0"),
        DMAWaitOp(event="ev_missing"),
    ]
    result = verify_accel_ops(ops)
    assert not result.valid
    assert any("ev_missing" in msg for _, msg in result.errors)


def test_dma_wait_without_start_fails() -> None:
    """DMAWait with no DMAStart at all should fail."""
    result = verify_accel_ops(DMAWaitOp(event="orphan"))
    assert not result.valid
    assert any("orphan" in msg for _, msg in result.errors)


def test_matrix_engine_invalid_kind_fails() -> None:
    """MatrixEngineOp with bad op_kind should fail."""
    op = MatrixEngineOp(op_kind="invalid", a_ref="a", b_ref="b", c_ref="c")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("op_kind" in msg for _, msg in result.errors)


def test_matrix_engine_empty_ref_fails() -> None:
    """MatrixEngineOp with empty a_ref should fail."""
    op = MatrixEngineOp(op_kind="matmul", a_ref="", b_ref="b", c_ref="c")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("a_ref" in msg for _, msg in result.errors)


def test_barrier_invalid_scope_fails() -> None:
    """BarrierOp with invalid scope should fail."""
    op = BarrierOp(scope="invalid_scope")
    result = verify_accel_ops(op)
    assert not result.valid
    assert any("scope" in msg for _, msg in result.errors)


# -- Result dataclass --------------------------------------------------------


def test_verification_result_fields() -> None:
    """AccelVerificationResult should have correct field defaults."""
    ok = AccelVerificationResult(valid=True)
    assert ok.valid
    assert ok.errors == []

    bad = AccelVerificationResult(valid=False, errors=[("X", "msg")])
    assert not bad.valid
    assert len(bad.errors) == 1
