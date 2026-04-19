"""Tests for W9 HMX tile primitives on the compgen.accel dialect."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.utils.exceptions import VerifyException

from compgen.ir.accel.ops import (
    ACCEL_IR_OPS,
    HMXAccumulatorClearIROp,
    HMXDMAOverlapIROp,
    HMXMatrixEngineIROp,
    HMXTileLoadIROp,
)


def _shape(*dims):
    return ArrayAttr([IntegerAttr(d, IntegerType(64)) for d in dims])


# --- HMXTileLoad -----------------------------------------------------------


def test_hmx_tile_load_basic():
    op = HMXTileLoadIROp.build(properties={
        "src_memref": StringAttr("A_dram"),
        "dst_memref": StringAttr("A_vtcm"),
        "tile_shape": _shape(32, 32),
        "format_xform": StringAttr("rm_to_ah"),
        "dtype": StringAttr("f16"),
    })
    op.verify()


def test_hmx_tile_load_identity_xform():
    op = HMXTileLoadIROp.build(properties={
        "src_memref": StringAttr("X"),
        "dst_memref": StringAttr("Y"),
        "tile_shape": _shape(64, 32),
        "format_xform": StringAttr("identity"),
        "dtype": StringAttr("bf16"),
    })
    op.verify()


def test_hmx_tile_load_rejects_unknown_xform():
    op = HMXTileLoadIROp.build(properties={
        "src_memref": StringAttr("X"),
        "dst_memref": StringAttr("Y"),
        "tile_shape": _shape(32, 32),
        "format_xform": StringAttr("bogus"),
        "dtype": StringAttr("f16"),
    })
    with pytest.raises(VerifyException, match="format_xform"):
        op.verify()


def test_hmx_tile_load_rejects_non_positive_shape():
    op = HMXTileLoadIROp.build(properties={
        "src_memref": StringAttr("X"),
        "dst_memref": StringAttr("Y"),
        "tile_shape": _shape(0, 32),
        "format_xform": StringAttr("rm_to_ah"),
        "dtype": StringAttr("f16"),
    })
    with pytest.raises(VerifyException, match="tile_shape"):
        op.verify()


# --- HMXMatrixEngine -------------------------------------------------------


def test_hmx_matrix_engine_matmul():
    op = HMXMatrixEngineIROp.build(properties={
        "a_tile": StringAttr("A"),
        "b_tile": StringAttr("B"),
        "c_tile": StringAttr("C"),
        "op_kind": StringAttr("matmul"),
        "shape": _shape(32, 32, 32),
        "dtype": StringAttr("f16"),
    })
    op.verify()


def test_hmx_matrix_engine_accumulate_variant():
    op = HMXMatrixEngineIROp.build(properties={
        "a_tile": StringAttr("A"),
        "b_tile": StringAttr("B"),
        "c_tile": StringAttr("C"),
        "op_kind": StringAttr("matmul_accumulate"),
        "shape": _shape(32, 32, 32),
        "dtype": StringAttr("f16"),
        "accumulate": StringAttr("yes"),
    })
    op.verify()


def test_hmx_matrix_engine_rejects_wrong_shape_rank():
    op = HMXMatrixEngineIROp.build(properties={
        "a_tile": StringAttr("A"),
        "b_tile": StringAttr("B"),
        "c_tile": StringAttr("C"),
        "op_kind": StringAttr("matmul"),
        "shape": _shape(32, 32),  # missing K
        "dtype": StringAttr("f16"),
    })
    with pytest.raises(VerifyException, match="shape must have 3 entries"):
        op.verify()


def test_hmx_matrix_engine_rejects_unknown_kind():
    op = HMXMatrixEngineIROp.build(properties={
        "a_tile": StringAttr("A"),
        "b_tile": StringAttr("B"),
        "c_tile": StringAttr("C"),
        "op_kind": StringAttr("zoombooster"),
        "shape": _shape(32, 32, 32),
        "dtype": StringAttr("f16"),
    })
    with pytest.raises(VerifyException, match="op_kind"):
        op.verify()


# --- HMXAccumulatorClear ---------------------------------------------------


def test_hmx_accumulator_clear_builds():
    op = HMXAccumulatorClearIROp.build(properties={
        "c_tile": StringAttr("C"),
        "dtype": StringAttr("f32"),
        "shape": _shape(32, 32),
    })
    op.verify()


# --- HMXDMAOverlap ---------------------------------------------------------


def test_hmx_dma_overlap_default_depth():
    op = HMXDMAOverlapIROp.build(properties={
        "producer_tile": StringAttr("prod"),
        "consumer_tile": StringAttr("cons"),
        "line_bytes": IntegerAttr(64, IntegerType(64)),
        "depth": IntegerAttr(2, IntegerType(64)),
    })
    op.verify()


def test_hmx_dma_overlap_quadruple():
    op = HMXDMAOverlapIROp.build(properties={
        "producer_tile": StringAttr("prod"),
        "consumer_tile": StringAttr("cons"),
        "line_bytes": IntegerAttr(128, IntegerType(64)),
        "depth": IntegerAttr(4, IntegerType(64)),
    })
    op.verify()


def test_hmx_dma_overlap_rejects_depth_1():
    op = HMXDMAOverlapIROp.build(properties={
        "producer_tile": StringAttr("p"),
        "consumer_tile": StringAttr("c"),
        "line_bytes": IntegerAttr(64, IntegerType(64)),
        "depth": IntegerAttr(1, IntegerType(64)),
    })
    with pytest.raises(VerifyException, match="depth"):
        op.verify()


def test_hmx_dma_overlap_rejects_zero_line_bytes():
    op = HMXDMAOverlapIROp.build(properties={
        "producer_tile": StringAttr("p"),
        "consumer_tile": StringAttr("c"),
        "line_bytes": IntegerAttr(0, IntegerType(64)),
        "depth": IntegerAttr(2, IntegerType(64)),
    })
    with pytest.raises(VerifyException, match="line_bytes"):
        op.verify()


# --- dialect registration -------------------------------------------------


def test_accel_ops_includes_all_4_hmx_variants():
    names = {op.name for op in ACCEL_IR_OPS}
    assert "compgen.accel.hmx_tile_load" in names
    assert "compgen.accel.hmx_matrix_engine" in names
    assert "compgen.accel.hmx_accumulator_clear" in names
    assert "compgen.accel.hmx_dma_overlap" in names


def test_accel_dialect_has_10_total_ops():
    # 6 original + 4 HMX.
    assert len(ACCEL_IR_OPS) == 10
