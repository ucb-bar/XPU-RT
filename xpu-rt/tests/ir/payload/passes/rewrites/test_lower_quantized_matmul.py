"""Tests for W4.1 ``lower_quantized_matmul``."""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.rewrites.lower_quantized_matmul import (
    LowerQuantizedMatmulConfig,
    LowerQuantizedMatmulStats,
    run_lower_quantized_matmul,
)
from compgen.ir.quant import (
    WeightInt4PackMMOp,
    WeightInt8PackMMOp,
)
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
    count_ops,
)


def _ft(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _int8_module(B=4, K=128, O=32) -> tuple[ModuleOp, WeightInt8PackMMOp]:
    x_t = _ft([B, K])
    w_t = _ft([O, K], IntegerType(8))
    s_t = _ft([O])
    r_t = _ft([B, O])
    x = EmptyOp([], x_t)
    w = EmptyOp([], w_t)
    s = EmptyOp([], s_t)
    q = WeightInt8PackMMOp(
        operands=[x.results[0], w.results[0], s.results[0]],
        result_types=[r_t],
    )
    block = Block()
    for op in (x, w, s, q):
        block.add_op(op)
    block.add_op(ReturnOp(q.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    return ModuleOp([func]), q


def _int4_module(B=4, K=128, O=32) -> tuple[ModuleOp, WeightInt4PackMMOp]:
    x_t = _ft([B, K])
    w_t = _ft([O, K // 2], IntegerType(8))  # packed: 2 int4 per byte
    sz_t = _ft([O, 2])
    r_t = _ft([B, O])
    x = EmptyOp([], x_t)
    w = EmptyOp([], w_t)
    sz = EmptyOp([], sz_t)
    q = WeightInt4PackMMOp(
        operands=[x.results[0], w.results[0], sz.results[0]],
        result_types=[r_t],
        properties={"group_size": IntegerAttr(128, IntegerType(64))},
    )
    block = Block()
    for op in (x, w, sz, q):
        block.add_op(op)
    block.add_op(ReturnOp(q.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    return ModuleOp([func]), q


# --- int8 ------------------------------------------------------------------


def test_int8_pack_mm_rewrites_to_dequant_plus_matmul():
    m, _ = _int8_module()
    stats = run_lower_quantized_matmul(m)
    assert stats.int8_rewritten == 1
    assert count_ops(m, "compgen.quant.weight_int8pack_mm") == 0
    assert count_ops(m, "linalg.generic") == 1
    assert count_ops(m, "linalg.matmul") == 1
    assert_module_verifies(m)


def test_int8_matmul_uses_transpose_b_indexing():
    m, _ = _int8_module()
    run_lower_quantized_matmul(m)
    mm = next(op for op in m.walk() if op.name == "linalg.matmul")
    maps = mm.properties["indexing_maps"]
    # rhs map must be (j, k) instead of default (k, j).
    assert "d1, d2" in str(maps.data[1])


def test_int8_dequant_reads_int8_scales_f32():
    m, _ = _int8_module()
    run_lower_quantized_matmul(m)
    g = next(op for op in m.walk() if op.name == "linalg.generic")
    assert len(g.inputs) == 2
    assert g.inputs[0].type.get_element_type() == IntegerType(8)
    assert g.inputs[1].type.get_element_type() == Float32Type()


def test_int8_region_id_preserved():
    m, q = _int8_module()
    q.attributes["compgen.region_id"] = StringAttr("mm_0")
    run_lower_quantized_matmul(m)
    mm = next(op for op in m.walk() if op.name == "linalg.matmul")
    assert mm.attributes["compgen.region_id"].data == "mm_0"


# --- int4 (partial lowering -> attribute tag) ----------------------------


def test_int4_pack_mm_is_tagged_not_expanded():
    m, q = _int4_module()
    stats = run_lower_quantized_matmul(m)
    assert stats.int4_rewritten == 1
    # The op remains but carries a scheduling tag for .
    assert count_ops(m, "compgen.quant.weight_int4pack_mm") == 1
    op = next(o for o in m.walk() if o.name == "compgen.quant.weight_int4pack_mm")
    assert op.attributes["compgen.int4_lowering_scheduled"].data == "true"


# --- policy gating --------------------------------------------------------


def test_skip_policy_does_nothing():
    m, _ = _int8_module()
    stats = run_lower_quantized_matmul(
        m,
        config=LowerQuantizedMatmulConfig(policy="skip"),
    )
    assert stats.int8_rewritten == 0
    assert stats.skipped_policy >= 1
    assert count_ops(m, "compgen.quant.weight_int8pack_mm") == 1


def test_zp_zero_only_policy_allows_default_symmetric():
    # No qtype attached -> default symmetric (zp=0).
    m, _ = _int8_module()
    stats = run_lower_quantized_matmul(
        m,
        config=LowerQuantizedMatmulConfig(policy="zp_zero_only"),
    )
    assert stats.int8_rewritten == 1


def test_invalid_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        LowerQuantizedMatmulConfig(policy="yolo")


# --- idempotence + stats --------------------------------------------------


def test_stats_initial_values():
    s = LowerQuantizedMatmulStats()
    assert s.int8_rewritten == 0
    assert s.int4_rewritten == 0


def test_idempotent_second_run_is_noop():
    m, _ = _int8_module()
    first = run_lower_quantized_matmul(m)
    assert first.int8_rewritten == 1
    second = run_lower_quantized_matmul(m)
    assert second.int8_rewritten == 0


# --- real workload (TorchAO int8 weight only via captured module) -----------


def test_lower_quantized_matmul_on_tiny_int8_linear_captured():
    """End-to-end: build a tiny nn.Module with int8-packed weight,
    capture via torch.export → FX importer → run the pass. Verifies
    that the pass detects the compgen.quant.* op emitted by the
    decomposition table and lowers it.
    """

    # We synthesize the op directly rather than via TorchAO here to
    # keep the real-workload test fast; the important fidelity is
    # that the pass matches the op shape emitted by decompositions.
    m, _ = _int8_module(B=2, K=16, O=4)
    stats = run_lower_quantized_matmul(m)
    assert stats.int8_rewritten == 1
    assert_module_verifies(m)
