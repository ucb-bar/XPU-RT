"""Tests for W4.4 ``normalize_subbyte``."""

from __future__ import annotations

from compgen.ir.payload.passes.rewrites.normalize_subbyte import (
    NormalizeSubbyteStats,
    run_normalize_subbyte,
)
from compgen.ir.quant import (
    AffineQuantizedTensorType,
    PackedIntTensorType,
    WeightInt4PackMMOp,
    WeightInt4PackQMOp,
    WeightInt8PackMMOp,
)
from xdsl.dialects.builtin import (
    Float32Type,
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import (
    assert_module_verifies,
)


def _ft(shape, elem=None):
    return TensorType(elem if elem is not None else Float32Type(), list(shape))


def _int4_module() -> tuple[ModuleOp, WeightInt4PackMMOp]:
    B, K, O = 4, 128, 32
    xt = _ft([B, K])
    w_t = _ft([O, K // 2], IntegerType(8))
    sz_t = _ft([O, 2])
    r_t = _ft([B, O])
    x = EmptyOp([], xt)
    w = EmptyOp([], w_t)
    sz = EmptyOp([], sz_t)
    op = WeightInt4PackMMOp(
        operands=[x.results[0], w.results[0], sz.results[0]],
        result_types=[r_t],
        properties={"group_size": IntegerAttr(128, IntegerType(64))},
    )
    block = Block()
    for o in (x, w, sz, op):
        block.add_op(o)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    return ModuleOp([func]), op


def _int8_module_with_packed_qtype() -> tuple[ModuleOp, WeightInt8PackMMOp]:
    """Build an int8 pack mm that carries an explicit PackedIntTensorType qtype."""
    B, K, O = 4, 128, 32
    xt = _ft([B, K])
    w_t = _ft([O, K], IntegerType(8))
    s_t = _ft([O])
    r_t = _ft([B, O])
    x = EmptyOp([], xt)
    w = EmptyOp([], w_t)
    s = EmptyOp([], s_t)
    op = WeightInt8PackMMOp(
        operands=[x.results[0], w.results[0], s.results[0]],
        result_types=[r_t],
    )
    # attach an explicit qtype carrying a PackedIntTensorType storage
    packed = PackedIntTensorType(bit_width=4, pack_dim=1)
    qtype = AffineQuantizedTensorType(
        storage_type=packed,
        scale_dtype=Float32Type(),
        granularity="per_channel",
        block_size=[1, 1],
    )
    op.properties["qtype"] = qtype

    block = Block()
    for o in (x, w, s, op):
        block.add_op(o)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    return ModuleOp([func]), op


# --- happy path ------------------------------------------------------------


def test_int4_pack_mm_gets_canonical_tag():
    m, op = _int4_module()
    stats = run_normalize_subbyte(m)
    assert stats.ops_with_qtype == 1
    assert "compgen.subbyte_canonical" in op.attributes
    data = op.attributes["compgen.subbyte_canonical"].data
    assert "bit_width=4" in data
    assert "pack_dim=1" in data
    assert_module_verifies(m)


def test_int4_pack_mm_gets_boundary_unpack():
    m, op = _int4_module()
    run_normalize_subbyte(m)
    assert op.attributes["compgen.subbyte_boundary"].data == "unpack"


def test_int4_qm_gets_canonical_tag():
    B, K, O = 2, 64, 16
    xt = _ft([B, K])
    w_t = _ft([B, O, K // 2], IntegerType(8))
    sz_t = _ft([B, O, 2])
    r_t = _ft([B, B, O])
    x = EmptyOp([], xt)
    w = EmptyOp([], w_t)
    sz = EmptyOp([], sz_t)
    op = WeightInt4PackQMOp(
        operands=[x.results[0], w.results[0], sz.results[0]],
        result_types=[r_t],
        properties={"group_size": IntegerAttr(64, IntegerType(64))},
    )
    block = Block()
    for o in (x, w, sz, op):
        block.add_op(o)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    m = ModuleOp([func])

    stats = run_normalize_subbyte(m)
    assert stats.ops_with_qtype >= 1
    assert "compgen.subbyte_canonical" in op.attributes


def test_int8_with_explicit_packed_qtype_is_annotated():
    m, op = _int8_module_with_packed_qtype()
    stats = run_normalize_subbyte(m)
    # The int8 pack mm carries an explicit packed qtype → picked up.
    assert stats.ops_with_qtype == 1
    assert "compgen.subbyte_canonical" in op.attributes


# --- non-matching cases ---------------------------------------------------


def test_plain_tensor_empty_is_untouched():
    t = _ft([4, 8])
    e = EmptyOp([], t)
    block = Block()
    block.add_op(e)
    block.add_op(ReturnOp(e.results[0]))
    func = FuncOp("forward", FunctionType.from_lists([], [t]), Region([block]))
    m = ModuleOp([func])
    stats = run_normalize_subbyte(m)
    assert stats.ops_with_qtype == 0


def test_int8_pack_mm_without_explicit_packed_qtype_is_skipped():
    # A plain int8 pack mm (no qtype attached) isn't sub-byte.
    B, K, O = 4, 128, 32
    xt = _ft([B, K])
    w_t = _ft([O, K], IntegerType(8))
    s_t = _ft([O])
    r_t = _ft([B, O])
    x = EmptyOp([], xt)
    w = EmptyOp([], w_t)
    s = EmptyOp([], s_t)
    op = WeightInt8PackMMOp(
        operands=[x.results[0], w.results[0], s.results[0]],
        result_types=[r_t],
    )
    block = Block()
    for o in (x, w, s, op):
        block.add_op(o)
    block.add_op(ReturnOp(op.result))
    func = FuncOp("forward", FunctionType.from_lists([], [r_t]), Region([block]))
    m = ModuleOp([func])

    stats = run_normalize_subbyte(m)
    assert stats.ops_with_qtype == 0


# --- idempotence + stats --------------------------------------------------


def test_idempotent_second_run_is_noop():
    m, _ = _int4_module()
    first = run_normalize_subbyte(m)
    assert first.ops_with_qtype == 1
    second = run_normalize_subbyte(m)
    assert second.ops_with_qtype == 0  # already annotated -> skipped


def test_stats_initial_values():
    s = NormalizeSubbyteStats()
    assert s.ops_with_qtype == 0
    assert s.boundaries_annotated == 0
    assert s.canonical_bit_widths == {}


def test_canonical_bit_widths_recorded():
    m, _ = _int4_module()
    stats = run_normalize_subbyte(m)
    assert stats.canonical_bit_widths.get(4, 0) == 1
