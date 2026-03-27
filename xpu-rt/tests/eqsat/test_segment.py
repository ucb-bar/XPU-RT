"""Tests for segmentation and blackboxing."""

from __future__ import annotations

from compgen.eqsat.blackbox import (
    OpClass,
    classify_module,
    classify_op,
    count_blackbox,
    count_profitable,
)
from compgen.eqsat.segment import segment_function, segment_module
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_arith_chain(n: int) -> ModuleOp:
    """Create a chain of n addi ops."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    prev = a
    for _ in range(n):
        add = arith.AddiOp(prev, b)
        block.add_op(add)
        prev = add.result
    block.add_op(func.ReturnOp(prev))
    return ModuleOp([func.FuncOp("chain", ([idx, idx], [idx]), Region([block]))])


def _make_mixed_module() -> ModuleOp:
    """Module with profitable (addi) and blackbox (call) ops."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args

    # Profitable
    add1 = arith.AddiOp(a, b)
    block.add_op(add1)

    # Blackbox (func.call)
    call = func.CallOp("ext_fn", [add1.result], [idx])
    block.add_op(call)

    # Profitable
    add2 = arith.AddiOp(call.results[0], b)
    block.add_op(add2)

    block.add_op(func.ReturnOp(add2.result))
    return ModuleOp([func.FuncOp("mixed", ([idx, idx], [idx]), Region([block]))])


# ============================================================================
# Blackbox tests
# ============================================================================


class TestBlackbox:
    def test_addi_is_profitable(self) -> None:
        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        op = arith.AddiOp(block.args[0], block.args[1])
        block.add_op(op)
        assert classify_op(op) == OpClass.PROFITABLE

    def test_call_is_blackbox(self) -> None:
        idx = IndexType()
        block = Block(arg_types=[idx])
        op = func.CallOp("ext", [block.args[0]], [idx])
        block.add_op(op)
        assert classify_op(op) == OpClass.BLACKBOX

    def test_classify_module(self) -> None:
        module = _make_mixed_module()
        classes = classify_module(module)
        profitable = count_profitable(classes)
        blackbox = count_blackbox(classes)
        assert profitable == 2  # two addi ops
        assert blackbox == 1  # one call op

    def test_constant_is_profitable(self) -> None:
        from xdsl.dialects.builtin import IntegerAttr
        block = Block()
        op = arith.ConstantOp(IntegerAttr.from_index_int_value(42))
        block.add_op(op)
        assert classify_op(op) == OpClass.PROFITABLE


# ============================================================================
# Segmentation tests
# ============================================================================


class TestSegmentation:
    def test_single_segment_small_module(self) -> None:
        module = _make_arith_chain(5)
        func_op = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        segments = segment_function(func_op, threshold=200)
        assert len(segments) == 1
        assert segments[0].profitable_count == 5

    def test_multiple_segments_at_threshold(self) -> None:
        module = _make_arith_chain(10)
        func_op = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        segments = segment_function(func_op, threshold=3)
        # 10 ops, threshold 3 → should produce ceil(10/3) = 4 segments
        assert len(segments) == 4
        assert segments[0].profitable_count == 3
        assert segments[1].profitable_count == 3
        assert segments[2].profitable_count == 3
        assert segments[3].profitable_count == 1

    def test_threshold_1_gives_many_segments(self) -> None:
        module = _make_arith_chain(5)
        func_op = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        segments = segment_function(func_op, threshold=1)
        assert len(segments) == 5

    def test_blackbox_ops_dont_count_against_threshold(self) -> None:
        module = _make_mixed_module()
        func_op = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        segments = segment_function(func_op, threshold=200)
        # Should be 1 segment (2 profitable < 200 threshold)
        assert len(segments) == 1
        assert segments[0].profitable_count == 2
        assert segments[0].blackbox_count == 1

    def test_segment_module(self) -> None:
        module = _make_arith_chain(6)
        segments = segment_module(module, threshold=2)
        assert len(segments) == 3
        # Check global IDs
        assert segments[0].segment_id == 0
        assert segments[1].segment_id == 1
        assert segments[2].segment_id == 2

    def test_dataflow_marking(self) -> None:
        module = _make_arith_chain(6)
        func_op = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
        segments = segment_function(func_op, threshold=2)
        # Chain: each segment depends on the previous
        assert len(segments) == 3
        assert not segments[0].has_dataflow_in  # First segment has no predecessors
        assert segments[0].has_dataflow_out  # But feeds into next
        assert segments[1].has_dataflow_in
        assert segments[1].has_dataflow_out
        assert segments[2].has_dataflow_in

    def test_empty_function(self) -> None:
        idx = IndexType()
        block = Block(arg_types=[idx])
        block.add_op(func.ReturnOp(block.args[0]))
        module = ModuleOp([func.FuncOp("empty", ([idx], [idx]), Region([block]))])
        segments = segment_module(module, threshold=200)
        assert len(segments) == 0

    def test_segment_ids_unique(self) -> None:
        module = _make_arith_chain(20)
        segments = segment_module(module, threshold=3)
        ids = [s.segment_id for s in segments]
        assert len(ids) == len(set(ids))
