"""Tests for W8.4 ``lower_event_tensor_to_atomic``."""

from __future__ import annotations

from compgen.ir.event import (
    EventCoordAttr,
    EventTensorOp,
    EventTensorTypeAttr,
    GraphOp,
    NotifyOp,
    WaitOp,
)
from compgen.ir.payload.passes.rewrites.lower_event_tensor_to_atomic import (
    LowerEventTensorToAtomicConfig,
    LowerEventTensorToAtomicStats,
    run_lower_event_tensor_to_atomic,
)
from xdsl.dialects.builtin import (
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region

from tests.ir.payload.passes._pattern_test_helpers import assert_module_verifies

# --- fixtures ---------------------------------------------------------------


def _graph_with_et_notify_wait(
    sym: str = "E",
    shape: tuple[int, ...] = (4,),
    wait_count: int = 1,
) -> tuple[ModuleOp, EventTensorOp, NotifyOp, WaitOp]:
    et_type = EventTensorTypeAttr(shape=list(shape), counter_dtype="i32", scope="device")
    et = EventTensorOp.build(
        properties={
            "sym_name": StringAttr(sym),
            "event_type": et_type,
            "wait_count": IntegerAttr(wait_count, IntegerType(64)),
        },
    )
    coord = EventCoordAttr(event_ref=sym, indices=["0"], decrement=1)
    notify = NotifyOp.build(properties={"coord": coord})
    wait = WaitOp.build(properties={"coord": coord})
    graph_body = Block()
    for op in (et, notify, wait):
        graph_body.add_op(op)
    graph = GraphOp("g0", "static", body=Region([graph_body]))
    block = Block()
    block.add_op(graph)
    block.add_op(ReturnOp())
    func = FuncOp("forward", FunctionType.from_lists([], []), Region([block]))
    return ModuleOp([func]), et, notify, wait


# --- basic lowering --------------------------------------------------------


def test_notify_becomes_func_call():
    m, et, notify, wait = _graph_with_et_notify_wait()
    stats = run_lower_event_tensor_to_atomic(m)
    assert stats.notifies_lowered == 1
    calls = [
        op for op in m.walk() if isinstance(op, CallOp) and op.callee.string_value() == "compgen_event_atomic_decrement"
    ]
    assert len(calls) == 1
    assert_module_verifies(m)


def test_wait_becomes_func_call():
    m, *_ = _graph_with_et_notify_wait()
    stats = run_lower_event_tensor_to_atomic(m)
    assert stats.waits_lowered == 1
    calls = [op for op in m.walk() if isinstance(op, CallOp) and op.callee.string_value() == "compgen_event_spin_wait"]
    assert len(calls) == 1


def test_event_tensor_gets_lowered_tag():
    m, et, *_ = _graph_with_et_notify_wait()
    stats = run_lower_event_tensor_to_atomic(m)
    assert stats.event_tensors_lowered == 1
    assert et.attributes["compgen.lowered_to_atomic"].data == "true"
    assert et.attributes["compgen.lowered_counter_dtype"].data == "i32"


def test_external_decls_are_emitted():
    m, *_ = _graph_with_et_notify_wait()
    run_lower_event_tensor_to_atomic(m)
    names = {op.sym_name.data for op in m.ops if isinstance(op, FuncOp)}
    assert "compgen_event_atomic_decrement" in names
    assert "compgen_event_spin_wait" in names
    assert "compgen_event_init" in names


# --- attribute carrying ----------------------------------------------------


def test_notify_call_carries_event_ref_and_indices():
    m, et, notify, _ = _graph_with_et_notify_wait()
    run_lower_event_tensor_to_atomic(m)
    call = next(
        op for op in m.walk() if isinstance(op, CallOp) and op.callee.string_value() == "compgen_event_atomic_decrement"
    )
    assert call.attributes["compgen.event_ref"].data == "E"
    indices = call.attributes["compgen.event_indices"]
    # indices is an ArrayAttr of StringAttr.
    assert indices.data[0].data == "0"
    assert call.attributes["compgen.event_decrement"].value.data == 1


def test_wait_call_carries_event_ref_and_indices():
    m, *_ = _graph_with_et_notify_wait()
    run_lower_event_tensor_to_atomic(m)
    call = next(
        op for op in m.walk() if isinstance(op, CallOp) and op.callee.string_value() == "compgen_event_spin_wait"
    )
    assert call.attributes["compgen.event_ref"].data == "E"


# --- graph tagging --------------------------------------------------------


def test_graph_tagged_after_lowering():
    m, *_ = _graph_with_et_notify_wait()
    run_lower_event_tensor_to_atomic(m)
    graph = next(op for op in m.walk() if isinstance(op, GraphOp))
    assert graph.attributes["compgen.event_lowered_to_atomic"].data == "true"


# --- custom config ---------------------------------------------------------


def test_custom_function_names():
    m, *_ = _graph_with_et_notify_wait()
    cfg = LowerEventTensorToAtomicConfig(
        atomic_decrement_fn="my_dec",
        spin_wait_fn="my_wait",
    )
    run_lower_event_tensor_to_atomic(m, config=cfg)
    names = {op.sym_name.data for op in m.ops if isinstance(op, FuncOp)}
    assert "my_dec" in names
    assert "my_wait" in names


def test_custom_counter_dtype_recorded_on_event_tensor():
    m, et, *_ = _graph_with_et_notify_wait()
    cfg = LowerEventTensorToAtomicConfig(counter_dtype="i64")
    run_lower_event_tensor_to_atomic(m, config=cfg)
    assert et.attributes["compgen.lowered_counter_dtype"].data == "i64"


# --- noop + idempotence --------------------------------------------------


def test_module_with_no_event_ops_is_noop():
    block = Block()
    block.add_op(ReturnOp())
    func = FuncOp("empty", FunctionType.from_lists([], []), Region([block]))
    m = ModuleOp([func])
    stats = run_lower_event_tensor_to_atomic(m)
    assert stats.event_tensors_lowered == 0
    assert stats.notifies_lowered == 0
    assert stats.waits_lowered == 0
    assert_module_verifies(m)


def test_idempotent_second_run_is_noop():
    m, *_ = _graph_with_et_notify_wait()
    first = run_lower_event_tensor_to_atomic(m)
    assert first.notifies_lowered == 1
    second = run_lower_event_tensor_to_atomic(m)
    # After first run all notify/wait ops are gone -> second run sees
    # nothing to process.
    assert second.notifies_lowered == 0
    assert second.waits_lowered == 0


def test_stats_initial_values():
    s = LowerEventTensorToAtomicStats()
    assert s.event_tensors_lowered == 0
    assert s.notifies_lowered == 0


# --- multiple events -----------------------------------------------------


def test_multiple_event_tensors_and_notifies():
    et_type_a = EventTensorTypeAttr(shape=[2], counter_dtype="i32", scope="device")
    et_type_b = EventTensorTypeAttr(shape=[4], counter_dtype="i32", scope="device")
    et_a = EventTensorOp.build(
        properties={
            "sym_name": StringAttr("A"),
            "event_type": et_type_a,
            "wait_count": IntegerAttr(1, IntegerType(64)),
        },
    )
    et_b = EventTensorOp.build(
        properties={
            "sym_name": StringAttr("B"),
            "event_type": et_type_b,
            "wait_count": IntegerAttr(2, IntegerType(64)),
        },
    )
    na = NotifyOp.build(properties={"coord": EventCoordAttr("A", ["0"])})
    nb = NotifyOp.build(properties={"coord": EventCoordAttr("B", ["3"])})
    wb = WaitOp.build(properties={"coord": EventCoordAttr("B", ["0"])})

    graph_body = Block()
    for op in (et_a, et_b, na, nb, wb):
        graph_body.add_op(op)
    graph = GraphOp("g", "dynamic", body=Region([graph_body]))
    block = Block()
    block.add_op(graph)
    block.add_op(ReturnOp())
    func = FuncOp("forward", FunctionType.from_lists([], []), Region([block]))
    m = ModuleOp([func])

    stats = run_lower_event_tensor_to_atomic(m)
    assert stats.event_tensors_lowered == 2
    assert stats.notifies_lowered == 2
    assert stats.waits_lowered == 1
    assert_module_verifies(m)
