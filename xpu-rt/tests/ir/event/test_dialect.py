"""Tests for the Event Tensor IR dialect."""

from __future__ import annotations

import pytest
from compgen.ir.event.attrs import (
    EventCoordAttr,
    EventTensorTypeAttr,
    SchedulingPolicyAttr,
)
from compgen.ir.event.contracts import check_event_graph
from compgen.ir.event.dialect import ALL_ATTRS, ALL_OPS, Event
from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
    MaterializeViewOp,
    NotifyOp,
    TriggerOp,
    UpdateOp,
    WaitOp,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

# ---------------------------------------------------------------------------
# Dialect registration
# ---------------------------------------------------------------------------


def test_dialect_registers_eight_ops_three_attrs() -> None:
    assert Event.name == "event"
    assert len(ALL_OPS) == 8
    assert len(ALL_ATTRS) == 3


def test_op_names_are_namespaced() -> None:
    for op in ALL_OPS:
        assert op.name.startswith("event.")


# ---------------------------------------------------------------------------
# Attribute construction
# ---------------------------------------------------------------------------


def test_event_tensor_type_attr_defaults() -> None:
    et = EventTensorTypeAttr([4, 8])
    assert et.counter_dtype.data == "i32"
    assert et.scope.data == "device"
    assert len(et.shape.data) == 2
    assert len(et.dim_names.data) == 2


def test_event_tensor_type_attr_symbolic_dim() -> None:
    et = EventTensorTypeAttr([-1, 32], dim_names=["B", ""], counter_dtype="u32", scope="device")
    assert et.dim_names.data[0].data == "B"


def test_event_coord_attr_default_decrement() -> None:
    coord = EventCoordAttr("E", ["i", "j"])
    assert coord.decrement.value.data == 1
    assert coord.event_ref.data == "E"


def test_scheduling_policy_attr_construction() -> None:
    p = SchedulingPolicyAttr("dynamic")
    assert p.policy.data == "dynamic"


# ---------------------------------------------------------------------------
# Op verification
# ---------------------------------------------------------------------------


def _decl(name: str, shape: list[int], wait_count: int = 1, scope: str = "device") -> EventTensorOp:
    return EventTensorOp.create(
        properties={
            "sym_name": StringAttr(name),
            "event_type": EventTensorTypeAttr(shape, scope=scope),
            "wait_count": IntegerAttr(wait_count, IntegerType(64)),
        },
    )


def test_event_tensor_op_rejects_negative_wait_count() -> None:
    op = EventTensorOp.create(
        properties={
            "sym_name": StringAttr("E"),
            "event_type": EventTensorTypeAttr([4]),
            "wait_count": IntegerAttr(-1, IntegerType(64)),
        },
    )
    with pytest.raises(VerifyException, match="wait_count"):
        op.verify_()


def test_event_tensor_op_rejects_invalid_scope() -> None:
    op = EventTensorOp.create(
        properties={
            "sym_name": StringAttr("E"),
            "event_type": EventTensorTypeAttr([4], scope="cluster"),
            "wait_count": IntegerAttr(1, IntegerType(64)),
        },
    )
    with pytest.raises(VerifyException, match="scope"):
        op.verify_()


def test_notify_rejects_non_positive_decrement() -> None:
    op = NotifyOp.create(properties={"coord": EventCoordAttr("E", ["0"], 0)})
    with pytest.raises(VerifyException, match="decrement"):
        op.verify_()


def test_call_device_rejects_zero_task_dim() -> None:
    from xdsl.dialects.builtin import ArrayAttr, SymbolRefAttr

    op = CallDeviceOp.create(
        properties={
            "device_func": SymbolRefAttr("partial_sum"),
            "task_shape": ArrayAttr([IntegerAttr(0, IntegerType(64))]),
        },
    )
    with pytest.raises(VerifyException, match="task_shape"):
        op.verify_()


def test_graph_op_rejects_invalid_policy() -> None:
    block = Block()
    block.add_op(_decl("E", [4]))
    g = GraphOp(sym_name="g", policy=SchedulingPolicyAttr("greedy"), body=Region([block]))
    with pytest.raises(VerifyException, match="policy"):
        g.verify_()


def test_materialize_view_rejects_negative_extent() -> None:
    from xdsl.dialects.builtin import ArrayAttr

    op = MaterializeViewOp.create(
        properties={
            "event_ref": StringAttr("E"),
            "concrete_shape": ArrayAttr([IntegerAttr(-3, IntegerType(64))]),
        },
    )
    with pytest.raises(VerifyException, match="non-negative"):
        op.verify_()


# ---------------------------------------------------------------------------
# Contracts (well-formedness checks over an event.graph body)
# ---------------------------------------------------------------------------


def _build_static_graph() -> GraphOp:
    block = Block()
    block.add_op(_decl("E", [4], wait_count=2))
    for k in range(8):
        block.add_op(NotifyOp.create(properties={"coord": EventCoordAttr("E", [str(k % 4)], 1)}))
    for k in range(4):
        block.add_op(WaitOp.create(properties={"coord": EventCoordAttr("E", [str(k)], 1)}))
    return GraphOp(sym_name="mm_rs", policy="static", sm_count=108, body=Region([block]))


def test_well_formed_static_graph_passes() -> None:
    graph = _build_static_graph()
    graph.verify()
    report = check_event_graph(graph)
    assert report.ok
    assert report.notify_counts == {"E": 8}
    assert report.wait_counts == {"E": 4}
    assert report.warnings == []


def test_undeclared_event_reference_is_an_error() -> None:
    block = Block()
    block.add_op(_decl("E", [4]))
    block.add_op(NotifyOp.create(properties={"coord": EventCoordAttr("F", ["0"], 1)}))
    g = GraphOp(sym_name="bad_ref", policy="static", body=Region([block]))
    report = check_event_graph(g)
    assert not report.ok
    assert any("undeclared event 'F'" in e for e in report.errors)


def test_insufficient_notifies_flags_deadlock() -> None:
    block = Block()
    block.add_op(_decl("E", [1], wait_count=4))
    block.add_op(NotifyOp.create(properties={"coord": EventCoordAttr("E", ["0"], 1)}))
    block.add_op(WaitOp.create(properties={"coord": EventCoordAttr("E", ["0"], 1)}))
    g = GraphOp(sym_name="deadlock", policy="static", body=Region([block]))
    report = check_event_graph(g)
    assert not report.ok
    assert any("deadlock" in e for e in report.errors)


def test_data_dep_op_forbidden_under_static_policy() -> None:
    block = Block()
    block.add_op(_decl("E", [4], wait_count=0))
    block.add_op(
        UpdateOp.create(
            properties={
                "target": EventCoordAttr("E", ["i"], 1),
                "source_tensor": StringAttr("topk"),
                "index_expr": StringAttr("i->topk[i,:]"),
            }
        )
    )
    g = GraphOp(sym_name="bad_static", policy="static", body=Region([block]))
    report = check_event_graph(g)
    assert not report.ok
    assert any("forbidden under static" in e for e in report.errors)


def test_data_dep_op_allowed_under_dynamic_policy() -> None:
    block = Block()
    block.add_op(_decl("E", [4], wait_count=0))
    block.add_op(
        UpdateOp.create(
            properties={
                "target": EventCoordAttr("E", ["i"], 1),
                "source_tensor": StringAttr("topk"),
                "index_expr": StringAttr("i->topk[i,:]"),
            }
        )
    )
    block.add_op(
        TriggerOp.create(
            properties={
                "target": EventCoordAttr("E", ["i"], 1),
                "trigger_range": StringAttr("exp_indptr"),
            }
        )
    )
    g = GraphOp(sym_name="moe_like", policy="dynamic", body=Region([block]))
    report = check_event_graph(g)
    assert report.ok
    assert "E" in report.data_dep_events


def test_unused_event_warns_but_does_not_error() -> None:
    block = Block()
    block.add_op(_decl("E", [4], wait_count=0))
    g = GraphOp(sym_name="unused", policy="static", body=Region([block]))
    report = check_event_graph(g)
    assert report.ok
    assert any("unused" in w for w in report.warnings)


def test_produced_but_never_waited_warns() -> None:
    block = Block()
    block.add_op(_decl("E", [1], wait_count=0))
    block.add_op(NotifyOp.create(properties={"coord": EventCoordAttr("E", ["0"], 1)}))
    g = GraphOp(sym_name="no_consumer", policy="static", body=Region([block]))
    report = check_event_graph(g)
    assert report.ok
    assert any("never waited" in w for w in report.warnings)
