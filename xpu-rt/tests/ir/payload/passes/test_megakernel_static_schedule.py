"""Tests for the StaticMegakernelSchedule pass (ETC Algorithm 1)."""

from __future__ import annotations

import json

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region

from compgen.ir.event.attrs import EventCoordAttr, EventTensorTypeAttr
from compgen.ir.event.ops import (
    CallDeviceOp,
    EventTensorOp,
    GraphOp,
)
from compgen.ir.payload.passes.megakernel_static_schedule import (
    StaticMegakernelSchedule,
    extract_event_edges,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemm_rs_module(sm_count: int = 4) -> tuple[ModuleOp, GraphOp]:
    """Build a 4-tile GEMM+ReduceScatter style graph.

    Layout:
        partial_sum (4 tasks) --notify--> E[i] --wait--> final_sum (4 tasks)
    Optimal makespan with 4 SMs: 2 us (one wave of producers, one of consumers).
    """
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([4]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("partial_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [
                        EventCoordAttr("E", [str(i)], 1)
                        for i in range(4)
                    ],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "in_edges": ArrayAttr(
                    [
                        EventCoordAttr("E", [str(i)], 1)
                        for i in range(4)
                    ],
                ),
            },
        ),
    )
    graph = GraphOp(
        sym_name="mm_rs",
        policy="static",
        sm_count=sm_count,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# extract_event_edges
# ---------------------------------------------------------------------------


def test_expand_task_grid_yields_one_id_per_coordinate() -> None:
    _, graph = _gemm_rs_module()
    tasks, _ = extract_event_edges(graph)
    ids = {t.task_id for t in tasks}
    assert {f"partial_sum:{i}" for i in range(4)} <= ids
    assert {f"final_sum:{i}" for i in range(4)} <= ids
    assert len(tasks) == 8


def test_event_edges_link_producers_to_consumers() -> None:
    _, graph = _gemm_rs_module()
    _, edges = extract_event_edges(graph)
    # Every (producer_task, consumer_task) on event E should appear at least
    # once.  We allow the over-approximation Phase A documents.
    pairs = {(e.producer, e.consumer) for e in edges}
    assert ("partial_sum:0", "final_sum:0") in pairs
    assert ("partial_sum:3", "final_sum:3") in pairs
    assert all(e.event == "E" for e in edges)


# ---------------------------------------------------------------------------
# StaticMegakernelSchedule.run
# ---------------------------------------------------------------------------


def test_pass_annotates_graph_with_schedule_attribute() -> None:
    mod, graph = _gemm_rs_module(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    assert "compgen.static_schedule" in graph.attributes
    payload = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert payload["status"] == "ok"
    assert payload["sm_count"] == 4
    assert payload["task_count"] == 8


def test_pass_produces_optimal_two_stage_makespan() -> None:
    mod, graph = _gemm_rs_module(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    payload = json.loads(graph.attributes["compgen.static_schedule"].data)
    # 4 producers in parallel (1us) + 4 consumers in parallel (1us).
    assert payload["makespan_us"] == 2.0


def test_pass_returns_per_sm_queues_and_assignment() -> None:
    mod, graph = _gemm_rs_module(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    payload = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert set(payload["per_sm_order"].keys()) <= {"0", "1", "2", "3"}
    # Every task is assigned to some SM.
    assert len(payload["assignment"]) == 8
    assert all(0 <= sm < 4 for sm in payload["assignment"].values())


def test_pass_serializes_event_tensor_decls() -> None:
    mod, graph = _gemm_rs_module()
    StaticMegakernelSchedule().run(mod)
    payload = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert payload["event_tensor_decls"] == [
        {
            "name": "E",
            "shape": [4],
            "wait_count": 1,
            "scope": "device",
            "counter_dtype": "i32",
        }
    ]


def test_pass_skips_dynamic_policy_graphs() -> None:
    mod, _ = _gemm_rs_module()
    # Mutate policy to dynamic on a fresh graph.
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([4]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    dyn_graph = GraphOp(
        sym_name="dyn_mm",
        policy="dynamic",
        sm_count=4,
        body=Region([block]),
    )
    mod_dyn = ModuleOp([])
    mod_dyn.body.block.add_op(dyn_graph)
    StaticMegakernelSchedule().run(mod_dyn)
    assert "compgen.static_schedule" not in dyn_graph.attributes


def test_pass_records_rejection_when_contracts_fail() -> None:
    """Underspecified wait_count -> deadlock detection -> rejected schedule."""
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([1]),
                "wait_count": IntegerAttr(4, IntegerType(64)),  # need 4 notifies
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("p"),
                "task_shape": ArrayAttr([IntegerAttr(1, IntegerType(64))]),
                "out_edges": ArrayAttr([EventCoordAttr("E", ["0"], 1)]),  # only 1
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("c"),
                "task_shape": ArrayAttr([IntegerAttr(1, IntegerType(64))]),
                "in_edges": ArrayAttr([EventCoordAttr("E", ["0"], 1)]),
            },
        ),
    )
    graph = GraphOp(
        sym_name="bad",
        policy="static",
        sm_count=2,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    StaticMegakernelSchedule().run(mod)
    payload = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert payload["status"] == "rejected"
    assert any("deadlock" in e for e in payload["errors"])


def test_pass_is_idempotent_in_makespan_and_task_set() -> None:
    """CP-SAT may pick different optimal solutions across runs; we only
    require the same makespan, task set, and SM partition cardinality."""
    mod, graph = _gemm_rs_module()
    StaticMegakernelSchedule().run(mod)
    a = json.loads(graph.attributes["compgen.static_schedule"].data)
    StaticMegakernelSchedule().run(mod)
    b = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert a["makespan_us"] == b["makespan_us"]
    assert a["task_count"] == b["task_count"]
    assert set(a["assignment"].keys()) == set(b["assignment"].keys())
    assert {len(q) for q in a["per_sm_order"].values()} == {
        len(q) for q in b["per_sm_order"].values()
    }
