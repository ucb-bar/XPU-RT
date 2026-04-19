"""Tests for the Phase-5 ``ExecutionPlan`` dataclass family."""

from __future__ import annotations

import pytest

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    CopyEdge,
    DependencyEdge,
    ExecutionPlan,
    FallbackTransition,
    Lifetime,
    QueueAssignment,
    QueueEntry,
    RegionPlacement,
    Resource,
    StreamAnnotation,
    SyncEdge,
    ticks_spanned,
)
from compgen.runtime.plan_builder import ExecutionPlanBuilder


# --- Lifetime -----------------------------------------------------------------


def test_lifetime_overlap_basic():
    a = Lifetime(0, 10)
    b = Lifetime(5, 15)
    c = Lifetime(11, 20)
    assert a.overlaps(b)
    assert not a.overlaps(c)
    assert b.overlaps(c)


def test_lifetime_persistent_always_overlaps():
    a = Lifetime(0, 10, persistent=True)
    b = Lifetime(100, 110)
    assert a.overlaps(b)


def test_lifetime_touching_endpoints_are_considered_overlap():
    # closed interval: [0, 10] and [10, 20] share tick 10.
    a = Lifetime(0, 10)
    b = Lifetime(10, 20)
    assert a.overlaps(b)


# --- Builder ------------------------------------------------------------------


def _build_basic_plan() -> ExecutionPlan:
    return (
        ExecutionPlanBuilder("toy", "cuda_h100")
        .add_resource("gpu0", "compute", device="cuda:0", capacity=1.0)
        .add_region("r0", "cuda:0", "q0")
        .add_region("r1", "cuda:0", "q0")
        .add_buffer("buf_a", 1024, "hbm", 0, 10)
        .add_buffer("buf_b", 2048, "hbm", 11, 20)
        .add_dependency("r0", "r1", value_ref="buf_a")
        .add_queue_entry("q0", "r0", 0, est_duration_ns=100)
        .add_queue_entry("q0", "r1", 5)
        .build()
    )


def test_builder_round_trip_through_dict():
    plan = _build_basic_plan()
    restored = ExecutionPlan.from_dict(plan.to_dict())
    restored.validate()
    assert restored.workload == plan.workload
    assert restored.buffer_ids == plan.buffer_ids
    assert restored.region_ids == plan.region_ids


def test_builder_validation_fires_on_duplicate_region():
    b = ExecutionPlanBuilder("t", "c")
    b.add_region("r0", "d", "q")
    b.add_region("r0", "d", "q")
    with pytest.raises(ValueError, match="duplicate region_id"):
        b.build()


def test_builder_validation_fires_on_duplicate_buffer():
    b = ExecutionPlanBuilder("t", "c")
    b.add_buffer("buf", 1024, "hbm", 0, 10)
    b.add_buffer("buf", 2048, "hbm", 0, 10)
    with pytest.raises(ValueError, match="duplicate buffer_id"):
        b.build()


def test_validate_rejects_inverted_lifetime():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers.append(
        BufferDescriptor(
            "buf", 1024, "hbm", Lifetime(10, 5), "exclusive", ""
        )
    )
    with pytest.raises(ValueError, match="first_use_tick"):
        plan.validate()


def test_validate_rejects_alias_without_target():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers.append(
        BufferDescriptor(
            "buf", 1024, "hbm", Lifetime(0, 10), "alias", ""
        )
    )
    with pytest.raises(ValueError, match="alias_of to be set"):
        plan.validate()


def test_validate_rejects_self_alias():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers.append(
        BufferDescriptor(
            "buf", 1024, "hbm", Lifetime(0, 10), "alias", "buf"
        )
    )
    with pytest.raises(ValueError, match="different buffer"):
        plan.validate()


def test_validate_rejects_dangling_alias_target():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers.append(
        BufferDescriptor(
            "buf", 1024, "hbm", Lifetime(0, 10), "alias", "ghost"
        )
    )
    with pytest.raises(ValueError, match="alias_of references unknown"):
        plan.validate()


def test_validate_rejects_dangling_copy_edge():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers.append(
        BufferDescriptor("a", 1024, "hbm", Lifetime(0, 10), "exclusive", "")
    )
    plan.copy_edges.append(
        CopyEdge(from_buffer="a", to_buffer="ghost", size_bytes=1024, transfer_path="x")
    )
    with pytest.raises(ValueError, match="to_buffer"):
        plan.validate()


def test_validate_rejects_dangling_dependency():
    plan = ExecutionPlan(workload="t", target="c")
    plan.region_placement.append(RegionPlacement("r0", "d", "q"))
    plan.dependency_edges.append(DependencyEdge("r0", "ghost", ""))
    with pytest.raises(ValueError, match="dependency_edge to_region"):
        plan.validate()


# --- accessors ----------------------------------------------------------------


def test_plan_accessors():
    plan = _build_basic_plan()
    assert plan.queue_ids == ["q0"]
    assert plan.buffer("buf_a").size_bytes == 1024
    assert plan.placement_for("r1").region_id == "r1"


def test_plan_buffer_missing_raises_key_error():
    plan = _build_basic_plan()
    with pytest.raises(KeyError):
        plan.buffer("nonexistent")


# --- apply_queue_assignment + apply_stream_annotation ------------------------


def test_apply_queue_assignment_overwrites_queue_and_priority():
    b = ExecutionPlanBuilder("t", "c")
    b.add_region("r0", "d", "q_old", priority=0)
    b.add_region("r1", "d", "q_old", priority=0)
    b.apply_queue_assignment([
        QueueAssignment("r0", "q_new", 5),
        QueueAssignment("r1", "q_new", 3),
    ])
    assert b.plan.region_placement[0].queue == "q_new"
    assert b.plan.region_placement[0].priority == 5
    assert b.plan.region_placement[1].priority == 3


def test_apply_stream_annotation_sets_stream_id_and_kind():
    b = ExecutionPlanBuilder("t", "c")
    b.add_region("r0", "d", "q")
    b.apply_stream_annotation([
        StreamAnnotation("r0", stream_id=2, kind="async_wrap"),
    ])
    assert b.plan.region_placement[0].stream_id == 2
    assert b.plan.summary["stream_kinds"]["r0"] == "async_wrap"


# --- ticks_spanned ------------------------------------------------------------


def test_ticks_spanned_uses_max_buffer_lifetime():
    plan = _build_basic_plan()
    assert ticks_spanned(plan) == 20  # buf_b last_use_tick


def test_ticks_spanned_considers_queue_entries():
    plan = ExecutionPlan(workload="t", target="c")
    plan.queue_timeline.append(QueueEntry("q", "r", 100))
    assert ticks_spanned(plan) == 100


# --- fallback + summary -------------------------------------------------------


def test_fallback_round_trip():
    plan = ExecutionPlan(workload="t", target="c")
    plan.fallback_transitions.append(
        FallbackTransition("device.queue0.depth > 4", "plan_b")
    )
    d = plan.to_dict()
    restored = ExecutionPlan.from_dict(d)
    assert len(restored.fallback_transitions) == 1
    assert restored.fallback_transitions[0].condition == "device.queue0.depth > 4"


def test_summary_round_trip():
    plan = ExecutionPlan(workload="t", target="c")
    plan.summary["note"] = "hand-built"
    d = plan.to_dict()
    restored = ExecutionPlan.from_dict(d)
    assert restored.summary == {"note": "hand-built"}
