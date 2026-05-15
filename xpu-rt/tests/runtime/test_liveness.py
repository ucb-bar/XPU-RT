"""Tests for buffer liveness + interference graph + greedy coloring."""

from __future__ import annotations

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    ExecutionPlan,
    Lifetime,
)
from compgen.runtime.liveness import (
    compute_interference_graph,
    compute_liveness,
    greedy_color,
)
from compgen.runtime.plan_builder import ExecutionPlanBuilder


def _mk(buffer_id, first, last, size=1024, space="hbm", persistent=False):
    return BufferDescriptor(
        buffer_id=buffer_id,
        size_bytes=size,
        memory_space=space,
        lifetime=Lifetime(first, last, persistent=persistent),
        ownership="exclusive",
        alias_of="",
    )


# --- compute_liveness ---------------------------------------------------------


def test_liveness_empty_plan():
    plan = ExecutionPlan(workload="t", target="c")
    rep = compute_liveness(plan)
    assert rep.per_buffer == {}
    assert rep.live_at == {}
    assert rep.peak_live_count == 0


def test_liveness_non_overlapping():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 5), _mk("b", 6, 10)]
    rep = compute_liveness(plan)
    assert 0 in rep.live_at and rep.live_at[0] == {"a"}
    assert 7 in rep.live_at and rep.live_at[7] == {"b"}
    assert rep.peak_live_count == 1


def test_liveness_overlapping_sums_bytes():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 10, size=1024), _mk("b", 5, 15, size=2048)]
    rep = compute_liveness(plan)
    assert rep.peak_live_count == 2
    assert rep.peak_live_bytes == 3072
    assert 5 <= rep.peak_tick <= 10


def test_liveness_persistent_stays_live():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [
        _mk("persist", 0, 0, size=512, persistent=True),
        _mk("ephemeral", 50, 60, size=256),
    ]
    rep = compute_liveness(plan)
    assert "persist" in rep.live_at[55]
    assert "ephemeral" in rep.live_at[55]


def test_liveness_lifetimes_overlap_helper():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 10), _mk("b", 11, 20)]
    rep = compute_liveness(plan)
    assert not rep.lifetimes_overlap("a", "b")
    plan.buffers = [_mk("a", 0, 15), _mk("b", 10, 20)]
    rep = compute_liveness(plan)
    assert rep.lifetimes_overlap("a", "b")


# --- compute_interference_graph -----------------------------------------------


def test_interference_graph_edges():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [
        _mk("a", 0, 10),
        _mk("b", 5, 15),  # overlaps a
        _mk("c", 20, 30),  # independent
    ]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    assert g.node_count == 3
    assert g.edge_count == 1
    assert "b" in g.neighbours("a")
    assert "c" not in g.neighbours("a")


def test_interference_graph_respects_memory_space():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [
        _mk("a", 0, 10, space="hbm"),
        _mk("b", 5, 15, space="scratchpad"),
    ]
    rep = compute_liveness(plan)
    g_restricted = compute_interference_graph(rep, only_same_memory_space=True)
    assert g_restricted.edge_count == 0

    g_all = compute_interference_graph(rep, only_same_memory_space=False)
    assert g_all.edge_count == 1


def test_interference_graph_persistent_forces_edges():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [
        _mk("persist", 0, 0, persistent=True),
        _mk("ephemeral", 100, 200),
    ]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    assert g.edge_count == 1


# --- greedy_color ------------------------------------------------------------


def test_greedy_color_independent_buffers_share_color():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 5), _mk("b", 10, 15)]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    colors = greedy_color(g)
    assert colors["a"] == colors["b"]


def test_greedy_color_overlapping_buffers_differ():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 10), _mk("b", 5, 15)]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    colors = greedy_color(g)
    assert colors["a"] != colors["b"]


def test_greedy_color_uses_at_most_chromatic_number():
    # Triangle: three mutually-overlapping buffers => 3 colors required.
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [
        _mk("a", 0, 20),
        _mk("b", 5, 25),
        _mk("c", 10, 30),
    ]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    colors = greedy_color(g)
    assert len(set(colors.values())) == 3


def test_greedy_color_is_deterministic():
    plan = ExecutionPlan(workload="t", target="c")
    plan.buffers = [_mk("a", 0, 10), _mk("b", 5, 15), _mk("c", 20, 30)]
    rep = compute_liveness(plan)
    g = compute_interference_graph(rep)
    c1 = greedy_color(g)
    c2 = greedy_color(g)
    assert c1 == c2


# --- integration with the builder --------------------------------------------


def test_liveness_from_builder_plan():
    plan = (
        ExecutionPlanBuilder("t", "c")
        .add_buffer("x", 1024, "hbm", 0, 10)
        .add_buffer("y", 512, "hbm", 5, 15)
        .add_buffer("z", 2048, "scratchpad", 5, 15)
        .build()
    )
    rep = compute_liveness(plan)
    assert set(rep.per_buffer) == {"x", "y", "z"}
    g = compute_interference_graph(rep)
    # x <-> y interfere; z is in a different space and should not.
    assert g.edge_count == 1
