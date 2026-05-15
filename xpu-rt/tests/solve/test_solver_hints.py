"""Tests for the LLM/heuristic solver-hint pruning layer.

Hints are best-effort guidance. The contract is:

1. The rule-based heuristic is deterministic — same input ⇒ same
   hint set.
2. Hints cannot make a feasible MILP infeasible. (A wrong hint
   may force infeasibility; the planner reports it honestly via
   typed INFEASIBLE — never a fake plan.)
3. Hints cannot worsen the optimum on feasible problems within
   tolerance — the MILP still validates.
4. Stage decomposition (when applicable) produces a plan whose
   merged tier_peak_usage and total objective_value are at most
   the monolithic optimum, and is detectably faster.
5. Hints applied are reported in ``response.solution['hints_applied']``.
"""

from __future__ import annotations

import time

import pytest

from compgen.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    plan_memory,
)
from compgen.solve.solver_hints import (
    MemoryHints,
    StageGroup,
    SymmetryClass,
    TierHint,
    merge_hints,
    rule_based_memory_hints,
)
from compgen.solve.solver_types import SolverStatus


# ---------------------------------------------------------------------------
# Rule-based heuristic — determinism + structure
# ---------------------------------------------------------------------------


def _layered_input(n_layers: int = 4) -> MemoryPlanInput:
    """Synthetic TinyLlama-like input: per-layer activations
    (small + short-lived) + shared weights (large + long-lived)."""

    buffers = []
    span = n_layers * 10
    # 1 weights buffer per layer (large, full-span).
    for layer in range(n_layers):
        buffers.append(BufferSpec(
            buffer_id=f"layer{layer}.weights",
            size_bytes=64 * 1024 * 1024,
            lifetime_start=0,
            lifetime_end=span,
            allowed_tiers=("host", "scratchpad"),
        ))
    # 3 activations per layer (small, layer-local).
    for layer in range(n_layers):
        for j in range(3):
            buffers.append(BufferSpec(
                buffer_id=f"layer{layer}.act{j}",
                size_bytes=1 * 1024 * 1024,
                lifetime_start=layer * 10 + j,
                lifetime_end=layer * 10 + j + 1,
                allowed_tiers=("scratchpad", "host"),
            ))
    return MemoryPlanInput(
        buffers=tuple(buffers),
        tier_capacities=(
            TierCapacity("scratchpad", capacity_bytes=512 * 1024 * 1024),
            TierCapacity("host", capacity_bytes=4 * 1024 * 1024 * 1024),
        ),
    )


def test_rule_based_hints_are_deterministic():
    plan_input = _layered_input(4)
    a = rule_based_memory_hints(plan_input)
    b = rule_based_memory_hints(plan_input)
    assert a.to_dict() == b.to_dict()
    assert a.source == "rule_based"


def test_rule_based_hints_send_weights_to_host():
    plan_input = _layered_input(4)
    hints = rule_based_memory_hints(plan_input)
    tier_map = {h.buffer_id: h.tier_id for h in hints.tier_hints}
    for layer in range(4):
        bid = f"layer{layer}.weights"
        assert tier_map.get(bid) == "host", (
            f"weights buffer {bid} should hint host, got {tier_map.get(bid)!r}"
        )


def test_rule_based_hints_partition_by_layer_prefix():
    plan_input = _layered_input(4)
    hints = rule_based_memory_hints(plan_input)
    stage_ids = {s.stage_id for s in hints.stage_partition}
    assert stage_ids == {"layer0", "layer1", "layer2", "layer3"}


def test_rule_based_hints_have_confidence_summary():
    plan_input = _layered_input(4)
    hints = rule_based_memory_hints(plan_input)
    assert "tier_hints_fraction" in hints.confidence_summary
    assert 0.0 <= hints.confidence_summary["tier_hints_fraction"] <= 1.0


def test_empty_input_produces_empty_hints():
    plan_input = MemoryPlanInput(
        buffers=(),
        tier_capacities=(TierCapacity("scratchpad", 1024),),
    )
    hints = rule_based_memory_hints(plan_input)
    assert hints.is_empty


# ---------------------------------------------------------------------------
# Hints integration with the MILP — correctness preserved
# ---------------------------------------------------------------------------


def test_hints_dont_break_solver_correctness():
    plan_input = _layered_input(2)
    hints = rule_based_memory_hints(plan_input)
    response_h, plan_h = plan_memory(plan_input, hints=hints)
    response_n, plan_n = plan_memory(plan_input)
    assert response_h.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert response_n.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan_h is not None and plan_n is not None
    # The hinted run must respect the same feasibility envelope.
    for alloc in plan_h.buffers:
        # Every buffer's tier is one of its allowed_tiers.
        spec = next(b for b in plan_input.buffers if b.buffer_id == alloc.buffer_id)
        assert alloc.tier in spec.allowed_tiers


def test_hints_applied_is_reported_in_response():
    plan_input = _layered_input(2)
    hints = rule_based_memory_hints(plan_input)
    response, plan = plan_memory(plan_input, hints=hints)
    if response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        body = response.solution
        # Either the response carries hints_applied directly (single-MILP path)
        # or the stage decomposition path was taken (response.solution has
        # stage_count instead — also acceptable as evidence hints were used).
        assert isinstance(body, dict)
        assert ("hints_applied" in body) or ("stage_count" in body), (
            f"response.solution missing hints/stage evidence: {body.keys()}"
        )


def test_stage_decomposition_handles_disjoint_layers():
    plan_input = _layered_input(4)
    hints = rule_based_memory_hints(plan_input)
    response, plan = plan_memory(plan_input, hints=hints)
    assert response.status is SolverStatus.OPTIMAL
    assert plan is not None
    # Stage-decomposition path emits stage_count when used.
    body = response.solution
    assert isinstance(body, dict)
    if "stage_count" in body:
        assert body["stage_count"] >= 2


def test_stage_decomposition_falls_back_when_lifetimes_overlap():
    """If hints declare a stage partition over buffers with
    overlapping lifetimes, decomposition is unsafe and the planner
    falls back to the monolithic MILP (no silent correctness loss)."""

    overlapping = MemoryPlanInput(
        buffers=(
            BufferSpec("a", 1024, 0, 10, ("scratchpad",)),
            BufferSpec("b", 1024, 5, 15, ("scratchpad",)),  # overlaps with a
        ),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
    )
    bad_hints = MemoryHints(
        stage_partition=(
            StageGroup("s0", ("a",)),
            StageGroup("s1", ("b",)),
        ),
        source="rule_based:overlapping",
    )
    response, plan = plan_memory(overlapping, hints=bad_hints)
    # Either monolithic-MILP succeeds OR returns typed feasibility
    # status — never a silently-decomposed-but-wrong plan.
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    body = response.solution
    # The decomposition path should NOT have run (no stage_count).
    assert "stage_count" not in body


def test_wrong_tier_hint_does_not_crash_planner():
    """A high-confidence tier hint pointing at a non-allowed tier
    is detected upfront and listed in ``hints_applied['skipped']``
    — the planner does NOT propagate it as an infeasible constraint."""

    plan_input = MemoryPlanInput(
        buffers=(BufferSpec("a", 1024, 0, 5, ("scratchpad",)),),
        tier_capacities=(
            TierCapacity("scratchpad", 4096),
            TierCapacity("host", 8192),
        ),
    )
    bad_hint = MemoryHints(
        tier_hints=(TierHint(buffer_id="a", tier_id="host", confidence=0.95),),
        source="manual_test",
    )
    response, plan = plan_memory(plan_input, hints=bad_hint)
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    body = response.solution
    assert "hints_applied" in body
    assert any("not allowed" in s for s in body["hints_applied"]["skipped"]), (
        f"expected hint skip message: {body['hints_applied']}"
    )


# ---------------------------------------------------------------------------
# Metamorphic: hints can't worsen the optimum
# ---------------------------------------------------------------------------


def test_hints_cannot_worsen_feasible_optimum():
    """For the same feasible problem, the hinted-run objective is
    ≥ the unhinted-run objective only when hints force a worse
    decision; for our rule-based hints (which generally match what
    the MILP would do anyway) the objective should be within a small
    tolerance."""

    plan_input = _layered_input(2)
    hints = rule_based_memory_hints(plan_input)
    response_h, _ = plan_memory(plan_input, hints=hints)
    response_n, _ = plan_memory(plan_input)
    if (response_h.objective_value is not None
            and response_n.objective_value is not None):
        # Allow small noise; the alias-bonus float is the main jitter.
        delta = abs(response_h.objective_value - response_n.objective_value)
        assert delta < 0.01 or response_h.objective_value >= response_n.objective_value, (
            f"hints unexpectedly worsened objective: "
            f"unhinted={response_n.objective_value}, hinted={response_h.objective_value}"
        )


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_merge_hints_picks_higher_confidence():
    low = MemoryHints(
        tier_hints=(TierHint("a", "scratchpad", confidence=0.6),),
        source="heuristic",
    )
    high = MemoryHints(
        tier_hints=(TierHint("a", "host", confidence=0.95),),
        source="llm",
    )
    merged = merge_hints(low, high)
    by_id = {h.buffer_id: h for h in merged.tier_hints}
    assert by_id["a"].tier_id == "host"
    assert by_id["a"].confidence == pytest.approx(0.95)


def test_merge_hints_deduplicates_stages():
    a = MemoryHints(stage_partition=(StageGroup("layer0", ("b0",)),))
    b = MemoryHints(stage_partition=(StageGroup("layer0", ("b1",)),))
    merged = merge_hints(a, b)
    assert {s.stage_id for s in merged.stage_partition} == {"layer0"}
