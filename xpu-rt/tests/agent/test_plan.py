"""Tests for the P2.4 Plan / replan-on-reject module."""

from __future__ import annotations

import pytest
from compgen.agent.plan import (
    EXHAUSTED_TACTIC,
    GLOBAL_OBJECTIVES,
    Budget,
    Plan,
    PlanError,
    RegionPlan,
    replan_on_reject,
)


def _plan(**overrides):
    region_partition = overrides.pop("region_partition", (
        RegionPlan(
            region_id="017",
            tactic="fuse",
            fallback_ladder=("fuse", "tile_only", "naive_sync"),
        ),
        RegionPlan(
            region_id="023",
            tactic="megakernel",
            fallback_ladder=("megakernel", "fused_chain", "naive_async"),
        ),
    ))
    return Plan(
        session_id=overrides.get("session_id", "ses_1"),
        plan_version=overrides.get("plan_version", 1),
        global_objective=overrides.get("global_objective", "minimize_p50_latency"),
        budget=overrides.get("budget", Budget(compile_seconds=600.0, llm_dollars=2.50)),
        region_partition=region_partition,
    )


# ---------- Positive --------------------------------------------------


def test_plan_constructs_with_closed_enum_objective():
    p = _plan()
    assert p.global_objective == "minimize_p50_latency"
    assert p.plan_version == 1
    assert len(p.region_partition) == 2


def test_plan_to_dict_roundtrips():
    body = _plan().to_dict()
    assert body["global_objective"] == "minimize_p50_latency"
    assert body["budget"]["llm_dollars"] == 2.50
    assert body["region_partition"][0]["tactic"] == "fuse"


def test_get_region_returns_matching_entry():
    p = _plan()
    assert p.get_region("017").tactic == "fuse"


def test_global_objectives_enum_closed():
    assert "minimize_p50_latency" in GLOBAL_OBJECTIVES
    assert "correctness_only" in GLOBAL_OBJECTIVES


# ---------- replan: tactic_fatal -------------------------------------


def test_tactic_fatal_advances_to_next_rung():
    p = _plan()
    next_plan = replan_on_reject(p, region_id="017", rejection_class="tactic_fatal")
    assert next_plan.plan_version == p.plan_version + 1
    new_region = next_plan.get_region("017")
    assert new_region.tactic == "tile_only"
    assert new_region.escalated is False  # tactic_fatal alone doesn't escalate
    # The other region is unchanged.
    assert next_plan.get_region("023").tactic == "megakernel"


def test_tactic_fatal_at_last_rung_marks_exhausted():
    """Walk to the bottom of the ladder and confirm the typed exhausted state."""

    p = _plan()
    p1 = replan_on_reject(p, region_id="017", rejection_class="tactic_fatal")  # fuse → tile_only
    p2 = replan_on_reject(p1, region_id="017", rejection_class="tactic_fatal")  # tile_only → naive_sync
    p3 = replan_on_reject(p2, region_id="017", rejection_class="tactic_fatal")  # naive_sync → exhausted
    r = p3.get_region("017")
    assert r.tactic == EXHAUSTED_TACTIC
    assert r.is_exhausted


# ---------- replan: tactic_recoverable -------------------------------


def test_tactic_recoverable_returns_unchanged_plan():
    p = _plan()
    same = replan_on_reject(p, region_id="017", rejection_class="tactic_recoverable")
    assert same is p  # identity — no new Plan allocated


# ---------- replan: surprising ---------------------------------------


def test_surprising_drops_rung_and_escalates():
    p = _plan()
    next_plan = replan_on_reject(p, region_id="017", rejection_class="surprising")
    r = next_plan.get_region("017")
    assert r.tactic == "tile_only"
    assert r.escalated is True


# ---------- Negative controls ----------------------------------------


def test_unknown_global_objective_rejected():
    with pytest.raises(PlanError, match="global_objective"):
        Plan(
            session_id="x",
            plan_version=0,
            global_objective="not_a_real_objective",
            budget=Budget(compile_seconds=0.0, llm_dollars=0.0),
        )


def test_duplicate_region_id_rejected():
    with pytest.raises(PlanError, match="duplicate region_id"):
        Plan(
            session_id="x",
            plan_version=0,
            global_objective="minimize_p50_latency",
            budget=Budget(compile_seconds=0.0, llm_dollars=0.0),
            region_partition=(
                RegionPlan(region_id="r", tactic="t", fallback_ladder=("t",)),
                RegionPlan(region_id="r", tactic="t", fallback_ladder=("t",)),
            ),
        )


def test_negative_plan_version_rejected():
    with pytest.raises(PlanError, match="plan_version"):
        Plan(
            session_id="x",
            plan_version=-1,
            global_objective="minimize_p50_latency",
            budget=Budget(compile_seconds=0.0, llm_dollars=0.0),
        )


def test_unknown_region_in_replan_rejected():
    with pytest.raises(PlanError, match="not present in plan"):
        replan_on_reject(_plan(), region_id="999", rejection_class="tactic_fatal")


def test_unknown_rejection_class_rejected():
    with pytest.raises(PlanError, match="rejection_class"):
        replan_on_reject(_plan(), region_id="017", rejection_class="bogus")


def test_negative_budget_rejected():
    with pytest.raises(PlanError, match="compile_seconds"):
        Budget(compile_seconds=-1.0, llm_dollars=0.0)
    with pytest.raises(PlanError, match="llm_dollars"):
        Budget(compile_seconds=0.0, llm_dollars=-1.0)


def test_empty_tactic_rejected():
    with pytest.raises(PlanError, match="tactic"):
        RegionPlan(region_id="r", tactic="", fallback_ladder=("a",))


def test_empty_region_id_rejected():
    with pytest.raises(PlanError, match="region_id"):
        RegionPlan(region_id="", tactic="t", fallback_ladder=("t",))
