"""Tests for the P2.5 Strategist + Tactician orchestrator.

End-to-end loop:

  Strategist.plan_session  →  Plan
                                ↓
                       for each region:
                         Tactician.pick_edit  →  TacticianDecision

On rejection the orchestrator calls
:func:`compgen.agent.plan.replan_on_reject` and re-invokes the
Tactician. Tests assert the full chain produces deterministic typed
outputs.
"""

from __future__ import annotations

import pytest
from compgen.agent.cost_preview import CandidateInput, compute_cost_previews
from compgen.agent.plan import (
    EXHAUSTED_TACTIC,
    Budget,
    RegionPlan,
    replan_on_reject,
)
from compgen.agent.strategist import (
    DEFAULT_FALLBACK_LADDER,
    DossierRegion,
    StrategistInput,
    plan_session,
)
from compgen.agent.tactician import TacticianDecision, pick_edit

# ---------- Strategist ----------------------------------------------


def test_plan_session_produces_a_region_per_dossier_region():
    inputs = StrategistInput(
        session_id="ses_1",
        global_objective="minimize_p50_latency",
        budget=Budget(compile_seconds=600.0, llm_dollars=2.50),
        regions=(
            DossierRegion(region_id="017", op_family="matmul"),
            DossierRegion(region_id="023", op_family="attention"),
        ),
    )
    plan = plan_session(inputs)
    assert plan.plan_version == 0
    assert {r.region_id for r in plan.region_partition} == {"017", "023"}


def test_plan_session_uses_objective_default_tactic():
    inputs = StrategistInput(
        session_id="ses_1",
        global_objective="minimize_p50_latency",
        budget=Budget(compile_seconds=600.0, llm_dollars=2.50),
        regions=(DossierRegion(region_id="r"),),
    )
    plan = plan_session(inputs)
    assert plan.get_region("r").tactic == "fuse"  # p50-latency default
    # Ladder is rooted at the picked tactic.
    assert plan.get_region("r").fallback_ladder[0] == "fuse"


def test_plan_session_honors_suggested_tactic_when_present():
    inputs = StrategistInput(
        session_id="ses_1",
        global_objective="minimize_p50_latency",
        budget=Budget(compile_seconds=600.0, llm_dollars=2.50),
        regions=(
            DossierRegion(region_id="r", suggested_tactic="megakernel"),
        ),
    )
    plan = plan_session(inputs)
    assert plan.get_region("r").tactic == "megakernel"
    # Megakernel head + default ladder appended (deduplicated)
    ladder = plan.get_region("r").fallback_ladder
    assert ladder[0] == "megakernel"
    assert "naive_sync" in ladder


def test_plan_session_ladder_always_non_empty():
    """Every region's ladder has at least naive_sync."""

    inputs = StrategistInput(
        session_id="ses",
        global_objective="correctness_only",
        budget=Budget(compile_seconds=10.0, llm_dollars=0.10),
        regions=(DossierRegion(region_id="r"),),
    )
    plan = plan_session(inputs)
    assert "naive_sync" in plan.get_region("r").fallback_ladder


def test_default_fallback_ladder_constant():
    assert "naive_sync" in DEFAULT_FALLBACK_LADDER
    assert DEFAULT_FALLBACK_LADDER[0] == "fuse"


# ---------- Tactician ----------------------------------------------


def _region(rung: str = "fuse") -> RegionPlan:
    return RegionPlan(
        region_id="017",
        tactic=rung,
        fallback_ladder=("fuse", "tile_only", "naive_sync"),
    )


def _previews(*entries):
    cands = [
        CandidateInput(candidate_id=cid, delta_static=cost, legality=leg)
        for cid, cost, leg in entries
    ]
    return compute_cost_previews(cands)


def test_pick_edit_returns_lowest_cost_survivor():
    decision = pick_edit(
        _region("fuse"),
        cost_previews=_previews(
            ("a", 10.0, "ok"),
            ("b", 5.0, "ok"),
            ("c", 20.0, "ok"),
        ),
    )
    assert decision.next_action == "apply"
    assert decision.chosen_candidate_id == "b"


def test_pick_edit_breaks_ties_by_candidate_id():
    decision = pick_edit(
        _region("fuse"),
        cost_previews=_previews(("zeta", 5.0, "ok"), ("alpha", 5.0, "ok")),
    )
    assert decision.chosen_candidate_id == "alpha"


def test_pick_edit_escalates_when_no_legal_candidates():
    decision = pick_edit(
        _region("fuse"),
        cost_previews=_previews(("a", 1.0, "blocked")),
    )
    assert decision.next_action == "escalate"
    assert decision.chosen_candidate_id is None


def test_pick_edit_exhausted_region():
    region = RegionPlan(
        region_id="017",
        tactic=EXHAUSTED_TACTIC,
        fallback_ladder=("fuse",),
    )
    decision = pick_edit(region, cost_previews=_previews(("a", 1.0, "ok")))
    assert decision.next_action == "exhausted"
    assert decision.chosen_candidate_id is None


def test_pick_edit_skips_dominated_candidates():
    """If all-but-one candidate is dominated, the survivor wins."""

    decision = pick_edit(
        _region("fuse"),
        cost_previews=_previews(
            ("a", 10.0, "ok"),
            ("b", 5.0, "ok"),  # dominates a and c
            ("c", 100.0, "ok"),
        ),
    )
    assert decision.chosen_candidate_id == "b"


# ---------- End-to-end -------------------------------------------------


def test_loop_strategist_replan_tactician():
    """Drive a tiny loop: Strategist plans → Tactician picks →
    rejection → replan → Tactician picks the next rung's edit."""

    inputs = StrategistInput(
        session_id="ses_loop",
        global_objective="minimize_p50_latency",
        budget=Budget(compile_seconds=60.0, llm_dollars=0.10),
        regions=(DossierRegion(region_id="017", op_family="matmul"),),
    )
    plan = plan_session(inputs)
    assert plan.get_region("017").tactic == "fuse"

    # Tactician picks an edit.
    decision = pick_edit(
        plan.get_region("017"),
        cost_previews=_previews(("a", 1.0, "ok"), ("b", 2.0, "ok")),
    )
    assert decision.next_action == "apply"
    assert decision.chosen_candidate_id == "a"

    # Verifier (simulated) rejects with tactic_fatal → walk ladder.
    plan2 = replan_on_reject(plan, region_id="017", rejection_class="tactic_fatal")
    assert plan2.get_region("017").tactic == "tile_only"
    assert plan2.plan_version == plan.plan_version + 1

    # Tactician picks again on the new rung.
    decision2 = pick_edit(
        plan2.get_region("017"),
        cost_previews=_previews(("a", 1.0, "ok"), ("b", 2.0, "ok")),
    )
    assert decision2.next_action == "apply"
    assert decision2.chosen_candidate_id == "a"


# ---------- Negative controls --------------------------------------


def test_tactician_decision_unknown_action_rejected():
    from compgen.agent.tactician import TacticianError

    with pytest.raises(TacticianError, match="next_action"):
        TacticianDecision(
            next_action="totally_made_up",
            chosen_candidate_id=None,
            reason="x",
        )
