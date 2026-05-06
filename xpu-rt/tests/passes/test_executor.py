"""Tests for compgen.passes.executor (M-37.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.errors import (
    PairContractViolation,
    PassPlanInvalid,
    PhaseTransitionViolation,
)
from compgen.passes.executor import (
    PassPlanExecutionLog,
    StepExecutionResult,
    execute_pass_plan,
)
from compgen.passes.scheduler import PassPlanStep


# --------------------------------------------------------------------------- #
# Empty / structural
# --------------------------------------------------------------------------- #


def test_empty_plan_logs_empty(tmp_path: Path) -> None:
    log = execute_pass_plan([], run_dir=tmp_path)
    assert log.overall == "empty"
    assert log.steps == []
    out = tmp_path / "03_recipe_planning" / "pass_plan_execution_log.json"
    assert out.exists()


def test_unknown_pass_id_raises_invalid(tmp_path: Path) -> None:
    plan = [PassPlanStep(pass_id="totally_made_up", region_id="r0")]
    with pytest.raises(PassPlanInvalid):
        execute_pass_plan(plan, run_dir=tmp_path)
    out = tmp_path / "03_recipe_planning" / "pass_plan_execution_log.json"
    assert out.exists()
    raw = json.loads(out.read_text())
    assert raw["overall"] == "rejected"


def test_phase_violation_raises(tmp_path: Path) -> None:
    """M-34.1: emit-phase pass before optimize-phase pass fails phase ordering."""
    # event_static_schedule is plan-level / event_tensor family →
    # phase=emit. set_tile_params is tiling family → phase=optimize.
    plan = [
        PassPlanStep(pass_id="event_static_schedule", region_id="r0"),
        PassPlanStep(pass_id="set_tile_params", region_id="r1"),
    ]
    with pytest.raises(PhaseTransitionViolation):
        execute_pass_plan(plan, run_dir=tmp_path)
    out = tmp_path / "03_recipe_planning" / "pass_plan_execution_log.json"
    assert out.exists()
    raw = json.loads(out.read_text())
    assert raw["overall"] == "rejected"
    assert raw["plan_validation"]["phase_ok"] is False


def test_valid_plan_logged_as_validated_only(tmp_path: Path) -> None:
    """Without apply_step_zero, all steps are deferred — but the plan
    is validated and logged."""
    plan = [
        PassPlanStep(pass_id="set_tile_params", region_id="m0", candidate_id="c0"),
        PassPlanStep(pass_id="event_static_schedule", region_id="r0"),
    ]
    log = execute_pass_plan(plan, run_dir=tmp_path)
    assert log.overall == "validated_only"
    assert len(log.steps) == 2
    assert log.steps[0].status == "deferred_to_future_run"
    assert log.steps[1].status == "deferred_to_future_run"
    out = tmp_path / "03_recipe_planning" / "pass_plan_execution_log.json"
    raw = json.loads(out.read_text())
    assert raw["plan_validation"]["phase_ok"] is True
    assert raw["plan_validation"]["structural_ok"] is True


# --------------------------------------------------------------------------- #
# apply_step_zero=True
# --------------------------------------------------------------------------- #


def test_apply_step_zero_writes_response_file(tmp_path: Path) -> None:
    plan = [
        PassPlanStep(
            pass_id="set_tile_params",
            region_id="matmul_0",
            candidate_id="tile_M16_N16_K16",
            rationale={"primary_reason": "test"},
        ),
        PassPlanStep(pass_id="event_static_schedule", region_id="event_0"),
    ]
    log = execute_pass_plan(plan, run_dir=tmp_path, apply_step_zero=True)
    assert log.overall == "applied_step_0"
    assert log.steps[0].status == "applied"
    assert log.steps[1].status == "deferred_to_future_run"
    response_path = (
        tmp_path / "03_recipe_planning" / "agent_decision"
        / "agent_decision_response.json"
    )
    assert response_path.exists()
    body = json.loads(response_path.read_text())
    assert body["selected_candidate_id"] == "tile_M16_N16_K16"
    assert body["pass_plan"][0]["pass_id"] == "set_tile_params"


def test_apply_step_zero_then_resume_in_new_dir(tmp_path: Path) -> None:
    """Operator can run step 0 in run_dir_a, then point the executor at
    run_dir_b for step 1 (the deferred row of plan)."""
    plan = [
        PassPlanStep(
            pass_id="set_tile_params",
            region_id="matmul_0",
            candidate_id="tile_M16_N16_K16",
        ),
        PassPlanStep(
            pass_id="event_static_schedule",
            region_id="event_0",
            candidate_id="schedule_static_0",
        ),
    ]
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    log_a = execute_pass_plan(plan, run_dir=run_a, apply_step_zero=True)
    assert log_a.steps[0].status == "applied"
    assert log_a.steps[1].status == "deferred_to_future_run"

    # Re-run the plan in a fresh dir; same shape — the executor doesn't
    # carry state between runs (Section 21's job).
    log_b = execute_pass_plan(plan, run_dir=run_b, apply_step_zero=True)
    assert log_b.steps[0].status == "applied"
    assert log_b.steps[1].status == "deferred_to_future_run"


# --------------------------------------------------------------------------- #
# Log shape
# --------------------------------------------------------------------------- #


def test_log_to_dict_round_trip(tmp_path: Path) -> None:
    plan = [PassPlanStep(pass_id="set_tile_params", region_id="r0", candidate_id="c0")]
    log = execute_pass_plan(plan, run_dir=tmp_path)
    raw = log.to_dict()
    assert raw["schema_version"] == "pass_plan_execution_log_v1"
    assert raw["step_count"] == 1
    assert raw["applied_step_count"] == 0  # apply_step_zero=False
    assert raw["deferred_step_count"] == 1


def test_step_execution_result_to_dict() -> None:
    s = StepExecutionResult(
        step_index=0,
        pass_id="set_tile_params",
        region_id="m0",
        candidate_id="c0",
        status="applied",
    )
    raw = s.to_dict()
    assert raw["step_index"] == 0
    assert raw["status"] == "applied"


def test_plan_decision_id_propagated(tmp_path: Path) -> None:
    plan = [PassPlanStep(pass_id="set_tile_params", region_id="r0", candidate_id="c0")]
    log = execute_pass_plan(plan, run_dir=tmp_path, plan_decision_id="abc123")
    raw = log.to_dict()
    assert raw["plan_decision_id"] == "abc123"
