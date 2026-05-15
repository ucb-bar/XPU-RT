"""Multi-step pass-plan executor MVP.

The agent's response can carry a ``pass_plan``: an ordered
list of pass steps. ships an executor that:

1. Validates the plan against 's invariants (structural / phase /
   requires_after / excludes). Invalid plans are rejected before any
   step runs.
2. Optionally applies step 0 by writing a single-step
   ``agent_decision_response.json`` that the existing
   ``selection_mode="agent-file"`` path can consume.
3. Logs every step's status to
   ``03_recipe_planning/pass_plan_execution_log.json``. Each step
   gets ``applied`` / ``deferred_to_future_run`` / ``rejected``.

What's deferred to Section 21+ (declared 's contract):

The pipeline today applies one transform per run starting from
``payload.mlir``. To execute step 1, the operator would need a
"continue from post-step-0 IR" mode where the second run starts from
the already-transformed payload. That requires either resumability
in ``run_graph_compilation`` or a copy-back step. explicitly
records ``deferred_to_future_run`` for steps 1+ so the agent doesn't
silently believe its full plan executed.

This MVP closes the residual ("multi-step pass plans aren't
executed yet") by shipping the executor contract, the validation,
the typed log, and a smoke-tested step-0 application path. The agent
proposing a 3-step plan today gets:
  step 0 → applied
  step 1 → deferred_to_future_run (operator runs the executor again
           with the post-step-0 IR as input — a Section 21+ workflow)
  step 2 → deferred_to_future_run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from compgen.audit.errors import (
    PairContractViolation,
    PassPlanInvalid,
    PhaseTransitionViolation,
)
from compgen.passes.cards import (
    PassCardRegistry,
    default_registry_root,
)
from compgen.passes.scheduler import (
    PassPlanStep,
    inspect_pass_plan,
)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Per-step result + per-plan log
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepExecutionResult:
    """One step's execution status."""

    step_index: int
    pass_id: str
    region_id: str
    candidate_id: str
    status: str  # applied | deferred_to_future_run | rejected
    detail: str = ""
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "pass_id": self.pass_id,
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "detail": self.detail,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class PassPlanExecutionLog:
    """Full execution log written to disk per plan."""

    schema_version: str = "pass_plan_execution_log_v1"
    overall: str = ""  # empty | rejected | validated_only | applied_step_0
    plan_decision_id: str = ""
    generated_at_utc: str = field(default_factory=_utc_now)
    plan_validation: dict[str, Any] = field(default_factory=dict)
    steps: list[StepExecutionResult] = field(default_factory=list)

    @property
    def applied_step_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "applied")

    @property
    def deferred_step_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "deferred_to_future_run")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overall": self.overall,
            "plan_decision_id": self.plan_decision_id,
            "generated_at_utc": self.generated_at_utc,
            "plan_validation": dict(self.plan_validation),
            "step_count": len(self.steps),
            "applied_step_count": self.applied_step_count,
            "deferred_step_count": self.deferred_step_count,
            "steps": [s.to_dict() for s in self.steps],
        }


def _write_log(log: PassPlanExecutionLog, *, run_dir: Path) -> Path:
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pass_plan_execution_log.json"
    out_path.write_text(
        json.dumps(log.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _write_response_for_step_zero(
    *,
    run_dir: Path,
    step: PassPlanStep,
    plan: Sequence[PassPlanStep],
    plan_decision_id: str,
) -> Path:
    """Synthesize agent_decision_response.json containing step-0's pick.

    The response carries:
    - ``selected_candidate_id`` — step 0's candidate (compatible with
      every pre-agent-file consumer).
    - ``pass_plan`` — the full plan list, so a future executor /
      validator can inspect intent (emission).
    """
    out_dir = run_dir / "03_recipe_planning" / "agent_decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent_decision_response.json"
    body: dict[str, Any] = {
        "schema_version": "agent_decision_response_v1",
        "selected_candidate_id": step.candidate_id,
        "rationale": dict(step.rationale or {"primary_reason":
            f"pass_plan executor: step 0 ({step.pass_id})"}),
        "pass_plan": [s.to_dict() for s in plan],
    }
    if plan_decision_id:
        body["decision_id"] = plan_decision_id
    out_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def execute_pass_plan(
    plan: Sequence[PassPlanStep] | Sequence[dict[str, Any]],
    *,
    run_dir: Path,
    apply_step_zero: bool = False,
    plan_decision_id: str = "",
    candidate_ids_allowed: Sequence[str] | None = None,
) -> PassPlanExecutionLog:
    """Validate + (optionally) apply a pass plan.

    Args:
        plan: Ordered list of :class:`PassPlanStep` (or dicts).
        run_dir: Run directory where the log + step-0 response are
            written.
        apply_step_zero: When True, write
            ``agent_decision_response.json`` carrying step 0's
            candidate so a follow-up
            ``run_graph_compilation(selection_mode="agent-file")``
            applies it. The executor itself does NOT spawn that run
            today — the operator does. ``apply_step_zero=False`` is
            the validation-only mode.
        plan_decision_id: Optional id propagated to the log + response.
        candidate_ids_allowed: Optional structural check — when
            provided, the plan's candidate_ids must be in this set.

    Returns:
        :class:`PassPlanExecutionLog`. Always written to disk;
        also returned to the caller.

    Raises:
        :class:`PassPlanInvalid` / :class:`PhaseTransitionViolation` /
        :class:`PairContractViolation` when the plan violates an
        invariant. The log is written to disk before the raise
        so the operator can inspect what failed.
    """
    coerced: list[PassPlanStep] = []
    for s in plan:
        if isinstance(s, PassPlanStep):
            coerced.append(s)
        else:
            coerced.append(PassPlanStep.from_dict(dict(s)))

    log = PassPlanExecutionLog(plan_decision_id=plan_decision_id)

    if not coerced:
        log.overall = "empty"
        log.plan_validation = {"holds": True, "structural_ok": True}
        _write_log(log, run_dir=run_dir)
        return log

    registry = PassCardRegistry.load(default_registry_root())
    report = inspect_pass_plan(
        coerced,
        registry=registry,
        candidate_ids_allowed=candidate_ids_allowed,
    )
    log.plan_validation = {
        "holds": report.holds,
        "structural_ok": report.structural_ok,
        "phase_ok": report.phase_ok,
        "requires_after_ok": report.requires_after_ok,
        "excludes_ok": report.excludes_ok,
        "structural_detail": report.structural_detail,
        "phase_detail": report.phase_detail,
        "requires_after_detail": report.requires_after_detail,
        "excludes_detail": report.excludes_detail,
    }

    if not report.holds:
        # Record every step as rejected with the first relevant
        # invariant detail; write the log; raise the most specific
        # typed error.
        for i, step in enumerate(coerced):
            log.steps.append(StepExecutionResult(
                step_index=i,
                pass_id=step.pass_id,
                region_id=step.region_id,
                candidate_id=step.candidate_id,
                status="rejected",
                detail=report.detail,
                timestamp_utc=_utc_now(),
            ))
        log.overall = "rejected"
        _write_log(log, run_dir=run_dir)
        if not report.structural_ok:
            raise PassPlanInvalid(report.structural_detail)
        if not report.phase_ok:
            raise PhaseTransitionViolation(report.phase_detail)
        if not report.requires_after_ok:
            raise PairContractViolation(report.requires_after_detail)
        if not report.excludes_ok:
            raise PairContractViolation(report.excludes_detail)
        raise PassPlanInvalid(report.detail)  # pragma: no cover (defensive)

    # Plan is valid. Optionally apply step 0; mark step 1+ deferred.
    for i, step in enumerate(coerced):
        if i == 0 and apply_step_zero:
            _write_response_for_step_zero(
                run_dir=run_dir,
                step=step,
                plan=coerced,
                plan_decision_id=plan_decision_id,
            )
            log.steps.append(StepExecutionResult(
                step_index=i,
                pass_id=step.pass_id,
                region_id=step.region_id,
                candidate_id=step.candidate_id,
                status="applied",
                detail=(
                    "agent_decision_response.json written; operator "
                    "drives run_graph_compilation(selection_mode='agent-file')"
                ),
                timestamp_utc=_utc_now(),
            ))
        else:
            log.steps.append(StepExecutionResult(
                step_index=i,
                pass_id=step.pass_id,
                region_id=step.region_id,
                candidate_id=step.candidate_id,
                status="deferred_to_future_run",
                detail=(
                    "M-37.6 MVP: only step 0 applies in one run; "
                    "step >= 1 requires pipeline resumability "
                    "(Section 21+)"
                ),
                timestamp_utc=_utc_now(),
            ))

    log.overall = "applied_step_0" if apply_step_zero else "validated_only"
    _write_log(log, run_dir=run_dir)
    return log
