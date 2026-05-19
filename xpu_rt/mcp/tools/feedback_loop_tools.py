"""MCP tool wrappers for the Stage-4 feedback-loop driver.

Four typed tools wrap :mod:`xpu_rt.scheduling.feedback_loop` for the
``xpu-rt-compile`` skill (and any other MCP client):

* :func:`xpu_rt_recommend_granularity` — Stage-3 specialty chunking +
  Stage-2 solver pick, no measurement required.
* :func:`xpu_rt_feedback_step` — One iteration of the loop. Initialises
  the state from scratch when ``loop_state_dict is None``.
* :func:`xpu_rt_apply_measurement` — Stage-1 EMA update only; no re-solve.
  Useful for inspecting calibration drift before deciding to re-step.
* :func:`xpu_rt_loop_status` — Read-only status; loads the persisted state
  from ``build/loops/<workload>__<target>.json`` if any.

All four tools accept and return JSON-typed dicts (no Python objects on
the MCP wire). Persistence lives under the gitignored ``build/loops/``
directory; pass ``persist=False`` to opt out.

Tool dicts are registered through :data:`FEEDBACK_LOOP_TOOLS` and joined
into ``ALL_TOOLS`` via :mod:`xpu_rt.mcp.tools` once the entry is added
there. The wiring follows the same shape as the other ``xpu_rt_qnn_*``
groups (``QNN_FLOW_TOOLS`` / ``QNN_GRANULARITY_TOOLS``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from xpu_rt.mcp.session import SessionManager
from xpu_rt.runtime.calibration import load as load_calibration
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix, make_chain_dag
from xpu_rt.scheduling.feedback_loop import (
    LoopState,
    has_converged,
    init_loop_state,
    load_loop_state,
    measurement_from_dict,
    save_loop_state,
    state_from_dict,
    state_to_dict,
    step,
)
from xpu_rt.scheduling.granularity import compute_specialty_matrix, propose_chunks
from xpu_rt.scheduling.multi_rate import analyze as analyze_multi_rate
from xpu_rt.scheduling.objectives import (
    MultiObjectiveSpec,
    ObjectiveKind,
    ObjectiveWeight,
    ScheduleMetrics,
    evaluate,
)
from xpu_rt.scheduling.policy import SchedulerPolicy

log = structlog.get_logger(__name__)

DEFAULT_TARGET_ID = "qrb5165"
DEFAULT_LOOPS_DIR = Path("build") / "loops"


def _state_path(workload_id: str, target_id: str) -> Path:
    return DEFAULT_LOOPS_DIR / f"{workload_id}__{target_id}.json"


def _load_calibration_for(target_id: str) -> Any:
    """Load the seeded calibration model for ``target_id``.

    Looks at ``xpu-rt/data/calibration/<target_id>.json`` relative to the
    repo root (same path Stage 1 writes to). Falls back to a synthetic
    zero-overhead model for tests and dry-runs.
    """

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "xpu-rt" / "data" / "calibration" / f"{target_id}.json"
    if candidate.is_file():
        return load_calibration(candidate)
    # Synthetic fallback so the tools work in untrained targets.
    from datetime import UTC, datetime

    from xpu_rt.runtime.calibration import (
        CALIBRATION_SCHEMA_VERSION,
        CalibrationModel,
    )

    return CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id=target_id,
        overhead_us={},
        contention_factor={},
        history=(),
        created_at=datetime.now(UTC).isoformat(),
        contention_provenance={},
    )


def xpu_rt_recommend_granularity(
    sm: SessionManager,  # noqa: ARG001
    *,
    workload_id: str,
    cost_matrix_path: str,
    target_id: str = DEFAULT_TARGET_ID,
    max_chunk_ops: int = 16,
    max_partitions: int = 200,
) -> dict[str, Any]:
    """Recommend a specialty-driven chunking + solver choice.

    Args:
        sm: MCP session manager (unused).
        workload_id: Workload key in the cost matrix.
        cost_matrix_path: Path to the loaded cost matrix JSON.
        target_id: Target identifier (selects the calibration model).
        max_chunk_ops: Hard cap on chunk size; defaults to 16 (Stage 3).
        max_partitions: Hard cap on chunk count; defaults to 200 to match
            CP-SAT's policy ceiling.

    Returns:
        ``{"ok", "chunks", "solver_choice", "n_partitions", "specialty",
        "reason"}``. ``chunks`` is a JSON-friendly list of chunk dicts.
    """

    cost_matrix = load_cost_matrix(cost_matrix_path)
    dag = make_chain_dag(workload_id, cost_matrix=cost_matrix)
    specialty = compute_specialty_matrix(cost_matrix, workload_id)
    plan = propose_chunks(
        dag,
        cost_matrix,
        workload_id,
        specialty,
        max_chunk_ops=max_chunk_ops,
        max_partitions=max_partitions,
    )
    policy = SchedulerPolicy()
    choice = policy.choose(n_partitions=plan.n_partitions)

    chunks_payload = [
        {
            "chunk_id": c.chunk_id,
            "op_ids": list(c.op_ids),
            "preferred_backend": c.preferred_backend,
            "n_ops": len(c.op_ids),
        }
        for c in plan.chunks
    ]
    return {
        "ok": True,
        "workload_id": workload_id,
        "target_id": target_id,
        "chunks": chunks_payload,
        "solver_choice": str(choice),
        "n_partitions": plan.n_partitions,
        "specialty": dict(plan.specialty_summary),
        "reason": policy.reason(plan.n_partitions),
    }


def xpu_rt_feedback_step(
    sm: SessionManager,  # noqa: ARG001
    *,
    cost_matrix_path: str,
    workload_id: str | None = None,
    target_id: str = DEFAULT_TARGET_ID,
    loop_state_dict: dict[str, Any] | None = None,
    measurement_dict: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Wrap :func:`xpu_rt.scheduling.feedback_loop.step`.

    If ``loop_state_dict`` is ``None``, an initial :class:`LoopState` is
    built from the seeded calibration model for ``target_id`` and a
    plan-only first iteration is run (so the caller gets a baseline
    prediction without supplying a measurement).

    Args:
        sm: MCP session manager (unused).
        cost_matrix_path: Path to the cost matrix JSON.
        workload_id: Required when ``loop_state_dict`` is ``None``.
        target_id: Target identifier (defaults to QRB5165).
        loop_state_dict: Existing loop state, JSON-typed.
        measurement_dict: Optional measurement to absorb this iteration.
        persist: When True, the new state is written to
            ``build/loops/<workload>__<target>.json`` (gitignored).

    Returns:
        ``{"ok", "state": <state_dict>, "converged": bool, "iteration": int,
        "decision_next": str, "predicted_us": float | None}``.
    """

    cost_matrix = load_cost_matrix(cost_matrix_path)

    state: LoopState
    if loop_state_dict is None:
        if not workload_id:
            return {
                "ok": False,
                "error": "workload_id required when loop_state_dict is None",
            }
        cal = _load_calibration_for(target_id)
        state = init_loop_state(
            workload_id=workload_id,
            target_id=target_id,
            cost_matrix=cost_matrix,
            calibration=cal,
        )
    else:
        state = state_from_dict(loop_state_dict)

    measurement = (
        measurement_from_dict(measurement_dict) if measurement_dict else None
    )
    new_state = step(state, measurement, cost_matrix=cost_matrix)

    if persist:
        save_loop_state(new_state, _state_path(new_state.workload_id, new_state.target_id))

    last_round = new_state.history[-1] if new_state.history else None
    return {
        "ok": True,
        "state": state_to_dict(new_state),
        "converged": has_converged(new_state),
        "iteration": new_state.iteration,
        "status": new_state.status,
        "decision_next": last_round.decision_next if last_round else None,
        "predicted_us": new_state.current_predicted_makespan_us,
    }


def xpu_rt_apply_measurement(
    sm: SessionManager,  # noqa: ARG001
    *,
    loop_state_dict: dict[str, Any],
    measurement_dict: dict[str, Any],
    persist: bool = True,
) -> dict[str, Any]:
    """Apply one calibration EMA update without re-scheduling.

    Lets the agent inspect the new calibration before deciding to re-step.
    The returned state has the same ``iteration`` / ``status`` / chunks
    as the input — only ``current_calibration`` changes.
    """

    from xpu_rt.runtime.calibration import update_from_measurement

    state = state_from_dict(loop_state_dict)
    measurement = measurement_from_dict(measurement_dict)
    new_cal = update_from_measurement(state.current_calibration, measurement)
    new_state = LoopState(
        schema_version=state.schema_version,
        workload_id=state.workload_id,
        target_id=state.target_id,
        iteration=state.iteration,
        current_calibration=new_cal,
        current_chunks=state.current_chunks,
        current_solver_choice=state.current_solver_choice,
        current_predicted_makespan_us=state.current_predicted_makespan_us,
        current_max_chunk_ops=state.current_max_chunk_ops,
        history=state.history,
        status=state.status,
        consecutive_in_band=state.consecutive_in_band,
        error_history=state.error_history,
        decision_history=state.decision_history,
        baseline_makespan_us=state.baseline_makespan_us,
    )
    if persist:
        save_loop_state(new_state, _state_path(new_state.workload_id, new_state.target_id))
    return {
        "ok": True,
        "state": state_to_dict(new_state),
        "calibration_overhead_us": {
            wid: dict(per_b) for wid, per_b in new_cal.overhead_us.items()
        },
    }


def xpu_rt_recommend_multiplicity(
    sm: SessionManager,  # noqa: ARG001
    *,
    workload_ids: list[str],
    cost_matrix_path: str,
    calibration_path: str | None = None,
) -> dict[str, Any]:
    """Recommend dominant workload + per-secondary multiplicity.

    Wraps :func:`xpu_rt.scheduling.multi_rate.analyze`. Use this before
    Stage-3 chunking to size the joint partition set from data instead
    of a hand-tuned constant.

    Args:
        sm: MCP session manager (unused).
        workload_ids: Workload keys to compare; all must appear in the
            cost matrix.
        cost_matrix_path: Path to ``qnn_cost_matrix.json``.
        calibration_path: Optional path to a calibration model JSON whose
            ``overhead_us`` field is added once per workload-cycle.

    Returns:
        ``{"ok", "dominant_workload_id", "dominant_period_us", "rates",
        "lane_availability", "notes"}``. ``rates`` is a list of dicts
        with ``workload_id``, ``period_us``, ``multiplicity``,
        ``preferred_lane``, ``primary_lane_busy_us``,
        ``achievable_frequency_hz``. ``lane_availability`` lists per-lane
        ``busy_us``, ``idle_us``, ``busy_fraction``.
    """

    cost_matrix = load_cost_matrix(cost_matrix_path)
    # v2 calibration is per-(workload, backend); multi_rate.analyze still
    # takes a single per-backend dict. We compose a workload-aware
    # analysis by picking each workload's own overhead when computing its
    # period, then re-using the dominant's overhead for lane availability.
    # For the MCP tool's recommend path we keep the simple analyze() call
    # but use the dominant's overhead (typically the largest workload,
    # whose per-backend overhead is the largest constant and therefore
    # the most conservative — under-recommendation rather than
    # over-recommendation on the secondary).
    overhead: dict[str, float] | None = None
    if calibration_path:
        cal = load_calibration(Path(calibration_path))
        if workload_ids:
            # Pick the workload whose summed per-backend overhead is
            # largest; that's the most likely dominant.
            best_w = max(
                workload_ids,
                key=lambda w: sum(cal.overhead_us.get(w, {}).values()),
            )
            overhead = dict(cal.overhead_us.get(best_w, {}))

    analysis = analyze_multi_rate(
        workload_ids=workload_ids,
        cost_matrix=cost_matrix,
        calibration_overhead_us=overhead,
    )
    rates_payload = [
        {
            "workload_id": r.workload_id,
            "period_us": r.period_us,
            "multiplicity": r.multiplicity,
            "preferred_lane": r.preferred_lane,
            "primary_lane_busy_us": r.primary_lane_busy_us,
            "achievable_frequency_hz": r.achievable_frequency_hz,
        }
        for r in analysis.rates
    ]
    lanes_payload = [
        {
            "lane": la.lane,
            "busy_us": la.busy_us_per_dominant_period,
            "idle_us": la.idle_us_per_dominant_period,
            "busy_fraction": la.busy_fraction,
        }
        for la in analysis.lane_availability
    ]
    return {
        "ok": True,
        "dominant_workload_id": analysis.dominant_workload_id,
        "dominant_period_us": analysis.dominant_period_us,
        "rates": rates_payload,
        "lane_availability": lanes_payload,
        "notes": list(analysis.notes),
    }


DEFAULT_LOOP_MEMORY_DIR = Path("build") / "loops" / "memory"


def xpu_rt_loop_memory_status(
    sm: SessionManager,  # noqa: ARG001
    *,
    target_id: str = DEFAULT_TARGET_ID,
    workload_set_key: str | None = None,
    workload_ids: list[str] | None = None,
    memory_dir: str | None = None,
) -> dict[str, Any]:
    """Summarise the cross-run bandit log for one ``(target, workload-set)``.

    Either ``workload_set_key`` (canonical form, e.g.
    ``"dronet*12+yolov8n*1"``) or ``workload_ids`` (list, will be
    canonicalised) must be supplied. ``memory_dir`` defaults to
    ``build/loops/memory/``.

    Returns:
        ``{"ok", "n_entries", "n_converged", "best_arm",
        "best_arm_mean_error_pct", "arm_stats"}``.
    """

    from xpu_rt.scheduling.loop_memory import (
        canonical_workload_set_key as _ck,
        summarize_memory,
    )

    if workload_set_key is None:
        if not workload_ids:
            return {
                "ok": False,
                "error": "either workload_set_key or workload_ids must be supplied",
            }
        workload_set_key = _ck(workload_ids)

    mdir = Path(memory_dir) if memory_dir else DEFAULT_LOOP_MEMORY_DIR
    summary = summarize_memory(target_id, workload_set_key, mdir)
    return {
        "ok": True,
        "target_id": target_id,
        "workload_set_key": workload_set_key,
        "memory_dir": str(mdir),
        **summary,
    }


def _multi_objective_from_dict(d: dict[str, Any]) -> MultiObjectiveSpec:
    weights_raw = d.get("weights", ())
    weights: list[ObjectiveWeight] = []
    for w in weights_raw:
        weights.append(
            ObjectiveWeight(
                kind=ObjectiveKind(str(w["kind"])),
                weight=float(w["weight"]),
                target_value=(
                    None if w.get("target_value") is None else float(w["target_value"])
                ),
            )
        )
    if not weights:
        return MultiObjectiveSpec()
    return MultiObjectiveSpec(weights=tuple(weights))


def xpu_rt_evaluate_objective(
    sm: SessionManager,  # noqa: ARG001
    *,
    loop_state_dict: dict[str, Any],
    multi_objective_spec_dict: dict[str, Any],
) -> dict[str, Any]:
    """Rescore the *last* round of a loop state under a new objective spec.

    Lets the agent compare alternative weightings without re-running the
    solver. Uses the persisted history's most recent round; metrics that
    weren't captured at solve time (deadlines, peak memory, etc.) fall
    back to the defaults exposed by :func:`compute_metrics`.

    Args:
        sm: MCP session manager (unused).
        loop_state_dict: A JSON-typed :class:`LoopState`.
        multi_objective_spec_dict: ``{"weights": [{"kind", "weight",
            "target_value"}, ...]}``.

    Returns:
        ``{"ok", "score", "component_scores", "raw_metrics", "iteration"}``.
    """

    state = state_from_dict(loop_state_dict)
    if not state.history:
        return {"ok": False, "error": "loop state has no history"}
    spec = _multi_objective_from_dict(multi_objective_spec_dict)
    last = state.history[-1]
    raw = last.objective_score.raw_metrics if last.objective_score else None
    if raw is None:
        raw = ScheduleMetrics(
            makespan_us=float(last.predicted_makespan_us),
            deadline_violations=0,
            deadline_violation_total_us=0.0,
            peak_memory_bytes=0,
            energy_proxy_joules=None,
            makespan_variance_us=0.0,
        )
    rescored = evaluate(spec, raw)
    return {
        "ok": True,
        "iteration": last.iteration,
        "score": rescored.score,
        "component_scores": dict(rescored.component_scores),
        "raw_metrics": {
            "makespan_us": raw.makespan_us,
            "deadline_violations": raw.deadline_violations,
            "deadline_violation_total_us": raw.deadline_violation_total_us,
            "peak_memory_bytes": raw.peak_memory_bytes,
            "energy_proxy_joules": raw.energy_proxy_joules,
            "makespan_variance_us": raw.makespan_variance_us,
        },
    }


def xpu_rt_loop_status(
    sm: SessionManager,  # noqa: ARG001
    *,
    workload_id: str,
    target_id: str = DEFAULT_TARGET_ID,
) -> dict[str, Any]:
    """Read-only: load the persisted loop state, if any."""

    path = _state_path(workload_id, target_id)
    if not path.is_file():
        return {
            "ok": False,
            "error": f"no persisted state at {path}",
            "workload_id": workload_id,
            "target_id": target_id,
        }
    state = load_loop_state(path)
    return {
        "ok": True,
        "state": state_to_dict(state),
        "converged": has_converged(state),
        "iteration": state.iteration,
        "status": state.status,
    }


# --------------------------------------------------------------------------- #
# Registration — joined into ALL_TOOLS via xpu_rt.mcp.tools.__init__
# --------------------------------------------------------------------------- #


FEEDBACK_LOOP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_recommend_granularity",
        "description": (
            "Recommend a specialty-driven chunking + solver choice for a "
            "workload. Returns the chunk list, solver choice (mosek/cpsat/"
            "greedy), partition count, and per-family specialty map."
        ),
        "phase": "inspect",
        "handler": xpu_rt_recommend_granularity,
        "input_schema": {
            "type": "object",
            "properties": {
                "workload_id": {"type": "string"},
                "cost_matrix_path": {"type": "string"},
                "target_id": {"type": "string"},
                "max_chunk_ops": {"type": "integer"},
                "max_partitions": {"type": "integer"},
            },
            "required": ["workload_id", "cost_matrix_path"],
        },
    },
    {
        "name": "xpu_rt_feedback_step",
        "description": (
            "Run one iteration of the Stage-4 feedback loop: apply the "
            "current calibration, re-chunk, solve, translation-validate, "
            "absorb the optional measurement, and classify the next "
            "decision (converged / recompile_finer / recompile_coarser / "
            "recalibrate_only / abort)."
        ),
        "phase": "transform",
        "handler": xpu_rt_feedback_step,
        "input_schema": {
            "type": "object",
            "properties": {
                "cost_matrix_path": {"type": "string"},
                "workload_id": {"type": "string"},
                "target_id": {"type": "string"},
                "loop_state_dict": {"type": ["object", "null"]},
                "measurement_dict": {"type": ["object", "null"]},
                "persist": {"type": "boolean"},
            },
            "required": ["cost_matrix_path"],
        },
    },
    {
        "name": "xpu_rt_apply_measurement",
        "description": (
            "Apply one calibration EMA update to the loop state without "
            "re-scheduling. Useful when the agent wants to inspect the "
            "new calibration before deciding to re-step."
        ),
        "phase": "transform",
        "handler": xpu_rt_apply_measurement,
        "input_schema": {
            "type": "object",
            "properties": {
                "loop_state_dict": {"type": "object"},
                "measurement_dict": {"type": "object"},
                "persist": {"type": "boolean"},
            },
            "required": ["loop_state_dict", "measurement_dict"],
        },
    },
    {
        "name": "xpu_rt_recommend_multiplicity",
        "description": (
            "Identify the dominant workload (longest min-over-backends "
            "critical-path latency) and recommend a per-secondary "
            "multiplicity + preferred lane that fits inside one dominant "
            "period. Static upper-bound model — see returned 'notes'."
        ),
        "phase": "inspect",
        "handler": xpu_rt_recommend_multiplicity,
        "input_schema": {
            "type": "object",
            "properties": {
                "workload_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cost_matrix_path": {"type": "string"},
                "calibration_path": {"type": ["string", "null"]},
            },
            "required": ["workload_ids", "cost_matrix_path"],
        },
    },
    {
        "name": "xpu_rt_loop_status",
        "description": (
            "Read-only: load the persisted loop state from "
            "build/loops/<workload>__<target>.json (if any) and return "
            "it for inspection."
        ),
        "phase": "inspect",
        "handler": xpu_rt_loop_status,
        "input_schema": {
            "type": "object",
            "properties": {
                "workload_id": {"type": "string"},
                "target_id": {"type": "string"},
            },
            "required": ["workload_id"],
        },
    },
    {
        "name": "xpu_rt_evaluate_objective",
        "description": (
            "Rescore the most recent round of a persisted loop state under "
            "a new MultiObjectiveSpec. Returns the weighted score plus the "
            "per-objective component breakdown — without re-running the "
            "solver. Used by the agent to explore weighting trade-offs."
        ),
        "phase": "inspect",
        "handler": xpu_rt_evaluate_objective,
        "input_schema": {
            "type": "object",
            "properties": {
                "loop_state_dict": {"type": "object"},
                "multi_objective_spec_dict": {"type": "object"},
            },
            "required": ["loop_state_dict", "multi_objective_spec_dict"],
        },
    },
    {
        "name": "xpu_rt_loop_memory_status",
        "description": (
            "Read-only: summarise the cross-run bandit log for a given "
            "(target, workload-set). Returns total entries, converged "
            "count, the best arm by mean abs_pct_error, and per-arm "
            "(successes, failures, mean_error_pct, n_observations)."
        ),
        "phase": "inspect",
        "handler": xpu_rt_loop_memory_status,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "workload_set_key": {"type": ["string", "null"]},
                "workload_ids": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "memory_dir": {"type": ["string", "null"]},
            },
            "required": [],
        },
    },
]


__all__ = [
    "FEEDBACK_LOOP_TOOLS",
    "xpu_rt_apply_measurement",
    "xpu_rt_evaluate_objective",
    "xpu_rt_feedback_step",
    "xpu_rt_loop_memory_status",
    "xpu_rt_loop_status",
    "xpu_rt_recommend_granularity",
    "xpu_rt_recommend_multiplicity",
]
