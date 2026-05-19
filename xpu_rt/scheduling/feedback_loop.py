"""Compile → granularity → solver → TV → execute → calibrate feedback loop.

Stage 4 driver. Ties together :mod:`xpu_rt.runtime.calibration`,
:mod:`xpu_rt.scheduling.policy`, :mod:`xpu_rt.scheduling.granularity`,
the joint CP-SAT solver in :mod:`xpu_rt.solve.schedule_joint_cpsat`, the
MOSEK MILP envelope in :mod:`xpu_rt.scheduler.scheduler`, an inline
EFT list scheduler, and the schedule TV gate in
:mod:`xpu_rt.solve.schedule_tv`.

The loop is a finite-state machine with an immutable :class:`LoopState`
threaded through :func:`step`. Each step:

1. Apply the current calibration to the raw cost matrix.
2. Re-chunk under the current ``max_chunk_ops`` (specialty-driven).
3. Pick a solver via :class:`SchedulerPolicy`.
4. Solve the joint problem under the chosen backend
   (MOSEK MILP / CP-SAT / greedy EFT). The MOSEK path uses the inline
   adapter lifted from ``scripts/experiments/exp7_real_perop_scheduling.py``
   to wrap the chunked DAG into :class:`xpu_rt.scheduler.workload.Workload`.
   Greedy is implemented in-module as an EFT list scheduler.
5. Translation-validate the schedule.
6. If a measurement was supplied, EMA-update the calibration model and
   classify ``decision_next`` for the next iteration.

The loop is intentionally pure: ``step`` returns a new :class:`LoopState`
and never mutates inputs. Callers (the MCP wrappers in
:mod:`xpu_rt.mcp.feedback_loop_tools`) handle persistence.

Driving from the Claude-Code-side ``xpu-rt-compile`` skill: the skill's
"Closed loop with contention feedback" section already describes the
overarching execute → measure → re-plan loop; the four MCP tools added
in :mod:`xpu_rt.mcp.feedback_loop_tools` plug this Stage-4 driver into
that flow in-process. The skill markdown was extended with a
"Feedback-loop closure (Stage 4)" appendix referencing the four tools.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog

from xpu_rt.runtime.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEPLOYMENT_MODE_COLD,
    DEPLOYMENT_MODE_WARM,
    WARM_TECHNIQUES,
    CalibrationModel,
    CalibrationRound,
    MeasurementRecord,
    update_from_measurement,
)
from xpu_rt.runtime.calibration import (
    apply as apply_calibration,
)
from xpu_rt.runtime.calibration import (
    compose_predicted_makespan_us,
)
from xpu_rt.runtime.measurement_cache import (
    DEFAULT_CACHE_DIR as DEFAULT_MEASUREMENT_CACHE_DIR,
)
from xpu_rt.runtime.measurement_cache import (
    load_cache as load_measurement_cache,
)
from xpu_rt.scheduler.qnn_real_workload import BACKENDS as _BACKEND_NAMES
from xpu_rt.scheduling.measurement_driven import (
    CandidateSchedule,
    evaluate_with_cache,
)
from xpu_rt.scheduler.qnn_real_workload import (
    QnnDag,
    chunk_dag_from_chunks,
    make_chain_dag,
)
from xpu_rt.scheduling.granularity import (
    Chunk,
    GranularityPlan,
    apply_fusion,
    compute_specialty_matrix,
    propose_chunks,
)
from xpu_rt.scheduling.loop_memory import (
    MemoryEntry,
    append_entry as append_memory_entry,
    canonical_workload_set_key,
    default_candidate_arms,
    recommend_initial_arm,
)
from xpu_rt.scheduling.objectives import (
    MultiObjectiveSpec,
    ObjectiveKind,
    ObjectiveScore,
    ScheduleMetrics,
    compute_metrics,
    evaluate,
)
from xpu_rt.scheduling.policy import (
    MemoryPlannerPolicy,
    SchedulerPolicy,
    SolverChoice,
)
from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint
from xpu_rt.solve.schedule_tv import translation_validate_schedule

log = structlog.get_logger(__name__)

LOOP_SCHEMA_VERSION = "feedback_loop_v1"

DecisionNext = Literal[
    "converged",
    "recompile_finer",
    "recompile_coarser",
    "recalibrate_only",
    "abort",
]

LoopStatus = Literal["init", "running", "converged", "failed", "max_iter"]

_MIN_CHUNK_OPS = 1
_MAX_CHUNK_OPS = 64

# Default location for the cross-iteration bandit log. Per-(target,
# workload-set) JSONL files live under this directory; the loop driver
# reads at init and appends at each step's end. See
# :mod:`xpu_rt.scheduling.loop_memory`.
DEFAULT_LOOP_MEMORY_DIR = Path("build") / "loops" / "memory"


@dataclass(frozen=True)
class LoopRound:
    """One round of the feedback loop, recorded for audit.

    Attributes:
        tv_memory_proved: Reserved. Currently always ``True`` because the
            loop does not yet plan memory; pair with ``tv_memory_skipped``
            to detect the no-op case in audit logs.
        tv_memory_skipped: ``True`` whenever the loop did not actually run
            buffer-level TV (Stage 4 does not yet wire memory planning).
            When the buffer-level TV is wired in, set this to ``False``
            and report the real proof status in ``tv_memory_proved``.
    """

    iteration: int
    predicted_makespan_us: float
    measured_makespan_us: float | None
    n_partitions: int
    solver_choice: str
    tv_schedule_proved: bool
    tv_memory_proved: bool
    decision_next: DecisionNext
    reason: str
    timestamp: str
    tv_memory_skipped: bool = True
    objective_score: ObjectiveScore | None = None
    # ``"measurement_cache"`` when the loop's predicted_makespan_us was
    # short-circuited from a real on-board measurement; ``"predicted"``
    # for the calibration-driven path; ``"mixed"`` when only a subset of
    # placements were measured. Default keeps backward compat.
    prediction_source: Literal["predicted", "measurement_cache", "mixed"] = "predicted"


@dataclass(frozen=True)
class LoopConfig:
    """Convergence / cap parameters for the loop.

    The convergence rule is an EMA-style "at least M of the last K rounds
    are in-band" check, which admits oscillating ground truth (where the
    older ``consecutive_required`` rule would never fire). ``consecutive_required``
    is retained for backward compatibility but is ignored by the decision
    logic — set ``convergence_min_in_band`` instead.
    """

    epsilon: float = 0.10
    # Deprecated alias retained for backward compatibility with callers
    # that still pass ``consecutive_required=...``. Ignored at runtime.
    consecutive_required: int = 2
    convergence_window: int = 3
    convergence_min_in_band: int = 2
    max_iterations: int = 8
    outlier_threshold: float = 2.0
    outlier_count_threshold: int = 2
    worst_outlier_ratio_threshold: float = 3.0
    transfer_dominance_threshold: float = 0.30
    regression_threshold: float = 1.20
    max_same_decision_retries: int = 2
    max_chunk_ops: int = 16
    max_partitions: int = 200
    cpsat_timeout_ms: int = 30_000
    fusion_gain_threshold: float = 0.3
    enable_fusion: bool = True
    granularity_perturbation_step: int = 4
    multi_objective: MultiObjectiveSpec = field(default_factory=MultiObjectiveSpec)
    objective_convergence_band: float = 0.10
    deployment_mode: str = DEPLOYMENT_MODE_COLD
    # When True, before falling back to the calibrated prediction, look
    # up the candidate schedule in the measurement cache. A full hit
    # short-circuits the predictor — the loop acts on real wall time.
    # Default False keeps backward compat with callers that don't yet
    # know about the cache; experiments opt in explicitly.
    measurement_first: bool = False
    measurement_cache_dir: Path | None = None


@dataclass(frozen=True)
class LoopState:
    """Immutable feedback-loop state.

    Attributes:
        schema_version: Loop schema tag (``feedback_loop_v1``).
        workload_id: The workload key into the cost matrix.
        target_id: Hardware target id (matches calibration model).
        iteration: 0 in :func:`init_loop_state`; increments per step.
        current_calibration: The calibration model in force at this step.
        current_chunks: The granularity decision in force at this step.
        current_solver_choice: The solver chosen for this step.
        current_predicted_makespan_us: Prediction from the most recent
            solve, or ``None`` before the first step has run.
        current_max_chunk_ops: The chunk-size cap currently in force
            (re-compile decisions adjust this and re-chunk on the next
            step).
        history: Append-only log of :class:`LoopRound` records.
        status: Current loop status (state-machine label).
        consecutive_in_band: Legacy field, retained for serialisation
            compatibility. The active rule lives in ``error_history``.
        error_history: Tuple of per-round absolute relative errors
            ``|pred - measured| / measured`` (in-band ⇔ ``< epsilon``).
            The convergence check inspects the last
            ``convergence_window`` entries.
        decision_history: Tuple of past ``decision_next`` values, used to
            detect "same decision N times in a row" cycles.
        baseline_makespan_us: Greedy-EFT makespan of the most permissive
            chunking, captured at init. Used as the regression-guard
            denominator. ``math.inf`` disables the guard.
    """

    schema_version: str
    workload_id: str
    target_id: str
    iteration: int
    current_calibration: CalibrationModel
    current_chunks: tuple[Chunk, ...]
    current_solver_choice: str
    current_predicted_makespan_us: float | None
    current_max_chunk_ops: int
    history: tuple[LoopRound, ...]
    status: LoopStatus
    consecutive_in_band: int = 0
    error_history: tuple[float, ...] = ()
    decision_history: tuple[str, ...] = ()
    baseline_makespan_us: float = math.inf


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_dag(workload_id: str, cost_matrix: dict[str, Any]) -> QnnDag:
    return make_chain_dag(workload_id, cost_matrix=cost_matrix)


def _propose(
    workload_id: str,
    cost_matrix: dict[str, Any],
    max_chunk_ops: int,
    max_partitions: int,
) -> GranularityPlan:
    dag = _build_dag(workload_id, cost_matrix)
    specialty = compute_specialty_matrix(cost_matrix, workload_id)
    return propose_chunks(
        dag,
        cost_matrix,
        workload_id,
        specialty,
        max_chunk_ops=max_chunk_ops,
        max_partitions=max_partitions,
    )


@dataclass(frozen=True)
class _SolveResult:
    """Internal solver-dispatch result.

    Normalised across CP-SAT, MOSEK and greedy backends so the loop's
    consumer (``step``) reads a single shape regardless of which backend
    produced it.
    """

    feasible: bool
    status: str
    makespan_us: float
    start_times: dict[str, float]
    end_times: dict[str, float]
    device_assignments: dict[str, int]


def _topo_order(pids: list[str], deps: dict[str, list[str]]) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def visit(p: str) -> None:
        if p in visited:
            return
        visited.add(p)
        for d in deps.get(p, []):
            visit(d)
        order.append(p)

    for pid in pids:
        visit(pid)
    return order


def _solve_cpsat(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float | None]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    transfer_us: list[list[float]],
    timeout_ms: int,
) -> _SolveResult:
    sol = solve_schedule_joint(
        partition_ids=partition_ids,
        durations_us_by_device=durations_us_by_device,
        dependencies=dependencies,
        num_devices=num_devices,
        transfer_us=transfer_us,
        timeout_ms=timeout_ms,
    )
    return _SolveResult(
        feasible=sol.feasible,
        status=sol.status,
        makespan_us=sol.makespan_us,
        start_times=dict(sol.start_times),
        end_times=dict(sol.end_times),
        device_assignments=dict(sol.device_assignments),
    )


def _solve_greedy_eft(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float | None]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    transfer_us: list[list[float]],
) -> _SolveResult:
    """Earliest-finish-time list scheduler over a chunked DAG.

    Walks chunks in topological order and, for each chunk, picks the
    ``(device, start_time)`` minimising completion time, accounting for
    cross-device transfers from already-scheduled predecessors. Skips
    devices marked as infeasible (``None`` / ``inf``).
    """

    topo = _topo_order(partition_ids, dependencies)
    device_avail = [0.0] * num_devices
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    chosen_dev: dict[str, int] = {}
    for pid in topo:
        durations = durations_us_by_device[pid]
        best_end = math.inf
        best_start = math.inf
        best_dev = -1
        for d in range(num_devices):
            dur = durations[d]
            if dur is None or (
                isinstance(dur, float) and (math.isinf(dur) or math.isnan(dur))
            ):
                continue
            ready = device_avail[d]
            for pred in dependencies.get(pid, []):
                pred_end = end_times[pred]
                pred_dev = chosen_dev[pred]
                ready = max(ready, pred_end + transfer_us[pred_dev][d])
            end = ready + float(dur)
            if end < best_end:
                best_end = end
                best_start = ready
                best_dev = d
        if best_dev < 0:
            return _SolveResult(
                feasible=False,
                status="infeasible",
                makespan_us=math.inf,
                start_times={},
                end_times={},
                device_assignments={},
            )
        start_times[pid] = best_start
        end_times[pid] = best_end
        chosen_dev[pid] = best_dev
        device_avail[best_dev] = best_end
    makespan = max(end_times.values()) if end_times else 0.0
    return _SolveResult(
        feasible=True,
        status="optimal_local",
        makespan_us=makespan,
        start_times=start_times,
        end_times=end_times,
        device_assignments=chosen_dev,
    )


def _solve_mosek(
    partition_ids: list[str],
    durations_us_by_device: dict[str, list[float | None]],
    dependencies: dict[str, list[str]],
    num_devices: int,
    transfer_us: list[list[float]],
    timeout_ms: int,
) -> _SolveResult:
    """Wrap a chunked DAG into ``Workload`` and route through the MOSEK MILP.

    The adapter (``Operation`` build, infeasibility marking, transfer
    matrix conversion, solver-state extraction) follows the recipe in
    ``scripts/experiments/exp7_real_perop_scheduling.py::mosek_milp``.
    Imports are local because the CVXPY/MOSEK stack is heavy and only
    needed when the policy actually picks MOSEK.
    """

    import contextlib
    import io

    import numpy as np

    from xpu_rt.scheduler.scheduler import schedule as mosek_schedule
    from xpu_rt.scheduler.workload import Operation, Workload

    pid_to_idx = {p: i for i, p in enumerate(partition_ids)}
    ops: list[Operation] = []
    for i, pid in enumerate(partition_ids):
        per_dev = durations_us_by_device[pid]
        finite_costs = [
            float(c)
            for c in per_dev
            if c is not None and not (isinstance(c, float) and math.isinf(c))
        ]
        # Placeholder must be finite so MOSEK can size big-M; the
        # infeasible-combo constraint forbids alpha[i,k]=1 anyway.
        placeholder = (max(finite_costs) if finite_costs else 1.0) * 10.0
        proc: list[float] = []
        infeas: list[int] = []
        for k, c in enumerate(per_dev):
            if c is None or (isinstance(c, float) and math.isinf(c)):
                proc.append(placeholder)
                infeas.append(k)
            else:
                proc.append(float(c))
        ops.append(
            Operation(
                processing_times=proc,
                operation_id=i,
                operation_name=pid,
                job_id=0,
                infeasible_combinations=infeas or None,
            )
        )
    for pid in partition_ids:
        for pred in dependencies.get(pid, []):
            ops[pid_to_idx[pid]].add_predecessor(ops[pid_to_idx[pred]])
    machines = [f"dev_{d}" for d in range(num_devices)]
    transfer = np.asarray(transfer_us, dtype=float)
    workload = Workload(ops, machines, transfer)
    silent_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(
            silent_buf
        ):
            t_arr, alpha, _, _ = mosek_schedule(
                workload, time_limit=max(0.001, timeout_ms / 1000.0)
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("feedback_loop_mosek_error", error=type(exc).__name__)
        return _SolveResult(
            feasible=False,
            status=f"error:{type(exc).__name__}",
            makespan_us=math.inf,
            start_times={},
            end_times={},
            device_assignments={},
        )
    state = getattr(workload, "solver_state", {}) or {}
    status_str = str(state.get("problem_status", "unknown"))
    feasible = (
        t_arr is not None
        and alpha is not None
        and status_str.lower() in ("optimal", "optimal_inaccurate")
    )
    if not feasible:
        return _SolveResult(
            feasible=False,
            status=status_str,
            makespan_us=math.inf,
            start_times={},
            end_times={},
            device_assignments={},
        )
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    device_assignments: dict[str, int] = {}
    for i, pid in enumerate(partition_ids):
        k = int(alpha[i].argmax())
        c = durations_us_by_device[pid][k]
        dur = float(c) if c is not None else 0.0
        start = float(t_arr[i])
        start_times[pid] = start
        end_times[pid] = start + dur
        device_assignments[pid] = k
    makespan = float(state.get("makespan") or 0.0)
    if makespan <= 0.0:
        makespan = max(end_times.values()) if end_times else 0.0
    return _SolveResult(
        feasible=True,
        status=status_str,
        makespan_us=makespan,
        start_times=start_times,
        end_times=end_times,
        device_assignments=device_assignments,
    )


def _solve_with_policy(
    workload_id: str,
    cost_matrix: dict[str, Any],
    chunks: tuple[Chunk, ...],
    scheduler_policy: SchedulerPolicy,
    timeout_ms: int,
) -> tuple[float, SolverChoice, bool, dict[str, Any]]:
    """Run the joint scheduler under the policy-selected solver.

    Dispatches to the MOSEK MILP, CP-SAT joint solver, or the inline EFT
    greedy list scheduler based on :class:`SchedulerPolicy`. All three
    backends return a normalised :class:`_SolveResult` so downstream TV
    and the caller's contract (``predicted_us, choice, tv_proved,
    schedule_dict``) stay identical.

    Returns:
        ``(predicted_makespan_us, solver_choice, tv_proved, schedule_dict)``.
        ``schedule_dict`` carries ``start_times``, ``end_times``,
        ``device_assignments`` for downstream TV / inspection.
    """

    dag = _build_dag(workload_id, cost_matrix)
    chunked = chunk_dag_from_chunks(dag, list(chunks))
    partition_ids = list(chunked.partition_ids)
    durations = {pid: list(row) for pid, row in chunked.durations_us_by_device.items()}
    deps = {k: list(v) for k, v in chunked.dependencies.items()}
    transfer = [row[:] for row in chunked.transfer_us]
    n = len(partition_ids)
    choice = scheduler_policy.choose(n_partitions=n)

    if choice is SolverChoice.MOSEK:
        sol = _solve_mosek(
            partition_ids, durations, deps, chunked.num_devices, transfer, timeout_ms
        )
    elif choice is SolverChoice.CPSAT:
        sol = _solve_cpsat(
            partition_ids, durations, deps, chunked.num_devices, transfer, timeout_ms
        )
    elif choice is SolverChoice.GREEDY:
        sol = _solve_greedy_eft(
            partition_ids, durations, deps, chunked.num_devices, transfer
        )
    else:  # pragma: no cover - StrEnum exhaustiveness
        raise ValueError(f"unknown solver choice: {choice}")

    if not sol.feasible:
        log.warning(
            "feedback_loop_solve_infeasible",
            workload=workload_id,
            n_partitions=n,
            solver=str(choice),
            status=sol.status,
        )
        return (math.inf, choice, False, {})

    durations_for_tv: dict[str, list[float]] = {
        pid: [(v if v is not None else 0.0) for v in row]
        for pid, row in durations.items()
    }
    tv = translation_validate_schedule(
        partition_ids=partition_ids,
        durations_us_by_device=durations_for_tv,
        dependencies=deps,
        num_devices=chunked.num_devices,
        start_times=sol.start_times,
        end_times=sol.end_times,
        device_assignments=sol.device_assignments,
        makespan_us=sol.makespan_us,
        transfer_us=transfer,
        use_z3=False,
    )
    schedule_dict: dict[str, Any] = {
        "start_times": sol.start_times,
        "end_times": sol.end_times,
        "device_assignments": sol.device_assignments,
        "makespan_us": sol.makespan_us,
        "n_partitions": n,
    }
    return (sol.makespan_us, choice, tv.proved, schedule_dict)


def _convergence_check(
    error_history: tuple[float, ...],
    *,
    window: int,
    min_in_band: int,
    epsilon: float,
) -> tuple[bool, int, int]:
    """Return ``(converged, in_band_count, window_size)`` over the last K errors.

    Looks at the most recent ``window`` entries of ``error_history``. The
    loop is converged when at least ``min_in_band`` of those are
    strictly below ``epsilon``. This admits oscillation: a single
    out-of-band round inside the window doesn't reset progress (unlike
    the old "consecutive" rule).
    """

    if window <= 0:
        return (False, 0, 0)
    window_slice = error_history[-window:]
    in_band = sum(1 for e in window_slice if e < epsilon)
    return (in_band >= min_in_band and len(window_slice) >= min_in_band,
            in_band, len(window_slice))


def _classify_decision(
    *,
    predicted_us: float,
    measurement: MeasurementRecord,
    iteration: int,
    config: LoopConfig,
    error_history: tuple[float, ...],
    decision_history: tuple[str, ...],
    schedule_dict: dict[str, Any],
    transfer_us_total: float,
    baseline_makespan_us: float,
) -> tuple[DecisionNext, str, tuple[float, ...]]:
    """Apply the decision rubric.

    Returns:
        ``(decision, reason, new_error_history)``. The caller is
        responsible for appending the decision to ``decision_history``
        and applying the cycle-breaker / regression guard at the
        :func:`step` level (so those concerns stay in one place).

    Rules in order:
      a. Convergence: ``min_in_band`` of last ``window`` errors are
         in-band → ``converged``.
      b. ``iteration >= max_iterations`` → ``abort`` (caller marks
         ``status='max_iter'``).
      c. Combined signal: if (outlier count ≥ ``outlier_count_threshold``)
         OR (worst outlier ratio > ``worst_outlier_ratio_threshold``),
         pick ``recompile_finer``.
      d. Else if ``transfer_us_total > transfer_dominance_threshold * measured``:
         ``recompile_coarser``.
      e. Else: ``recalibrate_only``.
    """

    measured = measurement.measured_us
    if measured <= 0:
        return ("abort", "measured_makespan_us<=0", error_history)

    err = abs(predicted_us - measured) / measured
    new_error_history = error_history + (err,)

    converged, in_band_count, window_size = _convergence_check(
        new_error_history,
        window=config.convergence_window,
        min_in_band=config.convergence_min_in_band,
        epsilon=config.epsilon,
    )
    if converged:
        # Regression guard is applied at the step() level after we have
        # baseline_makespan_us in hand; the decision itself reports
        # ``converged`` and lets step() veto it if needed.
        return (
            "converged",
            (
                f"err={err:.3f}; {in_band_count}/{window_size} of last "
                f"{config.convergence_window} rounds in-band (eps={config.epsilon})"
            ),
            new_error_history,
        )

    if iteration >= config.max_iterations:
        return (
            "abort",
            f"iteration={iteration} >= max_iterations={config.max_iterations}",
            new_error_history,
        )

    # Combined-signal classification: compute both outlier and transfer
    # signals up front so the rule reads as a single decision rather
    # than a cascade of overrides.
    op_outlier = _detect_op_outlier(
        measurement=measurement,
        schedule_dict=schedule_dict,
        threshold=config.outlier_threshold,
    )
    n_outliers = 1 if op_outlier is not None else 0
    pred_sum = float(measurement.per_op_sum_us)
    worst_ratio = (measured / pred_sum) if pred_sum > 0 else 0.0
    is_outlier = (
        n_outliers >= config.outlier_count_threshold
        or worst_ratio > config.worst_outlier_ratio_threshold
    )
    is_transfer_dominated = (
        transfer_us_total > config.transfer_dominance_threshold * measured
    )

    if is_outlier:
        return (
            "recompile_finer",
            (
                f"outlier signal: n_outliers={n_outliers}, "
                f"worst_ratio={worst_ratio:.2f} (thresholds: "
                f"count={config.outlier_count_threshold}, "
                f"ratio={config.worst_outlier_ratio_threshold})"
            ),
            new_error_history,
        )

    if is_transfer_dominated:
        return (
            "recompile_coarser",
            (
                f"transfer_us={transfer_us_total:.0f} > "
                f"{config.transfer_dominance_threshold} * measured={measured:.0f}"
            ),
            new_error_history,
        )

    return (
        "recalibrate_only",
        f"err={err:.3f} >= eps={config.epsilon}; recalibrate and re-solve",
        new_error_history,
    )


def _is_cycle(
    decision_history: tuple[str, ...],
    candidate: str,
    *,
    max_retries: int,
) -> bool:
    """Return True iff appending ``candidate`` would exceed ``max_retries``
    consecutive repeats of the same non-terminal decision.

    "Exceed" means ``candidate`` is the same as the last ``max_retries``
    decisions and would become the ``max_retries + 1``-th repetition.
    The terminal decisions ``converged`` / ``abort`` are never considered
    cycles.
    """

    if candidate in ("converged", "abort"):
        return False
    if len(decision_history) < max_retries:
        return False
    tail = decision_history[-max_retries:]
    return all(d == candidate for d in tail)


def _detect_op_outlier(
    *,
    measurement: MeasurementRecord,
    schedule_dict: dict[str, Any],
    threshold: float,
) -> str | None:
    """Return the offending op_id, or ``None`` if no per-op breakdown is present.

    Conservative heuristic: if the measurement carries a top-level
    chain-sum that's >threshold× the predicted per-op sum, flag the chain
    as a whole. Per-op timing breakdowns aren't in
    :class:`MeasurementRecord` today, so this is the strongest signal we
    can extract without changing the Stage-1 schema.
    """

    pred_sum = float(measurement.per_op_sum_us)
    measured = float(measurement.measured_us)
    if pred_sum <= 0 or measured <= 0:
        return None
    ratio = measured / pred_sum
    if ratio > threshold:
        return f"chain[{measurement.workload_id}/{measurement.backend}]"
    return None


def _apply_calibration_to_makespan(
    *,
    workload_id: str,
    calibration: CalibrationModel,
    schedule_dict: dict[str, Any] | None,
    plan_chunks: tuple[Chunk, ...] | list[Chunk],
    raw_solver_us: float,
    deployment_mode: str = DEPLOYMENT_MODE_COLD,
) -> float:
    """Fold v3 calibration overhead + contention into the solver makespan.

    The solver returns ``raw_solver_us = max over lanes of busy_time``
    where ``busy_time`` is the sum of raw chunk durations on that lane.
    The v3 model says ``predicted_per_lane = (busy + overhead) * contention``.
    This helper reconstructs per-lane busy time from the solver's
    assignment, applies the calibration per lane, and returns the
    max (= actual lane-parallel finish time).

    When the schedule_dict is missing or empty (e.g., greedy solver
    that doesn't expose lane-busy directly), we fall back to
    ``raw_solver_us + max overhead`` as a conservative correction
    that at least includes the overhead term.
    """

    from xpu_rt.runtime.calibration import DEPLOYMENT_MODE_WARM

    if deployment_mode == DEPLOYMENT_MODE_WARM:
        overhead_src = calibration.overhead_us_warm
    else:
        overhead_src = calibration.overhead_us

    if not schedule_dict:
        ovh = overhead_src.get(workload_id, {})
        max_ovh = max(ovh.values(), default=0.0)
        return float(raw_solver_us) + float(max_ovh)

    device_assignments = schedule_dict.get("device_assignments", {})
    if not device_assignments:
        ovh = overhead_src.get(workload_id, {})
        max_ovh = max(ovh.values(), default=0.0)
        return float(raw_solver_us) + float(max_ovh)

    # device_assignments may be {chunk_id: int_index} (CP-SAT/MOSEK joint
    # solvers use integer indices into BACKENDS) or {chunk_id: str_backend}
    # (some greedy paths emit strings directly). Normalise to backend name.
    from xpu_rt.scheduler.qnn_real_workload import BACKENDS as _BACKENDS

    def _to_backend(v: object) -> str | None:
        if isinstance(v, str):
            return v if v in _BACKENDS else None
        if isinstance(v, int) and 0 <= v < len(_BACKENDS):
            return _BACKENDS[v]
        return None

    durations_by_chunk = {c.chunk_id: c.durations_us_by_backend for c in plan_chunks}
    per_lane_busy: dict[str, float] = {}
    for chunk_id, lane in device_assignments.items():
        backend = _to_backend(lane)
        if backend is None:
            continue
        dur_map = durations_by_chunk.get(chunk_id, {})
        if backend not in dur_map:
            continue
        d = float(dur_map[backend])
        if not math.isfinite(d):
            continue
        per_lane_busy[backend] = per_lane_busy.get(backend, 0.0) + d

    if not per_lane_busy:
        return float(raw_solver_us)
    composed = compose_predicted_makespan_us(
        model=calibration,
        workload_id=workload_id,
        per_lane_busy_us=per_lane_busy,
        concurrent_workloads=(),  # solo by default; multi-workload loops will pass their own set
        deployment_mode=deployment_mode,
    )
    return composed


def _estimate_transfer_us(
    chunks: tuple[Chunk, ...],
    schedule_dict: dict[str, Any],
    transfer_us_value: float = 100.0,
) -> float:
    """Approximate total inter-chunk transfer time in the realised schedule.

    Counts every chain edge ``(c_i, c_{i+1})`` whose chunks landed on
    different devices, and charges the fixed cross-backend penalty per
    transfer. Matches :data:`xpu_rt.scheduler.qnn_real_workload.DEFAULT_TRANSFER_US`.
    """

    if not chunks or "device_assignments" not in schedule_dict:
        return 0.0
    da = schedule_dict["device_assignments"]
    total = 0.0
    for i in range(len(chunks) - 1):
        a = da.get(chunks[i].chunk_id)
        b = da.get(chunks[i + 1].chunk_id)
        if a is None or b is None or a == b:
            continue
        total += float(transfer_us_value)
    return total


def _compute_baseline_makespan_us(
    workload_id: str,
    cost_matrix: dict[str, Any],
    config: LoopConfig,
    calibration: CalibrationModel | None = None,
    deployment_mode: str = DEPLOYMENT_MODE_COLD,
) -> float:
    """Greedy-EFT makespan on the most permissive natural chunking.

    Used as the regression-guard denominator: if a "converged" schedule
    is worse than this baseline by ``regression_threshold``×, the loop
    refuses to converge. Returns ``math.inf`` on any failure (which
    disables the guard — preferring false negatives over crashing the
    loop on synthetic fixtures).

    The baseline applies the same v3 calibration overhead+contention
    post-processing as the loop's predicted_makespan so the regression
    guard compares like-for-like. Without this correction, post-Bug-2
    predicted (which includes overhead) would always exceed the
    uncalibrated baseline by ~1 overhead unit, falsely tripping the
    guard.
    """

    try:
        plan = _propose(
            workload_id,
            cost_matrix,
            max_chunk_ops=config.max_chunk_ops,
            max_partitions=config.max_partitions,
        )
        dag = _build_dag(workload_id, cost_matrix)
        chunked = chunk_dag_from_chunks(dag, list(plan.chunks))
        sol = _solve_greedy_eft(
            partition_ids=list(chunked.partition_ids),
            durations_us_by_device={
                pid: list(row) for pid, row in chunked.durations_us_by_device.items()
            },
            dependencies={k: list(v) for k, v in chunked.dependencies.items()},
            num_devices=chunked.num_devices,
            transfer_us=[row[:] for row in chunked.transfer_us],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "feedback_loop_baseline_unavailable",
            workload=workload_id,
            error=type(exc).__name__,
        )
        return math.inf
    if not sol.feasible:
        return math.inf
    # Apply the same v3 calibration overhead+contention post-processing
    # the loop uses for predicted_makespan, so the regression guard
    # compares apples to apples.
    if calibration is None:
        return sol.makespan_us
    schedule_dict = {
        "device_assignments": dict(sol.device_assignments),
        "start_times": dict(sol.start_times),
        "end_times": dict(sol.end_times),
        "makespan_us": sol.makespan_us,
        "n_partitions": len(sol.device_assignments),
    }
    return _apply_calibration_to_makespan(
        workload_id=workload_id,
        calibration=calibration,
        schedule_dict=schedule_dict,
        plan_chunks=plan.chunks,
        raw_solver_us=sol.makespan_us,
        deployment_mode=deployment_mode,
    )


def init_loop_state(
    workload_id: str,
    target_id: str,
    cost_matrix: dict[str, Any],
    calibration: CalibrationModel,
    scheduler_policy: SchedulerPolicy = SchedulerPolicy(),
    memory_policy: MemoryPlannerPolicy = MemoryPlannerPolicy(),  # noqa: ARG001
    config: LoopConfig = LoopConfig(),
    memory_dir: Path | None = None,
    rng_seed: int | None = None,
) -> LoopState:
    """Seed a fresh :class:`LoopState` (no measurements applied yet).

    ``memory_policy`` is accepted for forward compatibility with the
    Stage-2 memory planner; the current loop drives only the schedule
    solver and ignores memory policy beyond signature symmetry.

    Args:
        workload_id: Workload key in the cost matrix.
        target_id: Calibration-model target identifier.
        cost_matrix: Loaded cost matrix (post ``load_cost_matrix``).
        calibration: Calibration model to start from (Stage-1 bootstrap).
        scheduler_policy: Policy that maps ``n_partitions`` → solver.
        memory_policy: Reserved (see above).
        config: Convergence / cap parameters.

    Returns:
        A frozen :class:`LoopState` with ``status='init'`` and an empty
        history. The first :func:`step` call will run a full
        plan/solve/TV pass and transition status to ``'running'``.
    """

    calibrated = apply_calibration(
        calibration,
        cost_matrix,
        workload_id=workload_id,
        deployment_mode=config.deployment_mode,
    )

    # Consult the cross-run bandit log for this (target, workload-set).
    # Override config.max_chunk_ops only when a memory log already exists
    # for this key; empty memory falls back to the configured default.
    seeded_max_chunk_ops = config.max_chunk_ops
    seed_source = "config"
    mem_dir = memory_dir if memory_dir is not None else DEFAULT_LOOP_MEMORY_DIR
    workload_set_key = canonical_workload_set_key([workload_id])
    mem_log = Path(mem_dir) / f"{target_id}__{workload_set_key}.jsonl"
    if mem_log.is_file():
        try:
            arm = recommend_initial_arm(
                target_id=target_id,
                workload_set_key=workload_set_key,
                candidate_arms=default_candidate_arms(),
                memory_dir=Path(mem_dir),
                rng_seed=rng_seed,
            )
            seeded_max_chunk_ops = arm.max_chunk_ops
            seed_source = "bandit_memory"
        except Exception as exc:  # noqa: BLE001 — memory is advisory.
            log.warning("feedback_loop_memory_seed_error", error=type(exc).__name__)

    plan = _propose(
        workload_id,
        calibrated,
        max_chunk_ops=seeded_max_chunk_ops,
        max_partitions=config.max_partitions,
    )
    choice = scheduler_policy.choose(n_partitions=plan.n_partitions)
    baseline = _compute_baseline_makespan_us(
        workload_id,
        calibrated,
        config,
        calibration=calibration,
        deployment_mode=config.deployment_mode,
    )
    log.info(
        "feedback_loop_init",
        workload=workload_id,
        target=target_id,
        n_partitions=plan.n_partitions,
        solver_choice=str(choice),
        baseline_makespan_us=baseline,
        seed_source=seed_source,
        seeded_max_chunk_ops=seeded_max_chunk_ops,
    )
    return LoopState(
        schema_version=LOOP_SCHEMA_VERSION,
        workload_id=workload_id,
        target_id=target_id,
        iteration=0,
        current_calibration=calibration,
        current_chunks=plan.chunks,
        current_solver_choice=str(choice),
        current_predicted_makespan_us=None,
        current_max_chunk_ops=seeded_max_chunk_ops,
        history=(),
        status="init",
        baseline_makespan_us=baseline,
    )


def _adjust_chunk_cap(current: int, *, finer: bool, step_size: int) -> int:
    """Perturb the chunk cap by ``step_size`` (additive), clamped.

    Additive perturbation explores neighbouring granularities densely:
    halving / doubling jumped too far on workloads where the sweet spot
    is e.g. ``max_chunk_ops=12`` rather than 8 or 16.
    """

    delta = -abs(step_size) if finer else abs(step_size)
    nxt = current + delta
    return max(_MIN_CHUNK_OPS, min(_MAX_CHUNK_OPS, nxt))


def _candidate_from_state(
    state: LoopState,
    *,
    deployment_mode: str,
    schedule_dict: dict[str, Any] | None = None,
) -> CandidateSchedule:
    """Project a single-workload LoopState into a CandidateSchedule.

    The lane is derived from the solver's device assignments (mode over
    chunks); if the schedule isn't available yet, fall back to each
    chunk's preferred backend. Techniques are derived from the loop's
    deployment_mode: warm → WARM_TECHNIQUES, cold → ().
    """

    lane: str
    if schedule_dict and schedule_dict.get("device_assignments"):
        counts: dict[str, int] = {}
        for dev_idx in schedule_dict["device_assignments"].values():
            try:
                name = _BACKEND_NAMES[int(dev_idx)]
            except (IndexError, ValueError, TypeError):
                continue
            counts[name] = counts.get(name, 0) + 1
        lane = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "CPU"
    else:
        counts = {}
        for c in state.current_chunks:
            counts[c.preferred_backend] = counts.get(c.preferred_backend, 0) + 1
        lane = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "CPU"

    if deployment_mode == DEPLOYMENT_MODE_WARM:
        techniques = tuple(sorted(WARM_TECHNIQUES))
    else:
        techniques = ()
    return CandidateSchedule(
        target_id=state.target_id,
        placements=((state.workload_id, lane, 0, techniques),),
    )


def step(
    state: LoopState,
    measurement: MeasurementRecord | None,
    *,
    cost_matrix: dict[str, Any],
    config: LoopConfig = LoopConfig(),
    scheduler_policy: SchedulerPolicy = SchedulerPolicy(),
    memory_policy: MemoryPlannerPolicy = MemoryPlannerPolicy(),  # noqa: ARG001
    memory_dir: Path | None = None,
    run_id: str | None = None,
) -> LoopState:
    """Run one feedback-loop iteration.

    Args:
        state: Current loop state.
        measurement: Optional ground-truth measurement for the *previous*
            schedule. ``None`` triggers a plan-only refresh:
            re-apply the (possibly externally updated) calibration and
            re-solve with no decision change.
        cost_matrix: Raw cost matrix; ``apply_calibration`` is reapplied
            each step so calibration updates flow through.
        config: Convergence / cap parameters.
        scheduler_policy: Policy used to pick a solver for the new
            ``n_partitions``.
        memory_policy: Reserved (see :func:`init_loop_state`).

    Returns:
        A new :class:`LoopState` with one extra :class:`LoopRound` in
        ``history`` and the appropriate ``status``.
    """

    if state.status in ("converged", "failed"):
        return state

    iteration = state.iteration + 1

    # Stage 1: absorb the measurement (if any) before re-planning.
    new_cal = state.current_calibration
    if measurement is not None:
        new_cal = update_from_measurement(state.current_calibration, measurement)

    calibrated = apply_calibration(
        new_cal,
        cost_matrix,
        workload_id=state.workload_id,
        deployment_mode=config.deployment_mode,
    )

    # Stage 3: chunk under the *current* max_chunk_ops (set by the
    # previous round's recompile decision, if any), then fuse adjacent
    # chunks where cross-backend transfer dominates the smaller chunk's
    # serial cost.
    plan = _propose(
        state.workload_id,
        calibrated,
        max_chunk_ops=state.current_max_chunk_ops,
        max_partitions=config.max_partitions,
    )
    if config.enable_fusion:
        dag_for_fusion = _build_dag(state.workload_id, calibrated)
        plan = apply_fusion(
            plan,
            transfer_matrix=dag_for_fusion.transfer_us,
            fusion_gain_threshold=config.fusion_gain_threshold,
        )

    # Stage 2/4: solve under the policy-selected solver, then TV.
    raw_solver_us, choice, tv_proved, schedule_dict = _solve_with_policy(
        state.workload_id,
        calibrated,
        plan.chunks,
        scheduler_policy=scheduler_policy,
        timeout_ms=config.cpsat_timeout_ms,
    )

    # Bug-2 fix: the solver returns a makespan computed from raw chain-sum
    # chunk durations only — the v3 calibration overhead and contention
    # factors sit on the cost_matrix dict but no downstream consumer
    # applies them. Without this post-processing the predicted makespan
    # stays nailed at the chain-sum total regardless of calibration
    # evolution. We recompute predicted = max over lanes of
    # ``(busy + overhead) * contention`` from the solver's assignment +
    # end times, then forward that as the loop's predicted makespan.
    predicted_us = _apply_calibration_to_makespan(
        workload_id=state.workload_id,
        calibration=new_cal,
        schedule_dict=schedule_dict,
        plan_chunks=plan.chunks,
        raw_solver_us=raw_solver_us,
        deployment_mode=config.deployment_mode,
    )

    # Measurement-first short-circuit: when a real on-board measurement
    # exists for this candidate schedule, the calibrated prediction is
    # only a prior; act on the measurement directly so the loop's
    # convergence rules consume ground truth.
    prediction_source: Literal["predicted", "measurement_cache", "mixed"] = "predicted"
    if config.measurement_first:
        cache_dir = config.measurement_cache_dir or DEFAULT_MEASUREMENT_CACHE_DIR
        try:
            mcache = load_measurement_cache(cache_dir, state.target_id)
            candidate = _candidate_from_state(
                state, deployment_mode=config.deployment_mode,
                schedule_dict=schedule_dict,
            )
            eval_ = evaluate_with_cache(
                candidate, mcache,
                predicted_makespan_fallback_us=predicted_us,
            )
            prediction_source = eval_["source"]  # type: ignore[assignment]
            if prediction_source == "measurement_cache":
                log.info(
                    "feedback_loop_measurement_short_circuit",
                    workload=state.workload_id,
                    target=state.target_id,
                    predicted_us=predicted_us,
                    measured_us=eval_["makespan_us"],
                )
                predicted_us = float(eval_["makespan_us"])
        except Exception as exc:  # noqa: BLE001 — cache is advisory.
            log.warning(
                "feedback_loop_measurement_cache_error",
                error=type(exc).__name__,
            )

    # Memory planning is not yet wired into the Stage-4 loop. Keep
    # ``tv_memory_proved`` truthy (no obligation can fail when none was
    # generated) but also raise ``tv_memory_skipped`` so audit consumers
    # can distinguish "TV ran and proved" from "TV never ran".
    # Follow-up: call ``plan_memory_greedy`` over per-chunk buffers and
    # run ``translation_validate_memory_plan`` here, then set
    # ``tv_memory_skipped=False`` and report the real proof status.
    tv_memory_proved = True
    tv_memory_skipped = True

    transfer_us_total = _estimate_transfer_us(plan.chunks, schedule_dict)

    # Multi-objective scoring: distil the solver outputs (plus any
    # deadlines/buffers wired in by the caller via schedule_dict
    # side-channels) into ScheduleMetrics, then evaluate against the
    # spec. The default spec is makespan-only, so legacy loops see
    # ``score == makespan_us``.
    objective_score: ObjectiveScore | None = None
    if schedule_dict:
        sched_metrics = compute_metrics(
            start_times=schedule_dict.get("start_times", {}),
            end_times=schedule_dict.get("end_times", {}),
            device_assignments=schedule_dict.get("device_assignments", {}),
            deadlines_us=schedule_dict.get("deadlines_us"),
            buffer_specs=schedule_dict.get("buffer_specs"),
            backend_power_proxy_w=schedule_dict.get("backend_power_proxy_w"),
            measured_makespans_us=schedule_dict.get(
                "measured_makespans_us", ()
            ),
        )
        objective_score = evaluate(config.multi_objective, sched_metrics)

    decision: DecisionNext
    reason: str
    new_error_history = state.error_history
    measured_us: float | None = None

    if measurement is None:
        decision = "recalibrate_only"
        reason = "no measurement; plan-only refresh"
    else:
        measured_us = measurement.measured_us
        decision, reason, new_error_history = _classify_decision(
            predicted_us=predicted_us,
            measurement=measurement,
            iteration=iteration,
            config=config,
            error_history=state.error_history,
            decision_history=state.decision_history,
            schedule_dict=schedule_dict,
            transfer_us_total=transfer_us_total,
            baseline_makespan_us=state.baseline_makespan_us,
        )

    # Multi-objective convergence gate: when the spec weights anything
    # beyond MAKESPAN, also require the score to be stable across the
    # convergence window (relative drift below
    # ``objective_convergence_band``) before we accept a ``converged``
    # decision. Makespan-only specs hit the early-out below and behave
    # identically to the legacy loop.
    if decision == "converged" and objective_score is not None:
        active = config.multi_objective.active_kinds()
        non_makespan_active = tuple(
            k for k in active if k != ObjectiveKind.MAKESPAN
        )
        if non_makespan_active:
            prior_scores = tuple(
                r.objective_score.score
                for r in state.history[-config.convergence_window:]
                if r.objective_score is not None
            )
            if prior_scores:
                ref = prior_scores[-1]
                drift = (
                    abs(objective_score.score - ref) / ref if ref > 0 else 0.0
                )
                if drift > config.objective_convergence_band:
                    decision = "recalibrate_only"
                    reason = (
                        f"objective_band: score_drift={drift:.3f} > "
                        f"{config.objective_convergence_band} (kinds="
                        f"{[str(k) for k in non_makespan_active]})"
                    )
                else:
                    reason = (
                        reason
                        + f"; objective_score={objective_score.score:.2f}, "
                        + f"drift={drift:.3f}"
                    )
            else:
                decision = "recalibrate_only"
                reason = (
                    "objective_band: no prior objective_score for drift check"
                )

    # Regression guard: refuse to converge on a schedule whose predicted
    # makespan exceeds ``regression_threshold * baseline``. The baseline
    # is a greedy-EFT lower-effort plan computed at init; if it wasn't
    # available (``inf``) the guard is disabled.
    regression_triggered = False
    if (
        decision == "converged"
        and math.isfinite(state.baseline_makespan_us)
        and predicted_us > config.regression_threshold * state.baseline_makespan_us
    ):
        decision = "abort"
        reason = (
            f"regression_guard: predicted={predicted_us:.1f}us > "
            f"{config.regression_threshold} * baseline="
            f"{state.baseline_makespan_us:.1f}us"
        )
        regression_triggered = True

    # Cycle breaker: if the *prior* decision history already shows
    # ``max_same_decision_retries`` consecutive repeats of this decision,
    # force one round of ``recalibrate_only`` before re-evaluating. The
    # check uses ``state.decision_history`` (pre-append) so the first
    # repeated firing is allowed and only the (N+1)-th gets broken.
    if _is_cycle(
        state.decision_history,
        decision,
        max_retries=config.max_same_decision_retries,
    ):
        log.info(
            "feedback_loop_cycle_break",
            decision=decision,
            history_tail=state.decision_history[-config.max_same_decision_retries:],
        )
        original = decision
        decision = "recalibrate_only"
        reason = (
            f"cycle_break: {original} fired {config.max_same_decision_retries} "
            "times in a row; forcing recalibrate_only"
        )

    # Status transition.
    if decision == "converged":
        status: LoopStatus = "converged"
    elif decision == "abort":
        # Regression-guard aborts are a real failure, not max_iter.
        if regression_triggered:
            status = "failed"
        else:
            status = "max_iter" if iteration >= config.max_iterations else "failed"
    else:
        status = "running" if iteration < config.max_iterations else "max_iter"

    # Apply re-compile decisions to the *next* round's chunk cap via an
    # additive perturbation. The round we just recorded used
    # ``state.current_max_chunk_ops``.
    next_max_chunk_ops = state.current_max_chunk_ops
    next_chunks = plan.chunks
    if decision == "recompile_finer":
        next_max_chunk_ops = _adjust_chunk_cap(
            state.current_max_chunk_ops,
            finer=True,
            step_size=config.granularity_perturbation_step,
        )
    elif decision == "recompile_coarser":
        next_max_chunk_ops = _adjust_chunk_cap(
            state.current_max_chunk_ops,
            finer=False,
            step_size=config.granularity_perturbation_step,
        )

    # Eagerly re-chunk so the next step's view is consistent with the
    # decision we just recorded (and so consumers reading current_chunks
    # see the post-decision plan).
    if next_max_chunk_ops != state.current_max_chunk_ops:
        next_plan = _propose(
            state.workload_id,
            calibrated,
            max_chunk_ops=next_max_chunk_ops,
            max_partitions=config.max_partitions,
        )
        next_chunks = next_plan.chunks

    record = LoopRound(
        iteration=iteration,
        predicted_makespan_us=predicted_us,
        measured_makespan_us=measured_us,
        n_partitions=plan.n_partitions,
        solver_choice=str(choice),
        tv_schedule_proved=tv_proved,
        tv_memory_proved=tv_memory_proved,
        decision_next=decision,
        reason=reason,
        timestamp=_utc_now_iso(),
        tv_memory_skipped=tv_memory_skipped,
        objective_score=objective_score,
        prediction_source=prediction_source,
    )
    log.info(
        "feedback_loop_step",
        workload=state.workload_id,
        target=state.target_id,
        iteration=iteration,
        predicted_us=predicted_us,
        measured_us=measured_us,
        decision_next=decision,
        status=status,
    )
    new_decision_history = state.decision_history + (decision,)
    # Keep ``consecutive_in_band`` synced with the new error history as a
    # convenience for legacy consumers reading state JSON; the active rule
    # is the window-based check in ``_convergence_check``.
    consecutive = 0
    for e in reversed(new_error_history):
        if e < config.epsilon:
            consecutive += 1
        else:
            break

    # Append this iteration's outcome to the cross-run bandit log. The
    # log is advisory; persistence errors are warned but never raised so
    # a disk problem can never block scheduling.
    try:
        mem_dir = memory_dir if memory_dir is not None else DEFAULT_LOOP_MEMORY_DIR
        workload_set_key = canonical_workload_set_key([state.workload_id])
        abs_err = new_error_history[-1] if new_error_history else None
        entry = MemoryEntry(
            target_id=state.target_id,
            workload_set_key=workload_set_key,
            run_id=run_id or _utc_now_iso(),
            iteration=iteration,
            max_chunk_ops=state.current_max_chunk_ops,
            fusion_gain_threshold=config.fusion_gain_threshold,
            solver_choice=str(choice),
            n_partitions=plan.n_partitions,
            predicted_makespan_us=predicted_us,
            measured_makespan_us=measured_us,
            abs_pct_error=(None if abs_err is None else float(abs_err) * 100.0),
            was_converged=(status == "converged"),
        )
        append_memory_entry(entry, Path(mem_dir))
    except Exception as exc:  # noqa: BLE001 — memory is advisory.
        log.warning("feedback_loop_memory_append_error", error=type(exc).__name__)

    return LoopState(
        schema_version=state.schema_version,
        workload_id=state.workload_id,
        target_id=state.target_id,
        iteration=iteration,
        current_calibration=new_cal,
        current_chunks=next_chunks,
        current_solver_choice=str(choice),
        current_predicted_makespan_us=predicted_us,
        current_max_chunk_ops=next_max_chunk_ops,
        history=state.history + (record,),
        status=status,
        consecutive_in_band=consecutive,
        error_history=new_error_history,
        decision_history=new_decision_history,
        baseline_makespan_us=state.baseline_makespan_us,
    )


def has_converged(state: LoopState) -> bool:
    """True iff the loop has reached the ``converged`` terminal state."""

    return state.status == "converged"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _chunk_to_dict(c: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "op_ids": list(c.op_ids),
        "preferred_backend": c.preferred_backend,
        "durations_us_by_backend": {
            k: (None if isinstance(v, float) and math.isinf(v) else float(v))
            for k, v in c.durations_us_by_backend.items()
        },
    }


def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    durations: dict[str, float] = {}
    for k, v in d.get("durations_us_by_backend", {}).items():
        durations[str(k)] = math.inf if v is None else float(v)
    return Chunk(
        chunk_id=str(d["chunk_id"]),
        op_ids=tuple(d.get("op_ids", [])),
        preferred_backend=str(d.get("preferred_backend", "UNKNOWN")),
        durations_us_by_backend=durations,
    )


def _objective_score_to_dict(s: ObjectiveScore | None) -> dict[str, Any] | None:
    if s is None:
        return None
    m = s.raw_metrics
    return {
        "score": s.score,
        "component_scores": dict(s.component_scores),
        "raw_metrics": {
            "makespan_us": m.makespan_us,
            "deadline_violations": m.deadline_violations,
            "deadline_violation_total_us": m.deadline_violation_total_us,
            "peak_memory_bytes": m.peak_memory_bytes,
            "energy_proxy_joules": m.energy_proxy_joules,
            "makespan_variance_us": m.makespan_variance_us,
        },
    }


def _dict_to_objective_score(d: dict[str, Any] | None) -> ObjectiveScore | None:
    if d is None:
        return None
    raw = d.get("raw_metrics", {})
    metrics = ScheduleMetrics(
        makespan_us=float(raw.get("makespan_us", 0.0)),
        deadline_violations=int(raw.get("deadline_violations", 0)),
        deadline_violation_total_us=float(raw.get("deadline_violation_total_us", 0.0)),
        peak_memory_bytes=int(raw.get("peak_memory_bytes", 0)),
        energy_proxy_joules=(
            None
            if raw.get("energy_proxy_joules") is None
            else float(raw["energy_proxy_joules"])
        ),
        makespan_variance_us=float(raw.get("makespan_variance_us", 0.0)),
    )
    return ObjectiveScore(
        score=float(d.get("score", 0.0)),
        component_scores={
            str(k): float(v) for k, v in d.get("component_scores", {}).items()
        },
        raw_metrics=metrics,
    )


def _round_to_dict(r: LoopRound) -> dict[str, Any]:
    return {
        "iteration": r.iteration,
        "predicted_makespan_us": r.predicted_makespan_us,
        "measured_makespan_us": r.measured_makespan_us,
        "n_partitions": r.n_partitions,
        "solver_choice": r.solver_choice,
        "tv_schedule_proved": r.tv_schedule_proved,
        "tv_memory_proved": r.tv_memory_proved,
        "tv_memory_skipped": r.tv_memory_skipped,
        "decision_next": r.decision_next,
        "reason": r.reason,
        "timestamp": r.timestamp,
        "objective_score": _objective_score_to_dict(r.objective_score),
        "prediction_source": r.prediction_source,
    }


def _dict_to_round(d: dict[str, Any]) -> LoopRound:
    return LoopRound(
        iteration=int(d["iteration"]),
        predicted_makespan_us=float(d["predicted_makespan_us"]),
        measured_makespan_us=(
            None if d.get("measured_makespan_us") is None
            else float(d["measured_makespan_us"])
        ),
        n_partitions=int(d["n_partitions"]),
        solver_choice=str(d["solver_choice"]),
        tv_schedule_proved=bool(d["tv_schedule_proved"]),
        tv_memory_proved=bool(d["tv_memory_proved"]),
        decision_next=d["decision_next"],
        reason=str(d.get("reason", "")),
        timestamp=str(d["timestamp"]),
        tv_memory_skipped=bool(d.get("tv_memory_skipped", True)),
        objective_score=_dict_to_objective_score(d.get("objective_score")),
        prediction_source=d.get("prediction_source", "predicted"),
    )


def _calibration_to_dict(model: CalibrationModel) -> dict[str, Any]:
    return {
        "schema_version": model.schema_version,
        "target_id": model.target_id,
        "overhead_us": {
            wid: dict(per_b) for wid, per_b in model.overhead_us.items()
        },
        "contention_factor": {
            wid: dict(per_b) for wid, per_b in model.contention_factor.items()
        },
        "contention_provenance": {
            wid: dict(per_b) for wid, per_b in model.contention_provenance.items()
        },
        "overhead_us_warm": {
            wid: dict(per_b) for wid, per_b in model.overhead_us_warm.items()
        },
        "contention_factor_warm": {
            wid: dict(per_b) for wid, per_b in model.contention_factor_warm.items()
        },
        "contention_provenance_warm": {
            wid: dict(per_b) for wid, per_b in model.contention_provenance_warm.items()
        },
        "history": [
            {
                "round": r.round,
                "workload_id": r.workload_id,
                "backend": r.backend,
                "predicted_us": r.predicted_us,
                "measured_us": r.measured_us,
                "per_op_sum_us": r.per_op_sum_us,
                "delta_overhead_us": r.delta_overhead_us,
                "timestamp": r.timestamp,
            }
            for r in model.history
        ],
        "created_at": model.created_at,
    }


def _dict_to_calibration(d: dict[str, Any]) -> CalibrationModel:
    history = tuple(
        CalibrationRound(
            round=int(r["round"]),
            workload_id=str(r["workload_id"]),
            backend=str(r["backend"]),
            predicted_us=float(r["predicted_us"]),
            measured_us=float(r["measured_us"]),
            per_op_sum_us=float(r["per_op_sum_us"]),
            delta_overhead_us=float(r["delta_overhead_us"]),
            timestamp=str(r["timestamp"]),
        )
        for r in d.get("history", [])
    )
    overhead = {
        str(wid): {str(k): float(v) for k, v in per_b.items()}
        for wid, per_b in d["overhead_us"].items()
    }
    contention = {
        str(wid): {str(k): float(v) for k, v in per_b.items()}
        for wid, per_b in d["contention_factor"].items()
    }
    provenance = {
        str(wid): {str(k): str(v) for k, v in per_b.items()}
        for wid, per_b in d.get("contention_provenance", {}).items()
    }
    overhead_warm = {
        str(wid): {str(k): float(v) for k, v in per_b.items()}
        for wid, per_b in d.get("overhead_us_warm", {}).items()
    }
    contention_warm = {
        str(wid): {str(k): float(v) for k, v in per_b.items()}
        for wid, per_b in d.get("contention_factor_warm", {}).items()
    }
    provenance_warm = {
        str(wid): {str(k): str(v) for k, v in per_b.items()}
        for wid, per_b in d.get("contention_provenance_warm", {}).items()
    }
    return CalibrationModel(
        schema_version=str(d.get("schema_version", CALIBRATION_SCHEMA_VERSION)),
        target_id=str(d["target_id"]),
        overhead_us=overhead,
        contention_factor=contention,
        history=history,
        created_at=str(d["created_at"]),
        contention_provenance=provenance,
        overhead_us_warm=overhead_warm,
        contention_factor_warm=contention_warm,
        contention_provenance_warm=provenance_warm,
    )


def state_to_dict(state: LoopState) -> dict[str, Any]:
    """Serialise a :class:`LoopState` to a JSON-typed dict."""

    return {
        "schema_version": state.schema_version,
        "workload_id": state.workload_id,
        "target_id": state.target_id,
        "iteration": state.iteration,
        "current_calibration": _calibration_to_dict(state.current_calibration),
        "current_chunks": [_chunk_to_dict(c) for c in state.current_chunks],
        "current_solver_choice": state.current_solver_choice,
        "current_predicted_makespan_us": state.current_predicted_makespan_us,
        "current_max_chunk_ops": state.current_max_chunk_ops,
        "history": [_round_to_dict(r) for r in state.history],
        "status": state.status,
        "consecutive_in_band": state.consecutive_in_band,
        "error_history": list(state.error_history),
        "decision_history": list(state.decision_history),
        "baseline_makespan_us": (
            None
            if not math.isfinite(state.baseline_makespan_us)
            else state.baseline_makespan_us
        ),
    }


def state_from_dict(d: dict[str, Any]) -> LoopState:
    """Deserialise a :class:`LoopState` from a typed dict."""

    version = d.get("schema_version")
    if version != LOOP_SCHEMA_VERSION:
        raise ValueError(
            f"loop schema mismatch: expected {LOOP_SCHEMA_VERSION}, got {version}"
        )
    return LoopState(
        schema_version=str(version),
        workload_id=str(d["workload_id"]),
        target_id=str(d["target_id"]),
        iteration=int(d["iteration"]),
        current_calibration=_dict_to_calibration(d["current_calibration"]),
        current_chunks=tuple(_dict_to_chunk(c) for c in d.get("current_chunks", [])),
        current_solver_choice=str(d.get("current_solver_choice", "")),
        current_predicted_makespan_us=(
            None if d.get("current_predicted_makespan_us") is None
            else float(d["current_predicted_makespan_us"])
        ),
        current_max_chunk_ops=int(d.get("current_max_chunk_ops", 16)),
        history=tuple(_dict_to_round(r) for r in d.get("history", [])),
        status=d.get("status", "init"),
        consecutive_in_band=int(d.get("consecutive_in_band", 0)),
        error_history=tuple(float(e) for e in d.get("error_history", [])),
        decision_history=tuple(str(s) for s in d.get("decision_history", [])),
        baseline_makespan_us=(
            math.inf
            if d.get("baseline_makespan_us") is None
            else float(d["baseline_makespan_us"])
        ),
    )


def save_loop_state(state: LoopState, path: Path) -> None:
    """Persist a loop state as JSON (gitignored under ``build/loops/``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_to_dict(state), indent=2, sort_keys=True))


def load_loop_state(path: Path) -> LoopState:
    """Load a loop state from JSON; raises if schema mismatch."""

    payload = json.loads(Path(path).read_text())
    return state_from_dict(payload)


def measurement_from_dict(d: dict[str, Any]) -> MeasurementRecord:
    """Build a :class:`MeasurementRecord` from a JSON-typed dict."""

    return MeasurementRecord(
        workload_id=str(d["workload_id"]),
        backend=str(d["backend"]),
        measured_us=float(d["measured_us"]),
        per_op_sum_us=float(d["per_op_sum_us"]),
        predicted_us=float(d["predicted_us"]),
        concurrent_workloads=tuple(str(w) for w in d.get("concurrent_workloads", ())),
        deployment_techniques=tuple(str(t) for t in d.get("deployment_techniques", ())),
    )


def measurement_to_dict(m: MeasurementRecord) -> dict[str, Any]:
    """Serialise a :class:`MeasurementRecord` to a JSON-typed dict."""

    return {
        "workload_id": m.workload_id,
        "backend": m.backend,
        "measured_us": m.measured_us,
        "per_op_sum_us": m.per_op_sum_us,
        "predicted_us": m.predicted_us,
        "concurrent_workloads": list(m.concurrent_workloads),
        "deployment_techniques": list(m.deployment_techniques),
    }


__all__ = [
    "DecisionNext",
    "LOOP_SCHEMA_VERSION",
    "LoopConfig",
    "LoopRound",
    "LoopState",
    "LoopStatus",
    "has_converged",
    "init_loop_state",
    "load_loop_state",
    "measurement_from_dict",
    "measurement_to_dict",
    "save_loop_state",
    "state_from_dict",
    "state_to_dict",
    "step",
]
