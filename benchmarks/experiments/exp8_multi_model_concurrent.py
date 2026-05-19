"""Experiment 8 — Multi-model concurrent scheduling at finer granularities.

Reproduces the QNN closed-loop target (1× yolov8n + 12× DroNet on
CPU/GPU/DSP) at four granularities and compares predicted makespans
across solvers (greedy, CP-SAT joint, MOSEK MILP).

The closed-loop already proved this fits at ``whole_net`` granularity
with ~305 ms predicted makespan over 4 rounds (final report at
``xpu-rt/data/profiled/qnn_closed_loop/final_report.md``). The
question this experiment asks: does cutting yolov8n into 16 / 64 / 273
partitions (and DroNet into 4 / 8 / 30 partitions) buy us schedule
quality, or does it just inflate solver time?

Inputs
------
* ``xpu-rt/data/profiled/qnn_cost_matrix.json`` — per-op CPU/GPU/DSP
  microsecond costs for yolov8n (273 ops) and dronet (30 ops).
* ``xpu-rt/data/profiled/qnn_e2e/measurements.json`` — measured end-
  to-end per-network costs (used for ``whole_net`` granularity).
* ``xpu-rt/data/profiled/qnn_closed_loop/contention.jsonl`` — round-4
  contention multipliers (CPU=1.228, DSP=0.852, GPU=1.0).

Output
------
``build/experiments/exp8_multi_model_concurrent/``:

* ``results.jsonl`` — one row per ``(granularity, solver)``.
* ``summary.md`` — comparison table + honest framing on the
  prediction-vs-measured gap.
* ``makespan_vs_granularity.png`` — optional plot.

Helper module note
------------------
A parallel agent is expected to land
``xpu_rt.scheduler.qnn_real_workload`` with ``make_chain_dag`` and
``chunk_dag``. As of this script's writing the module is absent, so a
local minimal fallback is provided in this file. If/when the upstream
module appears, this script will prefer it.

Usage
-----
    uv run python scripts/experiments/exp8_multi_model_concurrent.py [--quick]

``--quick`` runs only ``{whole_net, per_chunk_16}`` and aims for under
4 minutes. The full sweep adds ``per_chunk_64`` and ``per_op``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logger = structlog.get_logger(__name__)

# Device ordering used throughout this experiment.
DEVICES: tuple[str, ...] = ("CPU", "GPU", "DSP")
NUM_DEVICES = len(DEVICES)

# Round-4 converged contention multipliers from contention.jsonl.
CONTENTION_FACTORS: dict[str, float] = {"CPU": 1.2282003597512188, "GPU": 1.0, "DSP": 0.8515060362408278}

# Target makespan (yolov8n CPU baseline, in microseconds) and dronet deadline.
TARGET_MAKESPAN_US = 325_000.0
DRONET_DEADLINE_US = 40_000.0
N_DRONET_INSTANCES = 12

DATA_DIR = REPO_ROOT / "xpu-rt" / "data" / "profiled"
COST_MATRIX_PATH = DATA_DIR / "qnn_cost_matrix.json"
E2E_PATH = DATA_DIR / "qnn_e2e" / "measurements.json"

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp8_multi_model_concurrent"

# Solver budgets.
SOLVER_TIMEOUT_S = 60.0
MOSEK_MAX_N = 60


# ---------------------------------------------------------------------------
# Helper module — try upstream first, fall back to local minimal version.
# ---------------------------------------------------------------------------

try:  # pragma: no cover — upstream module may or may not exist yet.
    from xpu_rt.scheduler.qnn_real_workload import (  # type: ignore[import-not-found]
        chunk_dag as _upstream_chunk_dag,
    )
    from xpu_rt.scheduler.qnn_real_workload import (
        load_cost_matrix as _upstream_load_cost_matrix,
    )
    from xpu_rt.scheduler.qnn_real_workload import (
        make_chain_dag as _upstream_make_chain_dag,
    )

    HELPER_SOURCE = "xpu_rt.scheduler.qnn_real_workload"
except Exception:  # noqa: BLE001 — fall back is intentional.
    _upstream_make_chain_dag = None
    _upstream_chunk_dag = None
    _upstream_load_cost_matrix = None
    HELPER_SOURCE = "local-fallback"


@dataclass(frozen=True)
class WorkloadDag:
    """A schedulable DAG mirroring SyntheticDag's shape.

    Attributes:
        partition_ids: Stable topological order of partition IDs.
        dependencies: Map ``succ_id -> [pred_ids]``.
        durations_us_by_device: ``pid -> [CPU_us, GPU_us, DSP_us]``.
        num_devices: Always 3 in this experiment.
        transfer_us: 3x3 transfer cost matrix.
        name: Human-readable label.
        sub_dag_first_pid: Map ``sub_dag_label -> first pid`` (used to bound
            per-sub-DAG latency for deadline checks).
        sub_dag_last_pid: Map ``sub_dag_label -> last pid``.
    """

    partition_ids: list[str]
    dependencies: dict[str, list[str]]
    durations_us_by_device: dict[str, list[float]]
    num_devices: int
    transfer_us: list[list[float]]
    name: str
    sub_dag_first_pid: dict[str, str] = field(default_factory=dict)
    sub_dag_last_pid: dict[str, str] = field(default_factory=dict)


def _zero_transfer(num_devices: int) -> list[list[float]]:
    return [[0.0] * num_devices for _ in range(num_devices)]


def _make_chain_dag_local(
    workload_name: str,
    op_costs_us_by_device: list[list[float]],
    op_names: list[str],
    *,
    k_lookahead: int = 1,
) -> WorkloadDag:
    """Build a chain-DAG over an ordered list of ops.

    Each op depends on the previous ``k_lookahead`` ops (default 1 = pure chain).

    Args:
        workload_name: Used as ID prefix.
        op_costs_us_by_device: ``[op_idx][device_idx]`` cost in microseconds.
        op_names: Stable op names (one per op).
        k_lookahead: How many prior ops each op depends on.

    Returns:
        A :class:`WorkloadDag` with chain dependencies.
    """
    pids = [f"{workload_name}/{op_names[i]}#{i}" for i in range(len(op_names))]
    deps: dict[str, list[str]] = {}
    for i, pid in enumerate(pids):
        if i == 0:
            deps[pid] = []
        else:
            start = max(0, i - k_lookahead)
            deps[pid] = pids[start:i]
    durations = {pids[i]: list(op_costs_us_by_device[i]) for i in range(len(pids))}
    return WorkloadDag(
        partition_ids=pids,
        dependencies=deps,
        durations_us_by_device=durations,
        num_devices=NUM_DEVICES,
        transfer_us=_zero_transfer(NUM_DEVICES),
        name=workload_name,
        sub_dag_first_pid={workload_name: pids[0]} if pids else {},
        sub_dag_last_pid={workload_name: pids[-1]} if pids else {},
    )


def _chunk_dag_local(dag: WorkloadDag, n_chunks: int) -> WorkloadDag:
    """Coalesce a chain-DAG into ``n_chunks`` consecutive partitions.

    Costs sum per-device across the constituent ops. If ``n_chunks`` is
    >= the existing number of partitions, the original DAG is returned.
    """
    n = len(dag.partition_ids)
    if n_chunks >= n or n_chunks <= 0:
        return dag
    boundaries = [round(i * n / n_chunks) for i in range(n_chunks + 1)]
    new_pids: list[str] = []
    new_dur: dict[str, list[float]] = {}
    for ci in range(n_chunks):
        lo, hi = boundaries[ci], boundaries[ci + 1]
        if hi <= lo:
            continue
        chunk_pid = f"{dag.name}/chunk_{ci:03d}"
        new_pids.append(chunk_pid)
        sums = [0.0] * dag.num_devices
        for op_idx in range(lo, hi):
            costs = dag.durations_us_by_device[dag.partition_ids[op_idx]]
            for d in range(dag.num_devices):
                sums[d] += float(costs[d])
        new_dur[chunk_pid] = sums
    new_deps: dict[str, list[str]] = {new_pids[0]: []}
    for i in range(1, len(new_pids)):
        new_deps[new_pids[i]] = [new_pids[i - 1]]
    return WorkloadDag(
        partition_ids=new_pids,
        dependencies=new_deps,
        durations_us_by_device=new_dur,
        num_devices=dag.num_devices,
        transfer_us=dag.transfer_us,
        name=dag.name,
        sub_dag_first_pid={dag.name: new_pids[0]} if new_pids else {},
        sub_dag_last_pid={dag.name: new_pids[-1]} if new_pids else {},
    )


def _qnn_to_workload_dag(qnn_dag: Any, *, label: str, apply_contention: bool) -> WorkloadDag:
    """Adapt the upstream :class:`QnnDag` to our :class:`WorkloadDag`.

    Normalises ``None`` durations to ``math.inf`` (infeasible) and
    optionally multiplies per-backend costs by the converged contention
    factors. The contention factors are NOT applied to the cross-backend
    transfer matrix (those are not per-backend execution costs).
    """
    durations: dict[str, list[float]] = {}
    pids = list(qnn_dag.partition_ids)
    for pid in pids:
        row_in: list[float | None] = list(qnn_dag.durations_us_by_device[pid])
        row_out: list[float] = []
        for d, backend in enumerate(DEVICES):
            v = row_in[d]
            if v is None:
                row_out.append(math.inf)
            else:
                mult = CONTENTION_FACTORS[backend] if apply_contention else 1.0
                row_out.append(float(v) * mult)
        durations[pid] = row_out
    deps = {pid: list(qnn_dag.dependencies.get(pid, [])) for pid in pids}
    transfer = [row[:] for row in qnn_dag.transfer_us]
    return WorkloadDag(
        partition_ids=pids,
        dependencies=deps,
        durations_us_by_device=durations,
        num_devices=qnn_dag.num_devices,
        transfer_us=transfer,
        name=label,
        sub_dag_first_pid={label: pids[0]} if pids else {},
        sub_dag_last_pid={label: pids[-1]} if pids else {},
    )


def make_chain_dag(
    workload_name: str,
    op_costs_us_by_device: list[list[float]],
    op_names: list[str],
    *,
    k_lookahead: int = 1,
) -> WorkloadDag:
    """Local fallback: build a chain DAG from raw per-op rows.

    Costs passed here are assumed already contention-adjusted. Used only
    when the upstream helper is unavailable.
    """
    return _make_chain_dag_local(
        workload_name, op_costs_us_by_device, op_names, k_lookahead=k_lookahead
    )


def chunk_dag(dag: WorkloadDag, n_chunks: int) -> WorkloadDag:
    """Local fallback chunker over :class:`WorkloadDag`."""
    return _chunk_dag_local(dag, n_chunks)


# ---------------------------------------------------------------------------
# Cost-matrix loading and contention.
# ---------------------------------------------------------------------------


def _load_cost_matrix() -> dict[str, list[tuple[str, list[float]]]]:
    """Load and apply contention factors. Returns op (name, [cpu, gpu, dsp])."""
    with COST_MATRIX_PATH.open() as f:
        raw = json.load(f)
    out: dict[str, list[tuple[str, list[float]]]] = {}
    for net in ("yolov8n", "dronet"):
        ops = raw[net]
        ordered: list[tuple[str, list[float]]] = []
        for opname, costs in ops.items():
            if not isinstance(costs, dict):
                continue
            row: list[float] = []
            feasible = False
            for backend in DEVICES:
                val = costs.get(backend)
                if val is None:
                    # Mark infeasible with math.inf so the solver excludes.
                    row.append(math.inf)
                else:
                    row.append(float(val) * CONTENTION_FACTORS[backend])
                    feasible = True
            if not feasible:
                continue
            ordered.append((opname, row))
        out[net] = ordered
    return out


def _load_e2e() -> dict[str, dict[str, float]]:
    """Load per-network end-to-end mean costs in microseconds."""
    with E2E_PATH.open() as f:
        raw = json.load(f)
    matrix = raw["matrix"]
    out: dict[str, dict[str, float]] = {}
    for net, by_backend in matrix.items():
        out[net] = {}
        for backend in DEVICES:
            entry = by_backend.get(backend)
            if entry is None or not entry.get("ok", False):
                continue
            out[net][backend] = float(entry["mean_us"]) * CONTENTION_FACTORS[backend]
    return out


# ---------------------------------------------------------------------------
# Workload construction at four granularities.
# ---------------------------------------------------------------------------


def _empty_whole_net_dag(
    workload_name: str, e2e_us_by_backend: dict[str, float]
) -> WorkloadDag:
    """Build a single-partition DAG using measured E2E per-backend costs."""
    pid = f"{workload_name}/whole_net"
    row: list[float] = []
    for backend in DEVICES:
        row.append(e2e_us_by_backend.get(backend, math.inf))
    return WorkloadDag(
        partition_ids=[pid],
        dependencies={pid: []},
        durations_us_by_device={pid: row},
        num_devices=NUM_DEVICES,
        transfer_us=_zero_transfer(NUM_DEVICES),
        name=workload_name,
        sub_dag_first_pid={workload_name: pid},
        sub_dag_last_pid={workload_name: pid},
    )


def _build_per_op_dag(
    workload_name: str,
    op_costs: list[tuple[str, list[float]]],
    raw_cost_matrix: dict[str, dict[str, dict[str, float]]] | None,
) -> WorkloadDag:
    """Build a per-op chain DAG.

    When the upstream :mod:`xpu_rt.scheduler.qnn_real_workload` helper is
    available, route through it (and apply contention afterward); else
    fall back to the local builder using pre-multiplied costs.
    """
    if _upstream_make_chain_dag is not None and raw_cost_matrix is not None:
        qnn_dag = _upstream_make_chain_dag(
            workload_name,
            raw_cost_matrix,
            k_lookahead=1,
        )
        return _qnn_to_workload_dag(qnn_dag, label=workload_name, apply_contention=True)
    names = [op[0] for op in op_costs]
    rows = [op[1] for op in op_costs]
    return _make_chain_dag_local(workload_name, rows, names, k_lookahead=1)


def _chunk_workload_dag(dag: WorkloadDag, n_chunks: int) -> WorkloadDag:
    """Chunk a WorkloadDag, preferring the upstream chunker when possible.

    The upstream chunker operates on a :class:`QnnDag` (with ``None`` for
    unsupported cells). Our local fallback works on ``WorkloadDag``
    directly (with ``math.inf`` for unsupported cells). To keep the
    code path simple we always use the local chunker here — the
    semantic is equivalent because we already normalised ``None`` to
    ``math.inf`` and the local chunker preserves infeasibility via the
    same any-op-infeasible rule (sum-with-inf stays inf).
    """
    return _chunk_dag_local(dag, n_chunks)


def _merge_sub_dags(label: str, sub_dags: list[WorkloadDag]) -> WorkloadDag:
    """Stitch independent sub-DAGs into a single solver input.

    Each sub-DAG keeps its internal chain edges; no cross-DAG edges
    are added. Sub-DAG names are renamed to ``inst_{i}/...`` so the
    same workload (e.g. dronet) can appear N times.
    """
    all_pids: list[str] = []
    all_deps: dict[str, list[str]] = {}
    all_dur: dict[str, list[float]] = {}
    first_pids: dict[str, str] = {}
    last_pids: dict[str, str] = {}
    for inst_idx, dag in enumerate(sub_dags):
        prefix = f"inst{inst_idx:02d}::{dag.name}"
        rename: dict[str, str] = {pid: f"{prefix}#{pid}" for pid in dag.partition_ids}
        for pid in dag.partition_ids:
            new_pid = rename[pid]
            all_pids.append(new_pid)
            all_dur[new_pid] = list(dag.durations_us_by_device[pid])
            all_deps[new_pid] = [rename[p] for p in dag.dependencies.get(pid, [])]
        sub_label = prefix
        # We expect a single sub-DAG inside each input dag here.
        first_pid_orig = dag.sub_dag_first_pid.get(dag.name)
        last_pid_orig = dag.sub_dag_last_pid.get(dag.name)
        if first_pid_orig is not None and last_pid_orig is not None:
            first_pids[sub_label] = rename[first_pid_orig]
            last_pids[sub_label] = rename[last_pid_orig]
    transfer = _zero_transfer(NUM_DEVICES)
    return WorkloadDag(
        partition_ids=all_pids,
        dependencies=all_deps,
        durations_us_by_device=all_dur,
        num_devices=NUM_DEVICES,
        transfer_us=transfer,
        name=label,
        sub_dag_first_pid=first_pids,
        sub_dag_last_pid=last_pids,
    )


def build_workload_for_granularity(
    granularity: str,
    cost_matrix: dict[str, list[tuple[str, list[float]]]],
    e2e: dict[str, dict[str, float]],
    raw_cost_matrix: dict[str, dict[str, dict[str, float]]] | None,
) -> WorkloadDag:
    """Compose 1× yolov8n + 12× dronet at the requested granularity."""
    if granularity == "whole_net":
        yolo = _empty_whole_net_dag("yolov8n", e2e["yolov8n"])
        dronets = [_empty_whole_net_dag("dronet", e2e["dronet"]) for _ in range(N_DRONET_INSTANCES)]
    elif granularity == "per_chunk_16":
        yolo_full = _build_per_op_dag("yolov8n", cost_matrix["yolov8n"], raw_cost_matrix)
        yolo = _chunk_workload_dag(yolo_full, 16)
        dronet_full = _build_per_op_dag("dronet", cost_matrix["dronet"], raw_cost_matrix)
        dronet_chunked = _chunk_workload_dag(dronet_full, 4)
        dronets = [dronet_chunked for _ in range(N_DRONET_INSTANCES)]
    elif granularity == "per_chunk_64":
        yolo_full = _build_per_op_dag("yolov8n", cost_matrix["yolov8n"], raw_cost_matrix)
        yolo = _chunk_workload_dag(yolo_full, 64)
        dronet_full = _build_per_op_dag("dronet", cost_matrix["dronet"], raw_cost_matrix)
        dronet_chunked = _chunk_workload_dag(dronet_full, 8)
        dronets = [dronet_chunked for _ in range(N_DRONET_INSTANCES)]
    elif granularity == "per_op":
        yolo = _build_per_op_dag("yolov8n", cost_matrix["yolov8n"], raw_cost_matrix)
        dronet_full = _build_per_op_dag("dronet", cost_matrix["dronet"], raw_cost_matrix)
        dronets = [dronet_full for _ in range(N_DRONET_INSTANCES)]
    else:
        raise ValueError(f"unknown granularity: {granularity}")
    return _merge_sub_dags(f"yolo+12dronet@{granularity}", [yolo, *dronets])


# ---------------------------------------------------------------------------
# Solver wrappers — each returns start_times + end_times for deadline checks.
# ---------------------------------------------------------------------------


@dataclass
class ScheduleSolution:
    """Solver output normalised across greedy / CP-SAT / MOSEK."""

    makespan_us: float
    solver_time_ms: float
    status: str
    feasible: bool
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


def _is_feasible(val: float | None) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
        return False
    return val >= 0.0


def greedy_eft(dag: WorkloadDag) -> ScheduleSolution:
    """Earliest-finish-time list scheduler with full start/end tracking."""
    t0 = time.perf_counter()
    topo = _topo_order(dag.partition_ids, dag.dependencies)
    device_avail = [0.0] * dag.num_devices
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    chosen_dev: dict[str, int] = {}
    for pid in topo:
        durs = dag.durations_us_by_device[pid]
        best_end = math.inf
        best_dev = -1
        best_start = 0.0
        for d in range(dag.num_devices):
            dur = durs[d]
            if not _is_feasible(dur):
                continue
            ready = device_avail[d]
            for pred in dag.dependencies.get(pid, []):
                pred_end = end_times[pred]
                pred_dev = chosen_dev[pred]
                ready = max(ready, pred_end + dag.transfer_us[pred_dev][d])
            end = ready + float(dur)
            if end < best_end:
                best_end = end
                best_dev = d
                best_start = ready
        start_times[pid] = best_start
        end_times[pid] = best_end
        chosen_dev[pid] = best_dev
        device_avail[best_dev] = best_end
    makespan = max(end_times.values()) if end_times else 0.0
    return ScheduleSolution(
        makespan_us=makespan,
        solver_time_ms=(time.perf_counter() - t0) * 1000,
        status="optimal_local",
        feasible=True,
        start_times=start_times,
        end_times=end_times,
        device_assignments=chosen_dev,
    )


def cpsat_joint(dag: WorkloadDag, timeout_ms: int) -> ScheduleSolution:
    """Joint CP-SAT placement + ordering via xpu_rt.solve.schedule_joint_cpsat."""
    from xpu_rt.solve.schedule_joint_cpsat import solve_schedule_joint

    sol = solve_schedule_joint(
        partition_ids=dag.partition_ids,
        durations_us_by_device=dag.durations_us_by_device,
        dependencies=dag.dependencies,
        num_devices=dag.num_devices,
        transfer_us=dag.transfer_us,
        timeout_ms=timeout_ms,
    )
    return ScheduleSolution(
        makespan_us=sol.makespan_us,
        solver_time_ms=sol.solve_time_ms,
        status=sol.status,
        feasible=sol.feasible,
        start_times=dict(sol.start_times),
        end_times=dict(sol.end_times),
        device_assignments=dict(sol.device_assignments),
    )


def mosek_milp(dag: WorkloadDag, time_limit_s: float) -> ScheduleSolution:
    """Route the WorkloadDag through the CVXPY/MOSEK MILP scheduler."""
    import numpy as np

    from xpu_rt.scheduler.scheduler import schedule
    from xpu_rt.scheduler.workload import Operation, Workload

    pids = dag.partition_ids
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    ops: list[Operation] = []
    for i, pid in enumerate(pids):
        proc = [float(x) for x in dag.durations_us_by_device[pid]]
        ops.append(
            Operation(
                processing_times=proc,
                operation_id=i,
                operation_name=pid,
                job_id=0,
            )
        )
    for pid in pids:
        for pred in dag.dependencies.get(pid, []):
            ops[pid_to_idx[pid]].add_predecessor(ops[pid_to_idx[pred]])
    machines = [f"dev_{d}" for d in range(dag.num_devices)]
    transfer = np.asarray(dag.transfer_us, dtype=float)
    workload = Workload(ops, machines, transfer)
    t0 = time.perf_counter()
    silent_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(silent_buf):
            t_arr, alpha, _, _ = schedule(workload, time_limit=time_limit_s)
    except Exception as exc:  # noqa: BLE001
        return ScheduleSolution(
            makespan_us=float("inf"),
            solver_time_ms=(time.perf_counter() - t0) * 1000,
            status=f"error:{type(exc).__name__}",
            feasible=False,
            start_times={},
            end_times={},
            device_assignments={},
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    state = getattr(workload, "solver_state", {}) or {}
    status_str = str(state.get("problem_status", "unknown")).lower()
    if t_arr is None or alpha is None or status_str not in ("optimal", "optimal_inaccurate"):
        return ScheduleSolution(
            makespan_us=float("inf"),
            solver_time_ms=elapsed_ms,
            status=status_str or "infeasible",
            feasible=False,
            start_times={},
            end_times={},
            device_assignments={},
        )
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    dev_assign: dict[str, int] = {}
    for i, pid in enumerate(pids):
        k = int(alpha[i].argmax())
        s = float(t_arr[i])
        dur = float(dag.durations_us_by_device[pid][k])
        start_times[pid] = s
        end_times[pid] = s + dur
        dev_assign[pid] = k
    makespan = max(end_times.values()) if end_times else 0.0
    return ScheduleSolution(
        makespan_us=makespan,
        solver_time_ms=elapsed_ms,
        status=status_str,
        feasible=True,
        start_times=start_times,
        end_times=end_times,
        device_assignments=dev_assign,
    )


# ---------------------------------------------------------------------------
# Deadline accounting.
# ---------------------------------------------------------------------------


def _deadlines_met(
    dag: WorkloadDag,
    sol: ScheduleSolution,
    *,
    deadline_us: float = DRONET_DEADLINE_US,
) -> dict[str, Any]:
    """Count per-sub-DAG dronet deadline hits and report yolov8n latency."""
    if not sol.feasible:
        return {
            "dronet_met": 0,
            "dronet_total": N_DRONET_INSTANCES,
            "dronet_latencies_us": [],
            "yolov8n_latency_us": None,
            "all_met": False,
            "fit_target": False,
        }
    dronet_latencies: list[float] = []
    yolo_latency: float | None = None
    for label, first_pid in dag.sub_dag_first_pid.items():
        last_pid = dag.sub_dag_last_pid[label]
        start = sol.start_times.get(first_pid, 0.0)
        end = sol.end_times.get(last_pid, 0.0)
        latency = end - start
        if "dronet" in label:
            dronet_latencies.append(latency)
        elif "yolov8n" in label:
            yolo_latency = latency
    dronet_met = sum(1 for x in dronet_latencies if x <= deadline_us + 1e-6)
    return {
        "dronet_met": dronet_met,
        "dronet_total": len(dronet_latencies),
        "dronet_latencies_us": dronet_latencies,
        "yolov8n_latency_us": yolo_latency,
        "all_met": dronet_met == len(dronet_latencies),
        "fit_target": sol.makespan_us <= TARGET_MAKESPAN_US + 1e-6,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    granularity: str
    solver: str
    makespan_ms: float
    n_partitions: int
    solver_time_ms: float
    deadlines_met: int
    deadlines_total: int
    yolov8n_latency_ms: float | None
    fit_target: bool
    status: str
    note: str = ""


def _run_one(
    granularity: str,
    solver_name: str,
    dag: WorkloadDag,
    *,
    timeout_s: float,
    skip_reason: str | None = None,
) -> RowResult:
    n_part = len(dag.partition_ids)
    if skip_reason is not None:
        return RowResult(
            granularity=granularity,
            solver=solver_name,
            makespan_ms=math.nan,
            n_partitions=n_part,
            solver_time_ms=0.0,
            deadlines_met=0,
            deadlines_total=N_DRONET_INSTANCES,
            yolov8n_latency_ms=None,
            fit_target=False,
            status=skip_reason,
            note=skip_reason,
        )
    if solver_name == "greedy":
        sol = greedy_eft(dag)
    elif solver_name == "cpsat_joint":
        sol = cpsat_joint(dag, timeout_ms=int(timeout_s * 1000))
    elif solver_name == "mosek_milp":
        sol = mosek_milp(dag, time_limit_s=timeout_s)
    else:
        raise ValueError(solver_name)
    dl = _deadlines_met(dag, sol)
    return RowResult(
        granularity=granularity,
        solver=solver_name,
        makespan_ms=sol.makespan_us / 1000.0,
        n_partitions=n_part,
        solver_time_ms=sol.solver_time_ms,
        deadlines_met=dl["dronet_met"],
        deadlines_total=dl["dronet_total"],
        yolov8n_latency_ms=(
            dl["yolov8n_latency_us"] / 1000.0 if dl["yolov8n_latency_us"] is not None else None
        ),
        fit_target=dl["fit_target"],
        status=sol.status,
    )


def _granularities_for(quick: bool) -> list[str]:
    if quick:
        return ["whole_net", "per_chunk_16"]
    return ["whole_net", "per_chunk_16", "per_chunk_64", "per_op"]


def _solvers_for(granularity: str, n_partitions: int) -> Iterable[tuple[str, str | None]]:
    """Yield ``(solver_name, skip_reason)`` pairs.

    ``skip_reason`` is None when the solver should run.
    """
    yield ("greedy", None)
    yield ("cpsat_joint", None)
    if n_partitions <= MOSEK_MAX_N:
        yield ("mosek_milp", None)
    else:
        yield ("mosek_milp", f"skipped_size:n={n_partitions}>{MOSEK_MAX_N}")


def run_experiment(quick: bool) -> list[RowResult]:
    cost_matrix = _load_cost_matrix()
    e2e = _load_e2e()
    # Raw cost matrix (untouched µs costs, no contention applied) for the
    # upstream helper. The helper applies its own k-lookahead chain logic
    # but does NOT know about contention — we adjust on the way out.
    if _upstream_load_cost_matrix is not None:
        raw_cost_matrix = _upstream_load_cost_matrix(COST_MATRIX_PATH)
    else:
        with COST_MATRIX_PATH.open() as f:
            raw = json.load(f)
        raw_cost_matrix = {k: v for k, v in raw.items() if k != "_meta"}
    results: list[RowResult] = []
    for granularity in _granularities_for(quick):
        dag = build_workload_for_granularity(granularity, cost_matrix, e2e, raw_cost_matrix)
        n_part = len(dag.partition_ids)
        logger.info(
            "exp8.granularity.start",
            granularity=granularity,
            n_partitions=n_part,
            helper=HELPER_SOURCE,
        )
        for solver_name, skip in _solvers_for(granularity, n_part):
            logger.info(
                "exp8.solver.run",
                granularity=granularity,
                solver=solver_name,
                skip=skip,
            )
            row = _run_one(
                granularity,
                solver_name,
                dag,
                timeout_s=SOLVER_TIMEOUT_S,
                skip_reason=skip,
            )
            logger.info(
                "exp8.solver.done",
                granularity=granularity,
                solver=solver_name,
                makespan_ms=row.makespan_ms,
                deadlines=f"{row.deadlines_met}/{row.deadlines_total}",
                status=row.status,
                solver_time_ms=row.solver_time_ms,
            )
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _write_jsonl(rows: list[RowResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


# Closed-loop measured ranges (final_report.md rounds 1..4).
CLOSED_LOOP_MEASURED_MS = [254.8, 350.9, 255.6, 257.3]
CLOSED_LOOP_PREDICTED_MS = [354.9, 304.8, 356.7, 305.5]


def _fmt_ms(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.1f}"


def _write_summary(rows: list[RowResult], path: Path, quick: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_cell: dict[tuple[str, str], RowResult] = {(r.granularity, r.solver): r for r in rows}
    granularities = sorted({r.granularity for r in rows}, key=lambda g: (
        {"whole_net": 0, "per_chunk_16": 1, "per_chunk_64": 2, "per_op": 3}.get(g, 99)
    ))
    solvers = ["greedy", "cpsat_joint", "mosek_milp"]

    lines: list[str] = []
    lines.append("# Experiment 8 — Multi-model concurrent scheduling at finer granularity\n")
    mode = "quick (whole_net + per_chunk_16)" if quick else "full sweep"
    lines.append(f"_Mode: {mode}._\n")
    lines.append(f"_Helper module source: `{HELPER_SOURCE}`._\n")
    lines.append("")
    lines.append("**Target workload:** 1× yolov8n + 12× DroNet on QNN (CPU/GPU/DSP).")
    lines.append(f"**Target makespan:** ≤ {TARGET_MAKESPAN_US/1000:.0f} ms (yolov8n CPU baseline).")
    lines.append(f"**DroNet deadline:** ≤ {DRONET_DEADLINE_US/1000:.0f} ms per instance.")
    lines.append("")
    lines.append(
        "**Contention multipliers applied** (round 4 from `qnn_closed_loop/contention.jsonl`): "
        f"CPU={CONTENTION_FACTORS['CPU']:.3f}, GPU={CONTENTION_FACTORS['GPU']:.3f}, "
        f"DSP={CONTENTION_FACTORS['DSP']:.3f}."
    )
    lines.append("")

    # Headline table: makespan (ms) and dronet deadlines met.
    lines.append("## Predicted makespan (ms) and DroNet deadlines met")
    lines.append("")
    lines.append("| granularity | n_part | greedy ms | greedy DL | CP-SAT ms | CP-SAT DL | MOSEK ms | MOSEK DL |")
    lines.append("|---|---:|---:|:--:|---:|:--:|---:|:--:|")
    for g in granularities:
        any_row = next((by_cell[(g, s)] for s in solvers if (g, s) in by_cell), None)
        n_part = any_row.n_partitions if any_row is not None else 0
        cells: list[str] = [f"`{g}`", str(n_part)]
        for s in solvers:
            r = by_cell.get((g, s))
            if r is None:
                cells.extend(["—", "—"])
                continue
            if r.status.startswith("skipped_size") or r.status.startswith("skipped"):
                cells.extend(["skipped", "—"])
                continue
            cells.append(_fmt_ms(r.makespan_ms))
            cells.append(f"{r.deadlines_met}/{r.deadlines_total}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Solver time table.
    lines.append("## Solver wall time (ms)")
    lines.append("")
    lines.append("| granularity | greedy | CP-SAT | MOSEK |")
    lines.append("|---|---:|---:|---:|")
    for g in granularities:
        cells = [f"`{g}`"]
        for s in solvers:
            r = by_cell.get((g, s))
            if r is None or r.status.startswith("skipped"):
                cells.append("skip")
                continue
            cells.append(f"{r.solver_time_ms:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Headline: does finer granularity reduce predicted makespan?
    whole = by_cell.get(("whole_net", "cpsat_joint"))
    finest_g = granularities[-1]
    finest = by_cell.get((finest_g, "cpsat_joint"))
    lines.append("## Headline")
    if whole is not None and finest is not None and whole.granularity != finest.granularity:
        delta = finest.makespan_ms - whole.makespan_ms
        pct = (delta / whole.makespan_ms * 100.0) if whole.makespan_ms > 0 else 0.0
        verdict = (
            "finer granularity REDUCES predicted makespan"
            if delta < -1.0
            else "finer granularity is NOT buying us anything"
            if abs(delta) <= 1.0
            else "finer granularity INCREASES predicted makespan"
        )
        lines.append(
            f"- CP-SAT predicted makespan: `whole_net`={whole.makespan_ms:.1f} ms → "
            f"`{finest_g}`={finest.makespan_ms:.1f} ms (Δ={delta:+.1f} ms / {pct:+.1f}%). "
            f"**{verdict}**."
        )
    if whole is not None:
        lines.append(
            f"- `whole_net` CP-SAT predicted: {whole.makespan_ms:.1f} ms. "
            "Closed-loop round-4 prediction was **305.5 ms** "
            "(reproduction target)."
        )
    lines.append("")

    # MOSEK reach.
    lines.append("## MOSEK reach")
    mosek_ran: list[str] = []
    mosek_skipped: list[str] = []
    for g in granularities:
        r = by_cell.get((g, "mosek_milp"))
        if r is None:
            continue
        if r.status.startswith("skipped"):
            mosek_skipped.append(f"`{g}` ({r.note})")
        elif r.status in ("optimal", "optimal_inaccurate", "feasible"):
            mosek_ran.append(f"`{g}` ({r.makespan_ms:.1f} ms)")
        else:
            mosek_skipped.append(f"`{g}` (status={r.status})")
    lines.append(f"- ran on: {', '.join(mosek_ran) if mosek_ran else 'none'}")
    lines.append(f"- skipped / failed: {', '.join(mosek_skipped) if mosek_skipped else 'none'}")
    lines.append("")

    # Comparison to closed-loop measured.
    lines.append("## Closed-loop comparison")
    lines.append("")
    lines.append(
        "| round | predicted (ms) | measured (ms) |"
    )
    lines.append("|---:|---:|---:|")
    for i, (p, m) in enumerate(zip(CLOSED_LOOP_PREDICTED_MS, CLOSED_LOOP_MEASURED_MS), start=1):
        lines.append(f"| {i} | {p:.1f} | {m:.1f} |")
    lines.append("")
    if whole is not None:
        round4_pred = CLOSED_LOOP_PREDICTED_MS[-1]
        gap = whole.makespan_ms - round4_pred
        lines.append(
            f"`whole_net` CP-SAT prediction here is {whole.makespan_ms:.1f} ms vs closed-loop "
            f"round-4 prediction {round4_pred:.1f} ms (Δ={gap:+.1f} ms)."
        )
        if abs(gap) > 25.0:
            lines.append(
                "**Flag:** the >25 ms drift suggests the cost matrix or chain-DAG model "
                "is biased relative to the closed-loop's scheduler invocation."
            )
        else:
            lines.append("This is within ±25 ms of the closed-loop round-4 prediction.")
    lines.append("")

    # Honest gap framing.
    lines.append("## Honest framing on the prediction-vs-reality gap")
    lines.append("")
    lines.append(
        "The closed-loop run measured makespans in the 254.8–350.9 ms range while "
        "predicting 304.8–356.7 ms — roughly a 13–40 % error band per round. This "
        "experiment only produces *predictions* (no new measured runs on the QRB5165), "
        "so it cannot tighten that absolute gap. What it can do is show whether "
        "richer schedule structure (finer granularity) moves the *predicted* makespan "
        "toward or away from the measured band."
    )
    lines.append("")
    if whole is not None and finest is not None and whole.granularity != finest.granularity:
        measured_mean = sum(CLOSED_LOOP_MEASURED_MS) / len(CLOSED_LOOP_MEASURED_MS)
        d_whole = whole.makespan_ms - measured_mean
        d_finest = finest.makespan_ms - measured_mean
        narrowed = abs(d_finest) < abs(d_whole)
        lines.append(
            f"- measured mean across 4 rounds: {measured_mean:.1f} ms"
        )
        lines.append(
            f"- `whole_net` prediction overshoot vs measured mean: {d_whole:+.1f} ms"
        )
        lines.append(
            f"- `{finest_g}` prediction overshoot vs measured mean: {d_finest:+.1f} ms"
        )
        lines.append(
            "- finer-granularity scheduling "
            f"{'narrowed' if narrowed else 'did NOT narrow'} the gap to measured."
        )
    lines.append("")
    path.write_text("\n".join(lines))


def _maybe_plot(rows: list[RowResult], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        logger.info("exp8.plot.skipped", reason="matplotlib unavailable")
        return
    order = {"whole_net": 0, "per_chunk_16": 1, "per_chunk_64": 2, "per_op": 3}
    by_solver: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if math.isnan(r.makespan_ms):
            continue
        by_solver.setdefault(r.solver, []).append((order.get(r.granularity, 99), r.makespan_ms))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for solver_name, pts in by_solver.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=solver_name)
    for m in CLOSED_LOOP_MEASURED_MS:
        ax.axhline(m, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["whole_net", "per_chunk_16", "per_chunk_64", "per_op"])
    ax.set_ylabel("predicted makespan (ms)")
    ax.set_xlabel("granularity")
    ax.set_title("yolov8n + 12× DroNet: predicted makespan vs granularity")
    ax.axhline(TARGET_MAKESPAN_US / 1000.0, color="red", linestyle="--", linewidth=0.9, label="325 ms target")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run {whole_net, per_chunk_16}; aim for < 4 min wall time.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_experiment(quick=args.quick)
    _write_jsonl(rows, OUT_DIR / "results.jsonl")
    _write_summary(rows, OUT_DIR / "summary.md", quick=args.quick)
    _maybe_plot(rows, OUT_DIR / "makespan_vs_granularity.png")
    logger.info(
        "exp8.done",
        rows=len(rows),
        out_dir=str(OUT_DIR),
        helper=HELPER_SOURCE,
    )


if __name__ == "__main__":
    main()
