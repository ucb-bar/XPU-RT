"""SchedulerReport — structured artifact for one scheduler invocation.

The report bundles everything a user might want to know about a single
``schedule()`` call so they can compare solvers, debug solve regressions, or
plug the predictions into a postmortem against measured cycles.

The data sources are:
- ``scheduler_name`` / ``solver_status`` / ``solve_wall_s`` from the caller
- everything else derived from ``compute_metrics(workload, t, alpha, ...)``
  plus a few extra aggregates this module adds (median duration, granularity
  buckets, raw per-op durations).

Persistence:
    SchedulerReport.from_solver_state(
        workload, t, alpha, solver_name="MOSEK", solve_wall_s=0.83,
        solver_status="optimal",
    ).write_json("/tmp/report.json")
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_metrics
from workload import Workload


GRANULARITY_BUCKETS = [
    ("lt_1k", 1_000),
    ("lt_10k", 10_000),
    ("lt_100k", 100_000),
    ("lt_1M", 1_000_000),
    ("ge_1M", float("inf")),
]


def _bucket_durations(durations: List[float]) -> Dict[str, int]:
    """Count durations into the granularity buckets above."""
    counts = {name: 0 for name, _ in GRANULARITY_BUCKETS}
    for d in durations:
        for name, ceil in GRANULARITY_BUCKETS:
            if d < ceil:
                counts[name] += 1
                break
        else:
            counts[GRANULARITY_BUCKETS[-1][0]] += 1
    return counts


def _feasible_targets(op: Any, machine_combinations: List[List[str]]) -> List[str]:
    """Machine names an op may legally run on.

    Derived from the complement of ``op.infeasible_combinations`` over all
    machine combinations, also skipping sentinel-cost (>=1e8) cells. Returns a
    sorted, de-duplicated list of machine names. Used by the advisor so it only
    ever proposes moving a dispatch to a backend it can actually run on.
    """
    names: set = set()
    pts = getattr(op, "processing_times", None)
    infeasible = getattr(op, "infeasible_combinations", set()) or set()
    for k, combo in enumerate(machine_combinations):
        if k in infeasible:
            continue
        if pts is not None and k < len(pts) and pts[k] >= 1e8:
            continue
        for m in combo:
            names.add(m)
    return sorted(names)


def _git_sha(repo_root: Optional[str] = None) -> str:
    cwd = repo_root or os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


@dataclass
class SchedulerReport:
    """One scheduler invocation's structured output."""

    schema_version: int
    solver_name: str
    solver_status: str
    solve_wall_s: float
    n_operations: int
    n_combinations: int
    n_resources_by_kind: Dict[str, int]
    makespan_cycles: float
    utilization: Dict[str, Dict[str, float]]
    granularity: Dict[str, Any]
    dispatch_durations: List[float]
    critical_path: float
    cross_device_transitions: int
    deadline_miss_count: int
    fusion_applied: bool
    fusion_map: Optional[Dict[str, Any]]
    git_sha: str
    captured_at: str
    # schema v2 (additive, optional so v1 readers/round-trips stay valid):
    # the frame deadline and the per-dispatch placement list the advisor needs.
    deadline_us: Optional[float] = None
    dispatches: Optional[List[Dict[str, Any]]] = None
    # schema v3 (additive): analytic lower-bound on makespan ("oracle floor")
    # so callers can read the solver's gap directly. The four fields are
    # critical-path / load / release components plus the unified
    # max-of-three. All in µs; None on workloads we couldn't analyze.
    oracle_floor_us: Optional[float] = None
    oracle_critical_path_us: Optional[float] = None
    oracle_load_us: Optional[float] = None
    oracle_release_us: Optional[float] = None
    oracle_gap_pct: Optional[float] = None

    @classmethod
    def from_solver_state(
        cls,
        workload: Workload,
        t: np.ndarray,
        alpha: np.ndarray,
        *,
        solver_name: str,
        solve_wall_s: float,
        solver_status: str = "unknown",
        fusion_map: Optional[Dict[str, Any]] = None,
    ) -> "SchedulerReport":
        """Build a report from a completed schedule().

        The (workload, t, alpha) triple is the canonical scheduler output;
        the rest are bookkeeping kwargs the caller already has.
        """
        m = compute_metrics(
            workload,
            t,
            alpha,
            scheduler_name=solver_name,
            solver_wall_time_s=solve_wall_s,
        )

        # Per-resource utilization in the {busy, idle, frac_busy} shape.
        # metrics.compute_metrics gives per-machine fractions; recompute
        # busy/idle in cycles for the report.
        makespan = float(m.get("makespan_us", 0.0))
        util_pct = m.get("per_machine_utilization", {})
        idle_us = m.get("per_machine_idle_us", {})
        utilization: Dict[str, Dict[str, float]] = {}
        for machine, frac in util_pct.items():
            busy = max(0.0, makespan - float(idle_us.get(machine, 0.0)))
            utilization[machine] = {
                "busy_cycles": busy,
                "idle_cycles": float(idle_us.get(machine, 0.0)),
                "frac_busy": float(frac),
            }

        # Resources grouped by kind prefix (CPU_P#0, CPU_P#1 -> {"CPU_P": 2}).
        n_resources_by_kind: Dict[str, int] = {}
        for machine in workload.machines:
            kind = machine.split("#", 1)[0]
            n_resources_by_kind[kind] = n_resources_by_kind.get(kind, 0) + 1

        # Granularity: numeric percentiles + bucket counts.
        machine_combinations = workload.get_machine_combinations()
        machines = workload.machines
        durations: List[float] = []
        for i, op in enumerate(workload.operations):
            combo_idx = int(np.argmax(alpha[i]))
            d = float(op.get_duration_for_combination(combo_idx, machine_combinations, machines))
            durations.append(d)

        granularity: Dict[str, Any] = {
            "p50": float(np.percentile(durations, 50)) if durations else 0.0,
            "p90": float(np.percentile(durations, 90)) if durations else 0.0,
            "p95": float(np.percentile(durations, 95)) if durations else 0.0,
            "p99": float(np.percentile(durations, 99)) if durations else 0.0,
            "mean": float(np.mean(durations)) if durations else 0.0,
            "max": float(np.max(durations)) if durations else 0.0,
            "buckets": _bucket_durations(durations),
        }

        # Per-dispatch placement list (schema v2). Reuses combo_idx/duration
        # already derived above; adds start/finish, dependency indices, and the
        # feasible-target set so the advisor can reason about and rebalance the
        # schedule without needing the live Workload.
        op_to_idx = {id(op): i for i, op in enumerate(workload.operations)}
        dispatches: List[Dict[str, Any]] = []
        op_deadlines: List[float] = []
        for i, op in enumerate(workload.operations):
            combo_idx = int(np.argmax(alpha[i]))
            combo = machine_combinations[combo_idx] if combo_idx < len(machine_combinations) else []
            start = float(t[i])
            d = float(durations[i])
            dl = getattr(op, "deadline_us", None)
            if dl is not None:
                op_deadlines.append(float(dl))
            dispatches.append({
                "id": i,
                "name": getattr(op, "operation_name", None) or f"op_{i}",
                "op": getattr(op, "operation_id", None),
                "target": "+".join(combo),
                "combo_idx": combo_idx,
                "start_us": start,
                "duration_us": d,
                "finish_us": start + d,
                "deps": [op_to_idx[id(p)] for p in op.get_predecessors() if id(p) in op_to_idx],
                "feasible_targets": _feasible_targets(op, machine_combinations),
                "deadline_us": (float(dl) if dl is not None else None),
            })
        report_deadline_us = min(op_deadlines) if op_deadlines else None

        n_combinations = len(machine_combinations) if machine_combinations else 0
        solver_state = getattr(workload, "solver_state", {}) or {}
        fusion_applied = bool(solver_state.get("fusion_applied", False))

        # Oracle floor (max of critical-path, load, release bounds). Pure
        # function of the workload, no solver in the loop. Guarded so a
        # bug in the floor analysis can't break SchedulerReport creation.
        try:
            from oracle import compute_floor, oracle_gap_pct as _gap_pct
            floor = compute_floor(workload)
            gap_pct_val = _gap_pct(makespan, floor["oracle_floor_us"])
        except Exception:
            floor = {}
            gap_pct_val = None

        return cls(
            schema_version=2,
            solver_name=solver_name,
            solver_status=solver_status,
            solve_wall_s=float(solve_wall_s),
            n_operations=int(m["num_operations"]),
            n_combinations=int(n_combinations),
            n_resources_by_kind=n_resources_by_kind,
            makespan_cycles=float(makespan),
            utilization=utilization,
            granularity=granularity,
            dispatch_durations=durations,
            critical_path=float(m.get("critical_path_us", 0.0)),
            cross_device_transitions=int(m.get("cross_device_transitions", 0)),
            deadline_miss_count=int(m.get("deadline_miss_count", 0)),
            fusion_applied=fusion_applied,
            fusion_map=fusion_map,
            git_sha=_git_sha(),
            captured_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            deadline_us=report_deadline_us,
            dispatches=dispatches,
            oracle_floor_us=floor.get("oracle_floor_us") if floor else None,
            oracle_critical_path_us=floor.get("critical_path_us") if floor else None,
            oracle_load_us=floor.get("load_us") if floor else None,
            oracle_release_us=floor.get("release_us") if floor else None,
            oracle_gap_pct=gap_pct_val,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
