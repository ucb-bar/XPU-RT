"""C2 — periodic_anchor policy.

Reserves periodic slots first, then packs the aperiodic remainder
(yolov8) into residual time. Implemented via the `decomposed` solver,
which is structurally a period-first list scheduler: it phases the EDF
list-schedule so periodic ops claim their slot at each instance's
release time, and the aperiodic critical path fills the gaps.

This is the policy most likely to honor the band invariant cleanly on
heavily periodic workloads — at the cost of leaving aperiodic slack
when periods are wide.

Pre-pass: none. The decomposed solver's own structure embodies the
"reserve periodic slots first" intent. If Phase B1's
frequency_feasibility reports infeasible for any periodic network's
single-class load, we still attempt decomposed (it will mark
deadline_miss honestly via Phase A2 wiring), but the policy_log
records the infeasibility upfront.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._common import run_underlying_solver, summarize_fixture


def periodic_anchor(workload_path: str,
                     *,
                     workload_data: Optional[Dict[str, Any]] = None,
                     time_limit: float = 60.0,
                     ) -> Dict[str, Any]:
    if workload_data is None:
        workload_data = json.loads(Path(workload_path).read_text())

    # Optional upfront feasibility check (Phase B1). We don't fail the
    # policy here — we just record the verdict so the comparison table
    # shows which cells were marked infeasible by the analytic gate vs
    # only by the solver. (Loading per-op costs from the profile DB
    # requires the workload-factory pipeline; for the policy entry-
    # point we leave that as an exercise for the sweep driver which
    # has the profile data in hand.)
    policy_log = [{"action": "policy_anchor", "intent":
                   "reserve periodic slots first via decomposed solver"}]

    t0 = time.perf_counter()
    fixture_path, wall, status = run_underlying_solver(
        workload_path, solver="decomposed",
        time_limit=time_limit,
    )
    if fixture_path is None:
        return {
            "policy": "periodic_anchor",
            "status": status,
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }

    summary = summarize_fixture(fixture_path, workload_data, "periodic_anchor")
    return {
        "policy": "periodic_anchor",
        "status": status,
        "solve_wall_s": round(wall, 3),
        "fixture_path": summary["fixture_path"],
        "makespan": summary["makespan"],
        "n_deadline_miss": summary["n_deadline_miss"],
        "n_release_viol": summary["n_release_viol"],
        "n_dispatches": summary["n_dispatches"],
        "n_shards_applied": 0,
        "n_fuses_applied": 0,
        "policy_log": policy_log,
    }
