"""C3 — critical_path_first policy.

Globally prioritizes ops on the critical path. Implemented via the
`heft` scheduler, whose `_upward_rank` priority is the textbook
critical-path heuristic (longest weighted path to the sink).

Trade-off: HEFT minimizes makespan, so on a workload where the
aperiodic critical path dominates (yolov8), it will pack the CP first
and trample periodic deadlines (the band audit shows this clearly —
heft posts the lowest makespan but the most deadline misses).

Policy log records the explicit choice so the comparison table can
read the structure off it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._common import run_underlying_solver, summarize_fixture


def critical_path_first(workload_path: str,
                          *,
                          workload_data: Optional[Dict[str, Any]] = None,
                          time_limit: float = 60.0,
                          ) -> Dict[str, Any]:
    if workload_data is None:
        workload_data = json.loads(Path(workload_path).read_text())

    policy_log = [{"action": "policy_anchor", "intent":
                   "CP-first global ordering via heft upward_rank"}]

    t0 = time.perf_counter()
    fixture_path, wall, status = run_underlying_solver(
        workload_path, solver="milp", scheduler="heft",
        time_limit=time_limit,
    )
    if fixture_path is None:
        return {
            "policy": "critical_path_first",
            "status": status,
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }

    summary = summarize_fixture(fixture_path, workload_data, "critical_path_first")
    return {
        "policy": "critical_path_first",
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
