"""C4 — cpsat_unconstrained policy.

Control baseline: invokes CP-SAT directly on the unmodified workload,
with no fuse/shard pre-applied and no structural hints. Any policy
that beats this is paying for its structure; any that loses is wasting
structure.

CPSAT is the registry's exact-solver baseline (replacing MOSEK while
the latter's reformulation is in flight). It encodes max_end_t as a
hard constraint, so when this policy emits a fixture with deadline
misses, the workload itself is infeasible at the given frequency mix
— the audit must surface that as `solver_status: infeasible_via_audit`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._common import run_underlying_solver, summarize_fixture


def cpsat_unconstrained(workload_path: str,
                         *,
                         workload_data: Optional[Dict[str, Any]] = None,
                         time_limit: float = 60.0,
                         ) -> Dict[str, Any]:
    """Run CPSAT on the workload as-is. Returns the policy report."""
    if workload_data is None:
        workload_data = json.loads(Path(workload_path).read_text())

    t0 = time.perf_counter()
    fixture_path, wall, status = run_underlying_solver(
        workload_path, solver="milp", scheduler="cpsat",
        time_limit=time_limit,
    )
    if fixture_path is None:
        return {
            "policy": "cpsat_unconstrained",
            "status": status,
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": [],
        }

    summary = summarize_fixture(fixture_path, workload_data, "cpsat_unconstrained")
    return {
        "policy": "cpsat_unconstrained",
        "status": status,
        "solve_wall_s": round(wall, 3),
        "fixture_path": summary["fixture_path"],
        "makespan": summary["makespan"],
        "n_deadline_miss": summary["n_deadline_miss"],
        "n_release_viol": summary["n_release_viol"],
        "n_dispatches": summary["n_dispatches"],
        "n_shards_applied": 0,
        "n_fuses_applied": 0,
        "policy_log": [
            {"action": "use_underlying", "scheduler": "cpsat",
             "reason": "exact solver, no pre-applied structure",
             "delta": None}
        ],
    }
