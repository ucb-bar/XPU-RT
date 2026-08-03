"""C1 — yolo_anchor policy.

Schedules the heaviest non-periodic network (yolov8) first against an
empty timeline, then layers periodic instances around it.

Implemented via the `greedy_periodic` solver, which is structurally
"non-periodic critical path first, periodic instances refined in
follow-up passes" (see scripts/run_xpurt_schedule.py:160-168
description). This matches the policy's intent: pick the big aperiodic
chain first, fill periodic slots into the residual.

Trade-off: when yolov8 is dense enough to span the whole window, the
periodic instances are squeezed and band misses surface — which the
honest-marking Phase A2 wiring reports cleanly via deadline_miss flags.

Pre-pass: none today. A natural extension is to manipulate the
workload to drop low-priority periodic instances entirely if Phase B1
feasibility says periods can't be honored; tracked as a follow-up.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._common import run_underlying_solver, summarize_fixture


def yolo_anchor(workload_path: str,
                  *,
                  workload_data: Optional[Dict[str, Any]] = None,
                  time_limit: float = 60.0,
                  ) -> Dict[str, Any]:
    if workload_data is None:
        workload_data = json.loads(Path(workload_path).read_text())

    policy_log = [{"action": "policy_anchor", "intent":
                   "anchor on heaviest aperiodic (yolov8) via greedy_periodic"}]

    t0 = time.perf_counter()
    fixture_path, wall, status = run_underlying_solver(
        workload_path, solver="greedy_periodic",
        time_limit=time_limit,
    )
    if fixture_path is None:
        return {
            "policy": "yolo_anchor",
            "status": status,
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }

    summary = summarize_fixture(fixture_path, workload_data, "yolo_anchor")
    return {
        "policy": "yolo_anchor",
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
