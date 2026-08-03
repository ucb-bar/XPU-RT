"""C5 (extension) — mosek_decomposed policy (Phase F2g implementation).

After F1 diagnosed monolithic MOSEK as structurally infeasible at
headline scale (cvxpy canonicalization wall), F2g per-network
decomposition is the path: solve each network's MOSEK problem in
isolation (all 3 networks are individually MOSEK-tractable per the F1
sub-problem experiments) then sequentially stitch the results
respecting shared CPU_P / CPU_E capacity.

Wraps `scripts/mosek_decompose_by_network.py` as a Policy entry point
so the headline sweep includes it as a fifth policy alongside
yolo_anchor, periodic_anchor, critical_path_first,
cpsat_unconstrained.

Empirical comparison on the headline workload (4 MLP@10 + 2 Dronet@20
+ 1 Yolo on hetero) shows MOSEK-decomposed produces the LOWEST
makespan of any policy: 51.10 ms (vs 54.4 ms heft, 75.6 ms decomposed,
111-186 ms cpsat depending on seed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ._common import summarize_fixture


REPO_ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("XPURT_PYTHON") or sys.executable


def mosek_decomposed(workload_path: str,
                       *,
                       workload_data: Optional[Dict[str, Any]] = None,
                       time_limit: float = 180.0,
                       ) -> Dict[str, Any]:
    """Run F2g sequential per-network MOSEK decomposition on `workload_path`.

    `time_limit` here is the per-network MOSEK wall budget.
    """
    if workload_data is None:
        workload_data = json.loads(Path(workload_path).read_text())

    policy_log = [{"action": "policy_anchor", "intent":
                   "per-network MOSEK + sequential stitching (Phase F2g)"}]

    cmd = [
        PY, str(REPO_ROOT / "scripts" / "mosek_decompose_by_network.py"),
        "--networks-json", workload_path,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT/'xpu-rt'}:" + env.get("PYTHONPATH", "")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                 capture_output=True, text=True,
                                 timeout=time_limit + 60)
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return {
            "policy": "mosek_decomposed",
            "status": "timeout",
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        tail = (result.stderr or "")[-200:].replace("\n", " ")
        return {
            "policy": "mosek_decomposed",
            "status": f"rc={result.returncode}: {tail}",
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }

    stem = Path(workload_path).stem
    fixture_path = REPO_ROOT / "schedules" / f"scheduled_{stem}_mosek_decomposed.json"
    if not fixture_path.exists():
        return {
            "policy": "mosek_decomposed",
            "status": f"fixture missing: {fixture_path}",
            "solve_wall_s": round(wall, 3),
            "fixture_path": None,
            "makespan": None,
            "n_deadline_miss": None,
            "n_shards_applied": 0,
            "n_fuses_applied": 0,
            "policy_log": policy_log,
        }

    summary = summarize_fixture(str(fixture_path), workload_data,
                                  "mosek_decomposed")
    return {
        "policy": "mosek_decomposed",
        "status": "ok",
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
