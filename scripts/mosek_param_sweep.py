#!/usr/bin/env python3
"""F2e — MOSEK solver-parameter sweep.

Tests parameter combinations that typically affect MILP convergence
on big-M-heavy scheduling formulations. For each combination, runs
MOSEK on a small (mlp_instances=1) variant of the headline workload
and records (status, wall, objective). The combinations that converge
fastest at small N are then promoted to the headline run.

Tested parameters:
  - MSK_DPAR_MIO_TOL_REL_GAP : 1e-4 (default tight) vs 1e-2 (loose 1%)
  - MSK_IPAR_PRESOLVE_USE    : ON vs OFF
  - MSK_IPAR_MIO_HEURISTIC_LEVEL : 1 (low) vs 5 (aggressive)
  - MSK_IPAR_NUM_THREADS     : 1 vs cpu_count() // 2
  - MSK_DPAR_OPTIMIZER_MAX_TIME : 60 s

Output:
  artifacts/audit/mosek_param_sweep.csv

Usage:
  python scripts/mosek_param_sweep.py [--workload <path>]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"


def _temp_env_solver(workload_path, params_env, time_limit):
    """Run scheduler with parameter env injection. We sneak the MOSEK
    params via an env var XPURT_MOSEK_PARAMS=key=value;key=value the
    wrapper inside scheduler.py reads."""
    cmd = [
        PY, str(REPO / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", workload_path,
        "--solver", "milp",
        "--scheduler", "mosek",
        "--use-profiled",
        "--time-limit", str(time_limit),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/xpu-rt:" + env.get("PYTHONPATH", "")
    env["XPURT_MOSEK_PARAMS"] = ";".join(f"{k}={v}" for k, v in params_env.items())
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(REPO),
                                 capture_output=True, text=True, env=env,
                                 timeout=time_limit + 60)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time_limit + 60.0,
                "objective_us": None, "tail": ""}
    wall = time.perf_counter() - t0
    tail = ((result.stdout or "")[-300:] + " | " +
             (result.stderr or "")[-200:]).replace("\n", " ")
    obj = None
    for line in (result.stdout or "").splitlines():
        if "makespan_us=" in line:
            try:
                obj = float(line.split("makespan_us=")[1].split()[0])
            except (ValueError, IndexError):
                pass
            break
    status = "ok" if result.returncode == 0 else f"rc={result.returncode}"
    if "Infeasible" in tail:
        status = "infeasible"
    if "TimeLimit" in tail or "time_limit_exceeded" in tail.lower():
        status = "time_limit"
    if "SolverError" in tail:
        status = "solver_error"
    return {"status": status, "wall_s": round(wall, 2),
            "objective_us": obj, "tail": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload",
                    default=str(REPO / "data" / "toplevel" /
                                "mosek_diag_mlp1_dr1.json"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out",
                    default=str(REPO.parent / "ModelBlaster" /
                                "artifacts" / "audit" / "mosek_param_sweep.csv"))
    args = ap.parse_args()

    # Param grid (small to keep wall reasonable).
    grid = {
        "MSK_DPAR_MIO_TOL_REL_GAP": [1e-4, 1e-2],
        "MSK_IPAR_PRESOLVE_USE": ["MSK_PRESOLVE_MODE_ON",
                                    "MSK_PRESOLVE_MODE_OFF"],
        "MSK_IPAR_MIO_HEURISTIC_LEVEL": [1, 5],
    }
    keys = list(grid.keys())
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        params["MSK_DPAR_OPTIMIZER_MAX_TIME"] = args.time_limit
        label = "/".join(f"{k.split('_',2)[-1][:8]}={v}" for k, v in params.items()
                          if k != "MSK_DPAR_OPTIMIZER_MAX_TIME")
        print(f"\n[run] {label}")
        r = _temp_env_solver(args.workload, params, args.time_limit)
        r["label"] = label
        for k, v in params.items():
            r[f"param_{k}"] = v
        rows.append(r)
        print(f"  status={r['status']} wall={r['wall_s']}s obj={r['objective_us']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "status", "wall_s", "objective_us"] + \
                 [f"param_{k}" for k in (keys + ["MSK_DPAR_OPTIMIZER_MAX_TIME"])] + \
                 ["tail"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote -> {out}")
    print()
    print(f"{'label':<60s} {'status':<14s} {'wall':>8s} {'obj_us':>14s}")
    for r in rows:
        obj = f"{r['objective_us']:.1f}" if r["objective_us"] else "n/a"
        print(f"{r['label']:<60s} {r['status']:<14s} {r['wall_s']:>8.1f} {obj:>14s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
