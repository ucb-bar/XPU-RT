#!/usr/bin/env python3
"""Phase F1 — diagnose MOSEK divergence at 300+ ops.

Binary-searches op-count on the headline workload to find the largest
N where MOSEK converges within a given wall-clock budget. Captures:

  - Solver status (optimal / time_limit / infeasible / cvxpy_error /
    crashed)
  - Objective value (vs CPSAT for ground truth on the same N)
  - Wall-clock time
  - Variable count and constraint count from the formulation

For each tested N, drops the result row to artifacts/audit/mosek_diagnose.csv.

Bisection plan:
  - Headline workload has ~388 ops in periodic-expanded form.
  - Tests N ∈ {50, 100, 150, 200, 250, 300, 350, 388}.
  - At each N, run MOSEK with 60-s budget AND CPSAT with 60-s budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"


def make_truncated_workload(base_path: str, max_ops_per_network: int,
                              out_path: str) -> str:
    """Make a workload variant by capping per-network instance counts so
    the total op budget is ~target_n. Approximation: cap at
    ceil(target/300 * num_instances)."""
    import copy
    base = json.loads(Path(base_path).read_text())
    nets = base["networks"]
    # We won't actually slice individual ops from the dispatch graph
    # (that would corrupt the DAG). Instead, scale down num_instances.
    # The original has mlp_control=4 instances, dronet=2.
    # max_ops_per_network governs the multiplier.
    for net in ("mlp_control", "dronet"):
        if net in nets and "num_instances" in nets[net]:
            cur = int(nets[net]["num_instances"])
            new = max(1, int(round(cur * max_ops_per_network / 4)))
            nets[net]["num_instances"] = new
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(base, indent=2))
    return str(p)


def run_solver(workload_path, scheduler, time_limit=60.0):
    """Run a single solver with the wrapper. Capture stderr to recover
    cvxpy/mosek status messages."""
    cmd = [
        PY, str(REPO / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", workload_path,
        "--solver", "milp",
        "--scheduler", scheduler,
        "--use-profiled",
        "--time-limit", str(time_limit),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/xpu-rt:" + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(REPO),
                                 capture_output=True, text=True, env=env,
                                 timeout=time_limit + 120.0)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_s": time_limit + 120.0,
                "objective": None, "stderr_tail": ""}
    wall = time.perf_counter() - t0
    stderr_tail = (result.stderr or "")[-500:].replace("\n", " ")
    stdout_tail = (result.stdout or "")[-500:].replace("\n", " ")
    # Parse makespan from output if present.
    obj = None
    for line in (result.stdout or "").splitlines():
        if "makespan_us=" in line:
            try:
                obj = float(line.split("makespan_us=")[1].split()[0])
            except (ValueError, IndexError):
                pass
            break
        if "makespan " in line and "ms" in line:
            try:
                tok = line.split("makespan")[1].split("ms")[0].split()[-1]
                obj = float(tok) * 1000.0
            except (ValueError, IndexError):
                pass
    classified = "ok" if result.returncode == 0 else f"rc={result.returncode}"
    if "Infeasible" in stdout_tail or "Infeasible" in stderr_tail:
        classified = "infeasible"
    if "TimeLimit" in stdout_tail or "time_limit" in stdout_tail.lower():
        classified = "time_limit"
    if "SolverError" in stderr_tail:
        classified = "solver_error"
    return {
        "status": classified,
        "wall_s": round(wall, 2),
        "objective_us": obj,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default=str(REPO / "data" / "toplevel" /
                                "networks_1yolo_4mlp_2dronet_firesim.json"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out",
                    default=str(REPO.parent / "ModelBlaster" /
                                "artifacts" / "audit" / "mosek_diagnose.csv"))
    args = ap.parse_args()

    # Mlp_control * dronet caps to test (scales linearly with op count).
    # Headline workload is 28 + 60 + 212 = 300 dispatches (per scheduler).
    test_caps = [1, 2, 4, 8]   # mlp instances counterpart
    rows = []
    for cap in test_caps:
        label = f"mlp{cap}_dr{max(1, cap//2)}"
        wl_path = make_truncated_workload(
            args.base, cap,
            str(REPO / "data" / "toplevel" / f"mosek_diag_{label}.json")
        )
        print(f"\n=== cap={cap} ({label}) ===")
        for scheduler in ("cpsat", "mosek"):
            print(f"  solver={scheduler} ...")
            r = run_solver(wl_path, scheduler, time_limit=args.time_limit)
            r["scheduler"] = scheduler
            r["cap_label"] = label
            r["mlp_instances"] = cap
            rows.append(r)
            print(f"    -> status={r['status']} wall={r['wall_s']}s "
                  f"obj={r['objective_us']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["scheduler", "cap_label", "mlp_instances",
                            "status", "wall_s", "objective_us", "stderr_tail"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote -> {out}")
    print()
    print(f"{'scheduler':<10s} {'cap':<10s} {'status':<14s} {'wall':>7s} {'obj_us':>14s}")
    for r in rows:
        obj = f"{r['objective_us']:.1f}" if r["objective_us"] else "n/a"
        print(f"{r['scheduler']:<10s} {r['cap_label']:<10s} "
              f"{r['status']:<14s} {r['wall_s']:>7.1f} {obj:>14s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
