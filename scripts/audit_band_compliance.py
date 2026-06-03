#!/usr/bin/env python3
"""Audit every registered scheduler against the band invariant.

Runs each solver on the headline 4 MLP + 2 Dronet + 1 Yolo workload (or
any workload JSON passed in), calls
`diagnostics.check_band_invariant`, and writes a CSV:

    solver, workload, n_ops, n_release_violations, n_deadline_violations,
    worst_release_overrun_us, worst_deadline_overrun_us,
    n_networks_with_misses, makespan_us, solve_wall_s, status

Solvers that error are recorded with status="error: <msg>" and zero
counts.

Usage:
    python scripts/audit_band_compliance.py \
        --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
        --out artifacts/audit_band_compliance.csv \
        [--solvers cpsat,greedy,decomposed,...] \
        [--skip mosek,gnn_placement,...] \
        [--reuse-fixtures schedules/]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Wire local xpu-rt to import path FIRST.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "xpu-rt"))

from diagnostics import check_band_invariant  # noqa: E402


# Solvers that are known-broken at the audit moment. Adding to this set
# means "the solver itself crashes before producing a fixture". Solvers
# that produce a *bad* fixture should NOT be skipped — that's the whole
# point of the audit.
DEFAULT_SKIP = {
    "mosek",          # known to diverge at >300 ops; Phase F1-F4 work
    "gnn_placement",  # ML placement scheduler — needs pre-trained model
    "rl_policy",      # RL scheduler — needs pre-trained policy
    "cost_model",     # ML cost model — needs training data
    "llm_ranker",     # routes to heft anyway; redundant
    "simulated_annealing",  # randomized; runs to convergence — slow
    "milp_gurobi",    # needs commercial backend
    "milp_highs",     # may not be installed
    "milp_scip",      # may not be installed
    "milp_cbc",       # may not be installed
    "cpsat_memory",   # variant — covered by cpsat
}


def _run_solver(solver_name, networks_json_path, schedules_dir):
    """Invoke `run_xpurt_schedule.py` for one solver. Returns
    (fixture_path, solve_wall_s, status_str)."""
    from subprocess import run, TimeoutExpired

    # The script has two flags. `--solver` chooses the top-level orchestration:
    #   milp           — single global solve via the registry-named scheduler
    #   greedy/greedy_periodic/decomposed — bespoke list schedulers
    # `--scheduler` selects which registry algorithm runs when --solver=milp.
    # Map each registry name to the right (--solver, --scheduler) pair.
    if solver_name == "greedy":
        flags = ["--solver", "greedy"]
    elif solver_name == "greedy_periodic":
        flags = ["--solver", "greedy_periodic"]
    elif solver_name == "decomposed":
        flags = ["--solver", "decomposed"]
    else:
        flags = ["--solver", "milp", "--scheduler", solver_name]
    cmd = [
        "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python",
        str(REPO_ROOT / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", networks_json_path,
        *flags,
        "--use-profiled",
        "--time-limit", "60",
    ]
    t0 = time.perf_counter()
    try:
        result = run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                     text=True, timeout=180)
    except TimeoutExpired:
        return None, time.perf_counter() - t0, "timeout"
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - t0, f"error: {exc}"
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        # Capture last 200 chars of stderr.
        err = (result.stderr or "")[-200:].replace("\n", " ")
        return None, wall, f"rc={result.returncode}: {err}"

    # The script writes `scheduled_<workload-stem>_<scheduler>_profiled.json`.
    # Use solver-specific path ONLY — the generic
    # scheduled_<stem>_profiled.json gets overwritten by every solver
    # run, so it's not a valid disambiguator.
    workload_stem = Path(networks_json_path).stem
    path = schedules_dir / f"scheduled_{workload_stem}_{solver_name}_profiled.json"
    if path.exists():
        return str(path), wall, "ok"
    return None, wall, "fixture missing (solver-specific path)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks-json", default=str(
        REPO_ROOT / "data" / "toplevel" / "networks_1yolo_4mlp_2dronet_firesim.json"
    ))
    parser.add_argument("--out", default=str(
        REPO_ROOT.parent / "ModelBlaster" / "artifacts" / "audit" / "band_compliance.csv"
    ))
    parser.add_argument("--solvers", default="",
                        help="Comma-separated subset; empty = all non-skipped")
    parser.add_argument("--skip", default=",".join(sorted(DEFAULT_SKIP)),
                        help="Comma-separated solver names to skip")
    parser.add_argument("--reuse-fixtures", default=str(REPO_ROOT / "schedules"),
                        help="Directory to check for pre-existing fixtures first")
    parser.add_argument("--rerun", action="store_true",
                        help="Even if fixture exists, re-run solver")
    args = parser.parse_args()

    networks_path = args.networks_json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workload_data = json.loads(Path(networks_path).read_text())
    workload_stem = Path(networks_path).stem

    # Resolve solver list.
    from schedulers import available_schedulers  # noqa: E402
    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())
    if args.solvers:
        solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    else:
        solvers = [s for s in available_schedulers() if s not in skip_set]

    schedules_dir = Path(args.reuse_fixtures)
    schedules_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for solver in solvers:
        # Pre-existing fixture? Only solver-specific paths count — the
        # generic `_profiled.json` is overwritten by the most-recent run
        # and can't be attributed to any one solver.
        fixture_path = None
        wall = 0.0
        status = "reused"
        if not args.rerun:
            c = schedules_dir / f"scheduled_{workload_stem}_{solver}_profiled.json"
            if c.exists():
                fixture_path = str(c)

        if fixture_path is None:
            print(f"[run] {solver}")
            fixture_path, wall, status = _run_solver(
                solver, networks_path, schedules_dir
            )
        else:
            print(f"[reuse] {solver} <- {fixture_path}")

        if fixture_path is None:
            rows.append({
                "solver": solver,
                "workload": workload_stem,
                "n_ops": 0,
                "n_release_violations": 0,
                "n_deadline_violations": 0,
                "worst_release_overrun_us": 0.0,
                "worst_deadline_overrun_us": 0.0,
                "n_networks_with_misses": 0,
                "makespan_us": 0.0,
                "solve_wall_s": round(wall, 3),
                "status": status,
            })
            continue

        try:
            fixture = json.loads(Path(fixture_path).read_text())
            report = check_band_invariant(fixture, workload_data,
                                          solver=solver,
                                          workload_label=workload_stem)
            row = report.as_row()
            row["makespan_us"] = float(fixture.get("metadata", {}).get("makespan", 0.0))
            row["solve_wall_s"] = round(wall, 3)
            row["status"] = status
            rows.append(row)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            rows.append({
                "solver": solver,
                "workload": workload_stem,
                "n_ops": 0,
                "n_release_violations": 0,
                "n_deadline_violations": 0,
                "worst_release_overrun_us": 0.0,
                "worst_deadline_overrun_us": 0.0,
                "n_networks_with_misses": 0,
                "makespan_us": 0.0,
                "solve_wall_s": round(wall, 3),
                "status": "audit_error",
            })

    # Write CSV.
    fieldnames = [
        "solver", "workload", "n_ops",
        "n_release_violations", "n_deadline_violations",
        "worst_release_overrun_us", "worst_deadline_overrun_us",
        "n_networks_with_misses", "makespan_us", "solve_wall_s", "status",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {out_path}")
    # Print sorted summary table.
    print()
    print(f"{'solver':<22s} {'n_ops':>6s} {'rel':>5s} {'dl':>5s} {'worst_dl':>10s} {'mksp':>10s} {'status':<16s}")
    for r in sorted(rows, key=lambda x: x["solver"]):
        print(f"{r['solver']:<22s} {r['n_ops']:>6d} {r['n_release_violations']:>5d} {r['n_deadline_violations']:>5d} {r['worst_deadline_overrun_us']:>10.3f} {r['makespan_us']:>10.3f} {r['status']:<16s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
