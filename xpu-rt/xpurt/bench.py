"""xpurt.bench — sweep multiple solvers over the same workload, emit per-solver reports.

Each (solver, rep) pair runs ``schedule()`` and produces a ``SchedulerReport``;
results are merged into one JSON keyed by solver. Useful for picking a
production solver and for tracking solve-time regressions over time.

Usage:
    python -m xpurt.bench \\
        --workload-script /path/to/build_workload.py \\
        --solvers MOSEK,HIGHS \\
        --reps 3 \\
        --time-limit 60 \\
        --out /tmp/sweep.json

The workload-script must export a callable ``make_workload() -> Workload``.
A fresh Workload is built per (solver, rep) — the MILP mutates state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Resolve flat-module imports without requiring the package to be installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT_FLAT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT_FLAT)


def _load_make_workload(script_path: str):
    spec = importlib.util.spec_from_file_location("__bench_workload__", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_workload"):
        raise AttributeError(f"{script_path} must define `make_workload()`")
    return mod.make_workload


def run_one(workload_factory, solver: str, time_limit: Optional[float],
            verbose: bool = False) -> Dict[str, Any]:
    """Build a fresh workload, schedule once, return the report dict + status."""
    from scheduler import schedule  # type: ignore
    from profiling import SchedulerReport  # type: ignore

    wl = workload_factory()
    t0 = time.time()
    try:
        t, alpha, _fused, _fmap = schedule(
            wl,
            cvxpy_solver=solver,
            time_limit=time_limit,
            verbose=verbose,
        )
        elapsed = time.time() - t0
    except Exception as exc:
        return {"solver": solver, "error": f"{type(exc).__name__}: {exc}",
                "solve_wall_s": time.time() - t0}

    status = getattr(wl, "solver_state", {}).get("problem_status", "unknown")
    report = getattr(wl, "solver_state", {}).get("report")
    if report is None:
        # Schedule succeeded but report wasn't built (rare — fall back).
        report = SchedulerReport.from_solver_state(
            wl, t, alpha, solver_name=solver, solve_wall_s=elapsed,
            solver_status=status,
        )
    return asdict(report)


def sweep(workload_script: str, solvers: List[str], reps: int,
          time_limit: Optional[float] = None,
          verbose: bool = False) -> Dict[str, Any]:
    """Run reps × solvers and return a {solver: [report, ...]} dict."""
    factory = _load_make_workload(workload_script)
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in solvers}
    for solver in solvers:
        for rep in range(reps):
            if verbose:
                print(f"=== {solver} rep {rep+1}/{reps} ===")
            out[solver].append(run_one(factory, solver, time_limit, verbose))
    return {"sweep": out, "workload_script": workload_script,
            "solvers": solvers, "reps": reps, "time_limit": time_limit}


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep solvers over a workload.")
    ap.add_argument("--workload-script", required=True,
                    help="Path to a Python script exporting make_workload() -> Workload")
    ap.add_argument("--solvers", default="MOSEK",
                    help="comma list of cvxpy solver names")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    result = sweep(args.workload_script, solvers, args.reps,
                   time_limit=args.time_limit, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
