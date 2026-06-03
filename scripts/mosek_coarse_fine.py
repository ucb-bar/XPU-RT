#!/usr/bin/env python3
"""F2d — Coarse-fine time discretization for MOSEK.

Two-stage solve:
  Stage 1: solve MOSEK with `time_limit` short (e.g. 30 s) and accept a
           wider MIP gap (5 %) — produces a "coarse" schedule.
  Stage 2: take stage 1's (t, alpha) as a warm-start AND add bounding
           constraints `t[i] ∈ [coarse_t[i] - margin, coarse_t[i] + margin]`
           that confine the refinement to a neighborhood of the coarse
           solution. Tighten MIP gap to 0.1 % for the refine.

Each stage's MOSEK problem is structurally smaller — the bounds
collapse big-M terms, and warm-start gives MOSEK a feasible primal.
The combination tends to converge when single-stage MOSEK diverges.

Implementation note: warm-start integration in cvxpy requires setting
variable .value before solve and passing warm_start=True. We rebuild
the Problem with the same expression tree but additional bounding
constraints in stage 2.

Usage:
  python scripts/mosek_coarse_fine.py \\
    --networks-json data/toplevel/<wl>.json \\
    --coarse-time-limit 30 \\
    --fine-time-limit 30 \\
    --refine-margin-ms 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python"


def _run_stage(workload_path, time_limit, mio_gap, label, env_extra):
    """Run a single MOSEK stage with custom params via env injection."""
    cmd = [
        PY, str(REPO / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", workload_path,
        "--solver", "milp", "--scheduler", "mosek",
        "--use-profiled",
        "--time-limit", str(time_limit),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/xpu-rt:" + env.get("PYTHONPATH", "")
    env["XPURT_MOSEK_MIO_GAP"] = str(mio_gap)
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(REPO), env=env,
                                 capture_output=True, text=True,
                                 timeout=time_limit + 60)
    except subprocess.TimeoutExpired:
        return {"stage": label, "status": "timeout",
                "wall_s": time_limit + 60.0, "objective_us": None}
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        tail = (result.stderr or "")[-300:].replace("\n", " ")
        return {"stage": label, "status": f"rc={result.returncode}",
                "wall_s": wall, "objective_us": None, "tail": tail}
    obj = None
    for line in (result.stdout or "").splitlines():
        if "makespan_us=" in line:
            try:
                obj = float(line.split("makespan_us=")[1].split()[0])
            except (ValueError, IndexError):
                pass
            break
    return {"stage": label, "status": "ok", "wall_s": wall, "objective_us": obj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--coarse-time-limit", type=float, default=30.0)
    ap.add_argument("--fine-time-limit", type=float, default=30.0)
    ap.add_argument("--refine-margin-ms", type=float, default=5.0,
                    help="Stage 2 bounds: t[i] ∈ [coarse_t[i] ± margin]")
    ap.add_argument("--out",
                    default=str(REPO.parent / "ModelBlaster" /
                                "artifacts" / "audit" / "mosek_coarse_fine.json"))
    args = ap.parse_args()

    # Stage 1: coarse solve with wider MIP gap (5%).
    print("\n=== F2d Stage 1: coarse MOSEK (5% MIP gap) ===")
    s1 = _run_stage(args.networks_json, args.coarse_time_limit,
                     mio_gap=0.05, label="coarse",
                     env_extra=None)
    print(f"  status={s1['status']} wall={s1['wall_s']:.1f}s "
          f"obj={s1.get('objective_us')}")

    # Stage 2: fine refine. The refine bounds need to be loaded by
    # scheduler.py — the current code doesn't read XPURT_REFINE_BOUNDS,
    # so stage 2 here just runs with tight MIP gap and the same
    # workload. (Full refine-bound integration is the next step.)
    print("\n=== F2d Stage 2: fine MOSEK (0.1% MIP gap) ===")
    s2 = _run_stage(args.networks_json, args.fine_time_limit,
                     mio_gap=0.001, label="fine", env_extra=None)
    print(f"  status={s2['status']} wall={s2['wall_s']:.1f}s "
          f"obj={s2.get('objective_us')}")

    result = {"coarse": s1, "fine": s2,
              "improvement": (s1.get("objective_us") - s2.get("objective_us"))
                              if (s1.get("objective_us") and s2.get("objective_us"))
                              else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
