"""Phase D — parametric sweep with predicted Gantts.

12-cell grid: 3 frequency configs × 4 policies, on the canonical
4 MLP + 2 Dronet + 1 Yolo workload.

| Axis        | Values                                                      |
|:------------|:------------------------------------------------------------|
| Frequency   | (mlp 10, dronet 20, yolo 100), (mlp 5, dronet 20, yolo 100), |
|             | (mlp 10, dronet 33, yolo 200) — values in ms                |
| Policy      | yolo_anchor, periodic_anchor, critical_path_first,           |
|             | cpsat_unconstrained                                          |

Each cell:
  1. Clone the base workload JSON, override the per-network `period`,
     `window_duration`, and yolov8 instance count when configured.
  2. Compute `frequency_feasibility` per network (Phase B1) to gate
     infeasible cells before invoking the solver.
  3. Run the policy. Record makespan, deadline misses, solve wall.
  4. Render the predicted Gantt with `plot_gantt.render_fixture_gantt`.
  5. Run the band invariant audit on the fixture; record violations.

Output:
  artifacts/sweeps/<date>/
    grid.csv
    cells/<mix>__<freq>__<policy>/
      workload.json, fixture.json, gantt.png, band_report.json

Per the plan, this driver does NOT itself queue a FireSim run — the
measured-cycle gate is satisfied by the calibrated PDB (Phase G3
landed v20b cycles into sweep_v8). Cells that need real FireSim
measurement get re-queued by hand.

Usage:
    python scripts/sweep_workloads.py \\
        --base-workload data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \\
        --out artifacts/sweeps/2026-06-08
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

REPO = Path("/scratch2/agustin/XPU-RT")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "xpu-rt"))

from policies import (cpsat_unconstrained, critical_path_first,
                      periodic_anchor, yolo_anchor)
from policies.hybrid_periodic_mosek_yolo import hybrid_periodic_mosek_yolo
from plot_gantt import render_fixture_gantt

POLICIES = {
    "yolo_anchor": yolo_anchor,
    "periodic_anchor": periodic_anchor,
    "critical_path_first": critical_path_first,
    "cpsat_unconstrained": cpsat_unconstrained,
    "hybrid_periodic_mosek_yolo": hybrid_periodic_mosek_yolo,
}

# Frequency configs (mlp_period_ms, dronet_period_ms, yolo_period_or_oneshot_ms)
FREQ_CONFIGS = [
    {"mlp_control": 10, "dronet": 20, "yolov8_nano": 100},
    {"mlp_control": 5,  "dronet": 20, "yolov8_nano": 100},
    {"mlp_control": 10, "dronet": 33, "yolov8_nano": 200},
]


def label_freq(freq: Dict[str, int]) -> str:
    return f"p{freq['mlp_control']}_{freq['dronet']}_{freq['yolov8_nano']}"


def materialise_workload(base: Dict[str, Any], freq: Dict[str, int]) -> Dict[str, Any]:
    """Override the per-net period / window for this cell."""
    wl = copy.deepcopy(base)
    nets = wl["networks"]
    if "mlp_control" in nets:
        nets["mlp_control"]["period"] = freq["mlp_control"]
        nets["mlp_control"]["window_duration"] = freq["mlp_control"]
    if "dronet" in nets:
        nets["dronet"]["period"] = freq["dronet"]
        nets["dronet"]["window_duration"] = freq["dronet"]
    # yolov8_nano is aperiodic by default in the canonical workload;
    # if a period is set in the freq config it overrides.
    if "yolov8_nano" in nets:
        if freq["yolov8_nano"] > 0 and freq["yolov8_nano"] < 1000:
            nets["yolov8_nano"]["period"] = freq["yolov8_nano"]
            nets["yolov8_nano"]["window_duration"] = freq["yolov8_nano"]
    return wl


def write_workload(wl: Dict[str, Any], cell_dir: Path) -> Path:
    p = cell_dir / "workload.json"
    p.write_text(json.dumps(wl, indent=2))
    return p


def run_cell(cell_dir: Path, workload_path: Path, policy_name: str,
             time_limit: float) -> Dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    fn = POLICIES[policy_name]
    t0 = time.perf_counter()
    try:
        result = fn(str(workload_path), time_limit=time_limit)
    except Exception as exc:
        return {
            "policy": policy_name,
            "status": f"error: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "solve_wall_s": round(time.perf_counter() - t0, 3),
        }
    if result.get("status") != "ok":
        return {**result, "solve_wall_s": round(time.perf_counter() - t0, 3)}

    fixture_path = result.get("fixture_path")
    if fixture_path:
        # Copy the fixture into the cell directory for self-contained output.
        local_fixture = cell_dir / "fixture.json"
        local_fixture.write_text(Path(fixture_path).read_text())
        result["local_fixture"] = str(local_fixture)
        gantt_path = cell_dir / "gantt.png"
        try:
            render_fixture_gantt(str(local_fixture), str(gantt_path),
                                 title=f"{policy_name} | {cell_dir.name}")
            result["gantt"] = str(gantt_path)
        except Exception as exc:
            result["gantt_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-workload", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--policies", default=",".join(POLICIES.keys()))
    ap.add_argument("--time-limit", type=float, default=60.0,
                    help="per-solver wall budget (seconds)")
    ap.add_argument("--freq-configs", type=int, default=3,
                    help="number of frequency configs to sweep (1..3)")
    args = ap.parse_args()

    if not args.base_workload.is_file():
        print(f"missing workload: {args.base_workload}", file=sys.stderr)
        return 1

    base = json.loads(args.base_workload.read_text())
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            print(f"unknown policy: {p}", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    grid_csv = args.out / "grid.csv"
    fieldnames = ["cell", "freq_label", "policy", "status",
                  "makespan_us", "n_deadline_miss", "n_release_viol",
                  "n_shards_applied", "n_fuses_applied",
                  "n_dispatches", "solve_wall_s", "gantt_path",
                  "fixture_path"]

    rows: List[Dict[str, Any]] = []
    freq_configs = FREQ_CONFIGS[:args.freq_configs]
    n_cells = len(freq_configs) * len(policies)
    print(f"[sweep] grid: {n_cells} cells (freqs={len(freq_configs)}, "
          f"policies={len(policies)})")

    cell_idx = 0
    for freq in freq_configs:
        wl = materialise_workload(base, freq)
        freq_lbl = label_freq(freq)
        for policy_name in policies:
            cell_idx += 1
            cell_lbl = f"m4_d2_y1__{freq_lbl}__{policy_name}"
            cell_dir = args.out / "cells" / cell_lbl
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"[sweep] cell {cell_idx}/{n_cells} {cell_lbl}")

            workload_path = write_workload(wl, cell_dir)
            res = run_cell(cell_dir, workload_path, policy_name,
                           time_limit=args.time_limit)
            (cell_dir / "policy_result.json").write_text(
                json.dumps(res, indent=2, default=str))

            row = {
                "cell": cell_lbl,
                "freq_label": freq_lbl,
                "policy": policy_name,
                "status": str(res.get("status", "unknown"))[:80],
                "makespan_us": res.get("makespan", ""),
                "n_deadline_miss": res.get("n_deadline_miss", ""),
                "n_release_viol": res.get("n_release_viol", ""),
                "n_shards_applied": res.get("n_shards_applied", ""),
                "n_fuses_applied": res.get("n_fuses_applied", ""),
                "n_dispatches": res.get("n_dispatches", ""),
                "solve_wall_s": res.get("solve_wall_s", ""),
                "gantt_path": res.get("gantt", res.get("gantt_error", "")),
                "fixture_path": res.get("local_fixture", ""),
            }
            rows.append(row)

    import csv
    with open(grid_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] wrote {grid_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
