"""Phase D auxiliary — 9-mix × 4-policy ablation grid.

Holds frequency at the canonical (10ms, 20ms, 100ms) and varies the
network counts: MLP ∈ {2, 4, 8} × Dronet ∈ {1, 2, 4} × Yolo = 1.
Records makespan + deadline-miss + solve wall per cell. No Gantt
per cell — those are too many — just the summary table.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

REPO = Path("/scratch2/agustin/XPU-RT")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "xpu-rt"))

from policies import (cpsat_unconstrained, critical_path_first,
                      periodic_anchor, yolo_anchor)

POLICIES = {
    "yolo_anchor": yolo_anchor,
    "periodic_anchor": periodic_anchor,
    "critical_path_first": critical_path_first,
    "cpsat_unconstrained": cpsat_unconstrained,
}

MIX_GRID = [(m, d, 1) for m in (2, 4, 8) for d in (1, 2, 4)]


def materialise(base: Dict[str, Any], m: int, d: int, y: int) -> Dict[str, Any]:
    wl = copy.deepcopy(base)
    if "mlp_control" in wl["networks"]:
        wl["networks"]["mlp_control"]["num_instances"] = m
    if "dronet" in wl["networks"]:
        wl["networks"]["dronet"]["num_instances"] = d
    return wl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-workload", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()

    base = json.loads(args.base_workload.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    n_cells = len(MIX_GRID) * len(POLICIES)
    print(f"[mix-sweep] {n_cells} cells")
    cell_idx = 0
    for (m, d, y) in MIX_GRID:
        wl = materialise(base, m, d, y)
        for policy_name, fn in POLICIES.items():
            cell_idx += 1
            cell_lbl = f"m{m}_d{d}_y{y}__{policy_name}"
            cell_dir = args.out / "cells" / cell_lbl
            cell_dir.mkdir(parents=True, exist_ok=True)
            wl_path = cell_dir / "workload.json"
            wl_path.write_text(json.dumps(wl, indent=2))
            print(f"[mix-sweep] {cell_idx}/{n_cells} {cell_lbl}")
            t0 = time.perf_counter()
            try:
                res = fn(str(wl_path), time_limit=args.time_limit)
            except Exception as exc:
                res = {"status": f"error: {type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            res["solve_wall_s"] = round(time.perf_counter() - t0, 3)
            (cell_dir / "result.json").write_text(
                json.dumps(res, indent=2, default=str))
            rows.append({
                "cell": cell_lbl, "m": m, "d": d, "y": y, "policy": policy_name,
                "status": str(res.get("status", "?"))[:60],
                "makespan_us": res.get("makespan", ""),
                "n_deadline_miss": res.get("n_deadline_miss", ""),
                "n_release_viol": res.get("n_release_viol", ""),
                "n_dispatches": res.get("n_dispatches", ""),
                "solve_wall_s": res["solve_wall_s"],
            })

    fields = ["cell", "m", "d", "y", "policy", "status", "makespan_us",
              "n_deadline_miss", "n_release_viol", "n_dispatches",
              "solve_wall_s"]
    with open(args.out / "mix_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[mix-sweep] wrote {args.out / 'mix_grid.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
