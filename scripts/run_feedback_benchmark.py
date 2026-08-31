#!/usr/bin/env python3
"""Run or evaluate a fair original-vs-feedback scheduler experiment.

Without ``--solve`` the manifest's existing cells are evaluated. With
``--solve``, each phase is first passed through ``profile_schedulers.py`` with
the same solver list, timeout, time limit, and periodic-iteration limit; the
resulting schedules are then evaluated by the exact same path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import feedback_benchmark  # noqa: E402


def _run_phase(manifest: dict, phase_name: str, work_dir: str) -> list[dict]:
    common = manifest["common"]
    phase = manifest["phases"][phase_name]
    out_dir = os.path.join(work_dir, phase_name)
    cmd = [
        sys.executable, os.path.join(_REPO, "scripts", "profile_schedulers.py"),
        "--networks-json", phase["networks_json"],
        "--solvers", ",".join(common["solvers"]),
        "--out-dir", out_dir,
        "--tag", phase.get("tag", phase_name),
        "--timeout", str(common.get("timeout_s", 0)),
        "--time-limit", str(common.get("solver_time_limit_s", 0)),
        "--max-periodic-iters", str(common.get("max_periodic_iters", 1)),
    ]
    if common.get("critical_models"):
        cmd += ["--critical-models", ",".join(common["critical_models"])]
    if common.get("heavy_model"):
        cmd += ["--heavy-model", common["heavy_model"]]
    if common.get("window_ms") is not None:
        cmd += ["--window-ms", str(common["window_ms"])]
    for key, flag in (("gen_root", "--gen-root"),
                      ("profile_target", "--profile-target"),
                      ("topo_tag", "--topo-tag"),
                      ("profile_hw", "--profile-hw"),
                      ("horizon_ms", "--horizon-ms"),
                      ("extra", "--extra")):
        if phase.get(key) is not None:
            cmd += [flag, str(phase[key])]
    subprocess.run(cmd, cwd=_REPO, check=True)

    with open(os.path.join(out_dir, "scheduler_sweep.csv")) as f:
        rows = list(csv.DictReader(f))
    by_solver = {r["solver"]: r for r in rows}
    cells = []
    for solver in common["solvers"]:
        row = by_solver[solver]
        status = "validated" if row["status"] == "ok" else row["status"]
        cell = {"solver": solver, "status": status,
                "detail": row.get("detail", ""), "wall_s": row.get("wall_s")}
        if status == "validated":
            cell["schedule"] = row["schedule_json"]
        cells.append(cell)
    return cells


def _snapshot(manifest: dict, result: dict, destination: str) -> None:
    """Copy the exact configs/schedules behind a verdict into tracked results."""
    os.makedirs(destination, exist_ok=True)
    for phase_name in ("original", "feedback"):
        phase = manifest["phases"][phase_name]
        src_cfg = phase["networks_json"] if os.path.isabs(phase["networks_json"]) \
            else os.path.join(_REPO, phase["networks_json"])
        dst_cfg = os.path.join(destination, f"{phase_name}_workload.json")
        shutil.copy2(src_cfg, dst_cfg)
        rel_cfg = os.path.relpath(dst_cfg, _REPO)
        phase["networks_json"] = rel_cfg
        result["phases"][phase_name]["networks_json"] = rel_cfg

        result_cells = {c["solver"]: c for c in result["cells"]
                        if c["phase"] == phase_name}
        for cell in phase["cells"]:
            if cell["status"] != "validated":
                continue
            src = cell["schedule"] if os.path.isabs(cell["schedule"]) \
                else os.path.join(_REPO, cell["schedule"])
            dst = os.path.join(destination,
                               f"{phase_name}_{cell['solver']}_schedule.json")
            shutil.copy2(src, dst)
            rel = os.path.relpath(dst, _REPO)
            cell["schedule"] = rel
            result_cells[cell["solver"]]["schedule"] = rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=None,
                    help="result JSON (default: beside the manifest)")
    ap.add_argument("--solve", action="store_true",
                    help="run both solver matrices before evaluating them")
    ap.add_argument("--work-dir", default="out/feedback_benchmark")
    ap.add_argument("--snapshot-dir", default=None,
                    help="copy exact workload/schedule inputs here and emit a "
                         "resolved manifest beside the result")
    args = ap.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if args.solve:
        work_dir = args.work_dir if os.path.isabs(args.work_dir) \
            else os.path.join(_REPO, args.work_dir)
        for phase in ("original", "feedback"):
            manifest["phases"][phase]["cells"] = _run_phase(
                manifest, phase, work_dir)

    try:
        result = feedback_benchmark.evaluate(manifest, _REPO)
    except feedback_benchmark.ManifestError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    out = args.out or os.path.join(os.path.dirname(manifest_path), "result.json")
    if args.snapshot_dir:
        snapshot_dir = args.snapshot_dir if os.path.isabs(args.snapshot_dir) \
            else os.path.join(_REPO, args.snapshot_dir)
        _snapshot(manifest, result, snapshot_dir)
        resolved = os.path.join(os.path.dirname(os.path.abspath(out)),
                                "experiment.resolved.json")
        with open(resolved, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {resolved}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{result['verdict']}: {result['why']}")
    for comparison in result["comparisons"]:
        print(f"  vs {comparison['baseline_solver']}: "
              f"{'ACCEPT' if comparison['accepted'] else 'REJECT'} -- "
              f"{comparison['why']}")
    print(f"wrote {out}")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
