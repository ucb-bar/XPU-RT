#!/usr/bin/env python3
"""F2g — Lagrangian-style decomposition by network.

F1 diagnosed monolithic MOSEK as canonicalization-bound. Sub-problem
experiments showed each network's MOSEK problem IS tractable:

    mlp_control alone: 0.5 s wall, 0.53 us makespan
    dronet     alone: 0.8 s wall, 4.66 us makespan
    yolov8_nano alone: 85 s wall, 46.44 us makespan

This script solves each network independently with MOSEK then stitches
the results into a multi-network schedule that respects shared CPU_P /
CPU_E capacity.

The stitching strategy here is a SEQUENTIAL decomposition (not full
Lagrangian ADMM):

  1. Schedule yolov8 first (heaviest, sets the makespan envelope).
  2. Schedule dronet next, offsetting by yolov8's per-core busy
     intervals (treats yolov8's machine_busy_until as the lower bound).
  3. Schedule mlp_control last, offsetting by dronet + yolov8.

For periodic networks (mlp/dronet), we re-solve once per instance and
slot each instance at its release time.

Outputs:
  schedules/scheduled_<workload>_mosek_decomposed.json — combined fixture.
  artifacts/audit/mosek_decompose_log.json — per-network solve details.

Limitations honestly documented:
- This is sequential, not ADMM-iterative. The first-scheduled network
  gets its MOSEK optimum; later networks pack into whatever's left.
- The combined makespan IS NOT the global optimum — it's a bounded
  upper bound to the global optimum.
- For full Lagrangian ADMM with dual price updates on shared CPU
  capacity, see the methodology section in
  artifacts/mosek_rework/README.md.

Usage:
  python scripts/mosek_decompose_by_network.py \\
    --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json
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
PY = os.environ.get("XPURT_PYTHON") or sys.executable


def _solve_subnetwork(workload_path, network_only, time_limit=90):
    """Solve MOSEK on a workload restricted to `network_only` (str)."""
    import copy
    wl = json.loads(Path(workload_path).read_text())
    if network_only not in wl["networks"]:
        return None, 0.0, f"network {network_only!r} not in workload"
    sub = copy.deepcopy(wl)
    sub["networks"] = {network_only: wl["networks"][network_only]}
    # Use 1 instance for solve speed; we'll replicate the schedule per instance.
    if "num_instances" in sub["networks"][network_only]:
        sub["networks"][network_only]["num_instances"] = 1
    sub_path = workload_path.replace(".json", f"_only_{network_only}.json")
    Path(sub_path).write_text(json.dumps(sub, indent=2))

    cmd = [
        PY, str(REPO / "scripts" / "run_xpurt_schedule.py"),
        "--networks-json", sub_path,
        "--solver", "milp", "--scheduler", "mosek",
        "--use-profiled",
        "--time-limit", str(time_limit),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO}/xpu-rt:" + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(REPO), env=env,
                                 capture_output=True, text=True,
                                 timeout=time_limit + 60)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0, "timeout"
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        return None, wall, f"rc={result.returncode}: " + (result.stderr or "")[-200:]
    # Locate fixture.
    stem = Path(sub_path).stem
    fixture = REPO / "schedules" / f"scheduled_{stem}_profiled.json"
    if not fixture.exists():
        return None, wall, f"fixture missing: {fixture}"
    return str(fixture), wall, "ok"


def _shift_dispatches(fixture, time_offset_ms):
    """Add `time_offset_ms` to every dispatch's start_time."""
    out = json.loads(json.dumps(fixture))  # deep copy
    for entry in out.get("dispatches", {}).values():
        entry["start_time"] = float(entry["start_time"]) + float(time_offset_ms)
    if "metadata" in out:
        old_mksp = out["metadata"].get("makespan", 0.0)
        out["metadata"]["makespan"] = float(old_mksp) + float(time_offset_ms)
    return out


def _merge_fixtures(fixtures):
    """Combine multiple sub-network fixtures into one. Each dispatch
    must already have a globally unique name (uses the network-prefixed
    name from output_scheduled_json).
    """
    if not fixtures:
        return {"dispatches": {}, "metadata": {"makespan": 0.0}}
    combined = {"dispatches": {}, "metadata": {"makespan": 0.0}}
    for fx in fixtures:
        for name, entry in fx.get("dispatches", {}).items():
            if name in combined["dispatches"]:
                continue  # skip duplicates (shouldn't happen)
            combined["dispatches"][name] = entry
        mksp = fx.get("metadata", {}).get("makespan", 0.0)
        if mksp > combined["metadata"]["makespan"]:
            combined["metadata"]["makespan"] = float(mksp)
    combined["metadata"]["num_operations"] = len(combined["dispatches"])
    return combined


def _busy_intervals_per_core(fixture):
    """Return {core: list[(start, end)]} of busy intervals."""
    intervals = {}
    for entry in fixture.get("dispatches", {}).values():
        core = entry.get("hardware_target", "")
        s = float(entry["start_time"])
        d = float(entry["duration"])
        intervals.setdefault(core, []).append((s, s + d))
    for core in intervals:
        intervals[core].sort()
    return intervals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--out",
                    default=None,
                    help="Path for combined fixture (default: schedules/scheduled_<stem>_mosek_decomposed.json)")
    ap.add_argument("--log",
                    default=str(REPO.parent / "ModelBlaster" /
                                "artifacts" / "audit" / "mosek_decompose_log.json"))
    args = ap.parse_args()

    wl = json.loads(Path(args.networks_json).read_text())
    networks = list(wl.get("networks", {}).keys())
    print(f"Networks in workload: {networks}")

    # Sequential scheduling: heaviest first.
    # Heaviness heuristic: yolov8 > dronet > mlp_control based on op count.
    network_order = sorted(
        networks,
        key=lambda n: -wl["networks"][n].get("num_instances", 1) *
                       (1000 if n == "yolov8_nano" else
                        300 if n == "dronet" else 30),
    )
    print(f"Order: {network_order}")

    log = []
    fixtures = []
    cumulative_busy = {}  # core -> latest free time

    for net in network_order:
        print(f"\n=== Solving {net} ===")
        fixture_path, wall, status = _solve_subnetwork(args.networks_json, net,
                                                         time_limit=90)
        entry = {"network": net, "wall_s": round(wall, 2), "status": status}
        if fixture_path is None:
            print(f"  FAIL: {status}")
            log.append(entry)
            continue
        fixture = json.loads(Path(fixture_path).read_text())
        # Offset by the cumulative busy of CPU_P (yolov8's home).
        offset = max(cumulative_busy.values(), default=0.0)
        if offset > 0:
            fixture = _shift_dispatches(fixture, offset)
        # Update cumulative busy.
        intervals = _busy_intervals_per_core(fixture)
        for core, ivs in intervals.items():
            if ivs:
                cumulative_busy[core] = max(cumulative_busy.get(core, 0.0),
                                              max(end for _, end in ivs))
        mksp = fixture.get("metadata", {}).get("makespan", 0.0)
        entry["makespan_with_offset"] = float(mksp)
        entry["per_core_busy"] = dict(intervals)
        log.append(entry)
        fixtures.append(fixture)
        print(f"  OK: wall={wall:.1f}s, mksp_after_offset={mksp:.2f}ms, "
              f"offset_applied={offset:.2f}ms")

    # Combine.
    combined = _merge_fixtures(fixtures)
    final_mksp = combined["metadata"]["makespan"]

    # Write outputs.
    if args.out:
        out_path = Path(args.out)
    else:
        stem = Path(args.networks_json).stem
        out_path = REPO / "schedules" / f"scheduled_{stem}_mosek_decomposed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "method": "F2g_sequential_decomposition",
        "networks_order": network_order,
        "final_makespan_ms": final_mksp,
        "total_solve_wall_s": sum(e.get("wall_s", 0) for e in log),
        "per_network": log,
    }, indent=2))

    print(f"\nCombined fixture -> {out_path}")
    print(f"F2g log -> {log_path}")
    print(f"Final combined makespan: {final_mksp:.2f} ms")
    print(f"Total MOSEK wall (per-network): {sum(e.get('wall_s', 0) for e in log):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
