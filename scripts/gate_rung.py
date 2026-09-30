#!/usr/bin/env python3
"""Gate a ladder rung EMPIRICALLY -- solve it and look, do not reason about it.

A rung is only worth running if BOTH gates pass:

  Gate 1 (AT STAKE)    the solved baseline, on PROFILE costs, misses at least one
                       deadline. Otherwise the loop can only win on terms nobody reads.
  Gate 2 (REACHABLE)   some lever set, scored on MEASURED BOARD costs, reaches zero
                       misses. Otherwise no scheduler and no feedback could ever succeed,
                       and a failure says nothing about the loop.

Gate 2 also buys tractability, which is not obvious and is worth stating: across the 204
CP-SAT certificates in this repo, a phase-1 objective of 0 came back OPTIMAL in 55 of 55
runs, while all 121 FEASIBLE runs had objective > 0 and a genuine open gap. CP-SAT is fast
at CONFIRMING an achievable target and slow at DISPROVING an unachievable one. So a
reachable rung is also a solvable one, and gating on reachability is not merely hygiene.

WHY THIS IS EMPIRICAL. Three analytical gates in a row got this wrong, and each time the
arithmetic was fine while the MODEL of "the baseline" was not:

  1. `one > window` -- ignored that a window may have to be widened for an unrelated
     reason (absorbing the t=0 cold-start burst), which then also clears the baseline.
  2. `one > period` -- false on a multicore machine: successive INSTANCES of a network are
     independent and the scheduler puts them on different harts. CP-SAT duly returned a
     baseline with zero misses by spreading ffn_block's three instances.
  3. `net_times()` vs the window -- that function returns a network's SERIAL sum, but the
     scheduler parallelises a network's DISPATCHES across cores even at width 1 each, so
     yolo's 47.73 ms never lands on a single core at all.

The only trustworthy test of "does the baseline miss" is to solve the baseline and look.

Usage (sweep one net's window, which is the usual shape of the question):
  scripts/gate_rung.py --spec data/toplevel/scaling/s5_solvable_reveal.json \
      --calibration results/codesign_feedback/k1_cal_s5_measured.json \
      --sweep-net ffn_block --windows 22,20,18,16,14,12 \
      --shard-sets yolov8_nano_64x96 yolov8_nano_64x96,ffn_block

Both gates must pass on the SAME row for the rung to be worth running.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from schedule_eval import summary  # noqa: E402

_venv = os.path.join(REPO, ".venv/bin/python")
PY = os.environ.get("XPURT_PY") or (_venv if os.path.exists(_venv) else sys.executable)

# XPURT_UNIFORM_PACKED_WIDTH: a packed-weight dispatch takes ONE width across its
# instances, which is what the codegen contract requires and what the board can build.
# XPURT_NO_COMPACT: keep the schedule as solved rather than compacting it, so the
# instance-miss count is the solver's answer and not a post-pass's.
ENV = dict(os.environ, XPURT_UNIFORM_PACKED_WIDTH="1", XPURT_NO_COMPACT="1")


def solve(spec: dict, tag: str, workdir: str, calibration: str | None = None,
          solver: str = "greedy", time_limit: int = 60) -> dict | None:
    """Solve one spec and return schedule_eval.summary, or None if no schedule came out.

    greedy by default and deliberately: this gate runs a handful of solves per candidate
    window and needs to be instant. It answers "does a lever set exist that reaches zero",
    which greedy can answer affirmatively; CP-SAT is for the run, not for the gate.
    """
    p = os.path.join(workdir, f"{tag}.json")
    with open(p, "w") as fh:
        json.dump(spec, fh, indent=1)
    cmd = [PY, "scripts/run_xpurt_schedule.py", "--networks-json", p, "--profiled",
           "--max-periodic-iters", "1", "--solver", solver, "--time-limit", str(time_limit)]
    if calibration:
        cmd += ["--board-calibration", calibration]
    subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    sched = os.path.join(REPO, f"schedules/scheduled_{tag}_{solver}_profiled.json")
    return summary(sched, p) if os.path.exists(sched) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="the rung to gate")
    ap.add_argument("--calibration", required=True,
                    help="MEASURED board costs for gate 2. Must come from a trace of THIS "
                         "rung: a calibration measured on a different rung is an "
                         "extrapolation, and mispredicted s5 by 1.66x when it was tried")
    ap.add_argument("--sweep-net", required=True, help="whose window to sweep")
    ap.add_argument("--windows", required=True, help="comma-separated candidate windows, ms")
    ap.add_argument("--shard-sets", nargs="+", required=True,
                    help="each a comma-separated net list to try as shard_only_networks")
    ap.add_argument("--pin", nargs="*", default=[],
                    help="NET=WINDOW to hold fixed while sweeping, so the swept net is the "
                         "only net whose at-stake status is in question")
    ap.add_argument("--solver", default="greedy", choices=["greedy", "cpsat"])
    ap.add_argument("--time-limit", type=int, default=60)
    a = ap.parse_args()

    base = json.load(open(a.spec))
    sets = [s.split(",") for s in a.shard_sets]
    pins = dict(kv.split("=", 1) for kv in a.pin)

    any_pass = False
    with tempfile.TemporaryDirectory(prefix="gate_rung_") as workdir:
        for w in [float(x) for x in a.windows.split(",")]:
            b = json.loads(json.dumps(base))
            for net, val in pins.items():
                b["networks"][net]["window_duration"] = float(val)
            b["networks"][a.sweep_net]["window_duration"] = w

            s = solve(b, "gate_base", workdir, solver=a.solver, time_limit=a.time_limit)
            baseline = s["instance_misses"] if s else None

            best = None
            for st in sets:
                c = json.loads(json.dumps(b))
                c["scheduler"] = dict(c.get("scheduler") or {})
                c["scheduler"]["machine_combination_mode"] = "shard"
                c["scheduler"]["shard_only_networks"] = st
                r = solve(c, "gate_fix", workdir, calibration=a.calibration,
                          solver=a.solver, time_limit=a.time_limit)
                if r and r.get("instance_misses") is not None:
                    if best is None or r["instance_misses"] < best[0]:
                        best = (r["instance_misses"], "+".join(st), r.get("misses_by_network"))

            ok = bool(baseline) and best is not None and best[0] == 0
            any_pass |= ok
            print(f"{a.sweep_net}_window={w:6.1f}  baseline_misses={baseline}  "
                  f"best_on_board={best}{'   <== BOTH GATES PASS' if ok else ''}", flush=True)

    if not any_pass:
        print("\nNo window passes both gates. The rung is mis-sized, not the loop: either "
              "every baseline already meets its deadlines (nothing at stake) or no lever "
              "set reaches zero on measured costs (unreachable). Adjust the PERIOD to put "
              "it at stake and the WINDOW to make it reachable -- those are different "
              "knobs and conflating them is what made the earlier hand-tuning fail.")
    return 0 if any_pass else 1


if __name__ == "__main__":
    sys.exit(main())
