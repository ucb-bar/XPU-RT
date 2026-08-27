#!/usr/bin/env python3
"""Run the B0-B4 baseline ladder on the K1 and report measured, not predicted.

Each rung answers a different question, and the point of the ladder is that a
rung which helps in the schedule may not help on the board:

  B0  static model-level placement -- whole model pinned to one cluster,
      no per-dispatch scheduling. The thing XPU-RT has to beat.
  B1  XPU-RT scheduling over the fixed dispatch graph, one implementation.
  B2  B1 plus per-dispatch implementation selection, from compile_advice.
  B3  B2 plus compiler granularity feedback (split / fuse).
  B4  B3 plus sharding.

Predicted and measured are reported side by side and never merged. A rung is
only claimed as an improvement if the *board* says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
import trace_metrics  # noqa: E402


def run_on_board(host, remote_root, schedule_remote, trace_remote, *,
                 cpu_p, cpu_e, variant_p, variant_e, timeout=1800):
    cmd = (f"cd {remote_root} && ./bin/merlin-dispatch-scheduler "
           f"{schedule_remote} local-task 1 1 0 "
           f"--vmfb_dir={remote_root} --cpu_p_cpu_ids={cpu_p} "
           f"--cpu_e_cpu_ids={cpu_e} --visible_cores=8 "
           f"--variant_p={variant_p} --variant_e={variant_e} --pin_per_core=1 "
           f"--trace_csv={remote_root}/{trace_remote}")
    p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode


def fetch(host, remote_root, name, local):
    subprocess.run(["scp", "-q", f"{host}:{remote_root}/{name}", local],
                   check=True, timeout=600)


def _periods_from_schedules(paths):
    """Union of metadata.periodic_networks across the rungs being run."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            md = json.load(open(p)).get("metadata", {})
        except (OSError, ValueError):
            continue
        for k, v in (md.get("periodic_networks") or {}).items():
            out[k] = float(v)
    return out


def summarise(trace_path, periods):
    """Measured metrics, delegated to xpu-rt/trace_metrics.py.

    This used to be a local copy of the instance-collapsing logic. There were
    three such copies -- here, in xpu-rt/metrics.py, and in
    scripts/plot_k1_evolution.py -- and they disagreed: metrics.py counted
    deadline misses per *dispatch* (B1: 160) while this one counted per
    *instance* (B1: 10), and neither reported a rate, so "10 misses" read like a
    near miss when it was in fact 10 of 10. One implementation now, with both
    definitions named distinctly.
    """
    rows = trace_metrics.read_trace(trace_path)
    if not rows:
        return None
    s = trace_metrics.summarise_trace(rows, periods)
    # Flat keys the existing table printer and baselines.json consumers expect.
    s["deadline_misses"] = s["instance_deadline_misses"]
    worst = [m["worst_lateness_ms"] for m in s["per_model"].values()]
    s["worst_lateness_ms"] = max(worst) if worst else 0.0
    return s


def predicted(schedule_path, periods):
    s = json.load(open(schedule_path))
    d = s["dispatches"]
    service = sum(float(v["duration"]) for v in d.values()) * 1000.0
    makespan = max(float(v["start_time"]) + float(v["duration"])
                   for v in d.values()) * 1000.0
    inst = defaultdict(lambda: -1e18)
    for v in d.values():
        j = v["job_name"]
        inst[j] = max(inst[j], float(v["start_time"]) + float(v["duration"]))
    misses = 0
    for job, en in inst.items():
        b = job.rstrip("0123456789") or job
        i = int(job[len(b):] or 0)
        T = periods.get(b)
        if T is not None and en > i * T + T:
            misses += 1
    return {"service_us": round(service, 1), "makespan_us": round(makespan, 1),
            "deadline_misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="k1")
    ap.add_argument("--remote-root", default="/root/mb_k1")
    ap.add_argument("--out-dir", default="artifacts/k1_run/baselines")
    ap.add_argument("--rungs", default="B0,B1,B2",
                    help="which rungs to run; each needs a staged schedule")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # rung -> (local schedule, remote schedule, cpu_p, cpu_e, variant_p, variant_e)
    RUNGS = {
        # B0 pins each model to a cluster with no per-dispatch scheduling: the
        # same graph, but every dispatch of a model on one core.
        "B0": ("schedules/scheduled_k1_B0_static.json", "schedule_b0.json",
               "0,1,2,3", "4,5,6,7", "RVV", "RVV"),
        "B1": ("schedules/scheduled_networks_k1_mlp_dronet_greedy_profiled.json",
               "schedule.json", "0,1,2,3", "4,5,6,7", "RVV", "RVV"),
        "B2": ("schedules/scheduled_k1_advice_applied.json",
               "schedule_advice.json", "0,1,2,3", "4,5,6,7", "RVV", "RVV"),
        # Not a rung of the compiler ladder: a scheduler-policy control. B1 uses
        # the makespan-minimising greedy solver, which is free to queue a 10 ms
        # periodic model behind a 22 ms convolution. This runs the same graph
        # and the same profiles through the periodic-aware solver instead, to
        # separate "the compiler needs to change" from "the policy was wrong".
        "P1": ("schedules/scheduled_networks_k1_mlp_dronet_greedy_periodic_profiled.json",
               "schedule_periodic.json", "0,1,2,3", "4,5,6,7", "RVV", "RVV"),
        # C2 is the only self-consistent configuration available with this
        # runner: one scheduler machine per worker pool, one physical core per
        # pool, timed with single-core profiles. The 8-machine rungs above
        # over-subscribe 4 machines onto 1 core, so their queueing figures
        # measure that mismatch as much as anything else.
        "C2": ("schedules/scheduled_networks_k1_2core_greedy_profiled.json",
               "schedule_2core.json", "0", "4", "RVV", "RVV"),
    }
    # Periods come from the schedule the rung actually ran, so adding a third
    # model does not silently skip it (a hardcoded two-model dict would drop
    # every yolo instance without a word).
    periods = _periods_from_schedules(
        [RUNGS[r][0] for r in a.rungs.split(",") if r in RUNGS])
    if not periods:
        periods = {"mlp": 10.0, "dronet": 33.3}
        print("WARN: no metadata.periodic_networks found; falling back to "
              f"{periods}", file=sys.stderr)
    results = {}
    for rung in [r for r in a.rungs.split(",") if r]:
        if rung not in RUNGS:
            print(f"SKIP {rung}: not defined (needs a schedule)", file=sys.stderr)
            continue
        local_sched, remote_sched, cp, ce, vp, ve = RUNGS[rung]
        if not os.path.exists(local_sched):
            print(f"SKIP {rung}: {local_sched} missing", file=sys.stderr)
            continue
        subprocess.run(["scp", "-q", local_sched,
                        f"{a.host}:{a.remote_root}/{remote_sched}"],
                       check=True, timeout=600)
        trace = f"trace_{rung}.csv"
        print(f"[{rung}] running on {a.host} ...", flush=True)
        rc = run_on_board(a.host, a.remote_root, remote_sched, trace,
                          cpu_p=cp, cpu_e=ce, variant_p=vp, variant_e=ve)
        if rc != 0:
            print(f"[{rung}] runner exited {rc}", file=sys.stderr)
            continue
        local_trace = os.path.join(a.out_dir, trace)
        fetch(a.host, a.remote_root, trace, local_trace)
        results[rung] = {
            "schedule": local_sched,
            "predicted": predicted(local_sched, periods),
            "measured": summarise(local_trace, periods),
        }

    out = os.path.join(a.out_dir, "baselines.json")
    json.dump(results, open(out, "w"), indent=1)

    print(f"\n{'rung':<5}{'pred service':>14}{'meas service':>14}"
          f"{'meas makespan':>15}{'queue%':>8}{'misses':>8}")
    for rung, r in results.items():
        m, p = r["measured"], r["predicted"]
        if not m:
            continue
        print(f"{rung:<5}{p['service_us']:>14.0f}{m['service_us']:>14.0f}"
              f"{m['makespan_us']:>15.0f}{m['queue_share_pct']:>8.1f}"
              f"{m['deadline_misses']:>8d}")

    # The per-model block is the one that answers the actual question. A raw
    # miss count hides whether a model failed once or failed every time, and it
    # says nothing about achieved frequency -- which is what a periodic
    # workload is specified in.
    print()
    for rung, r in results.items():
        if r["measured"]:
            print(trace_metrics.format_summary(rung, r["measured"]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
