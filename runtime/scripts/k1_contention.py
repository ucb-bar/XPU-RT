#!/usr/bin/env python3
"""Measure how much a co-runner slows a dispatch down on the K1.

Solo profiles are necessary and not sufficient: the scheduler places several
dispatches at once, and the predicted-vs-actual join on the first real run showed
the large convolutions running 17-25% slower than their solo profile. That gap is
either contention or a profiling error, and the only way to tell is to create the
contention deliberately.

Design: pin the dispatch under test to one core and run a co-runner on a chosen
other core, then compare against the same dispatch with nothing else running.
The co-runner is the *same* benchmark binary, so the interference is
representative of the workload rather than a synthetic memory hog.

Placements worth separating on this part:
  solo            nothing else running
  same cluster    co-runner shares the 512K L2 (cores 0-3, or 4-7)
  other cluster   co-runner has its own L2

If same-cluster and other-cluster slowdowns differ, the L2 is the mechanism and
the scheduler should prefer spreading across clusters. If they are the same, it
is DRAM or the interconnect and cluster placement does not help.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys

_ROW = re.compile(r"real_time_mean\s+([0-9.]+)\s*(ns|us|ms|s)")
_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def bench(host, rdir, tool, module, cpu, reps, co_cpu=None,
          co_module=None, timeout=900):
    """Run one benchmark on `cpu`, optionally with a co-runner on `co_cpu`."""
    esc = module.replace("$", r"\$")
    run = (f'{tool} --module="{esc}" --device=local-task '
           f'--task_topology_cpu_ids={{CPU}} --benchmark_repetitions={{R}}')
    if co_cpu is None:
        cmd = f"cd {rdir} && " + run.format(CPU=cpu, R=reps)
    else:
        # Start the co-runner first, give it a moment to reach steady state,
        # measure, then stop it. The co-runner's own output is discarded.
        co_mod = (co_module or module).replace("$", r"\$")
        co = (f'{tool} --module="{co_mod}" --device=local-task '
              f'--task_topology_cpu_ids={co_cpu} '
              f'--benchmark_repetitions={reps * 40}')
        cmd = (f"cd {rdir} && ({co} >/dev/null 2>&1 & echo $! > /tmp/co.pid) && "
               f"sleep 2 && " + run.format(CPU=cpu, R=reps) +
               " ; kill $(cat /tmp/co.pid) 2>/dev/null; true")
    p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    m = _ROW.search(p.stdout)
    if not m:
        return None
    return float(m.group(1)) * _TO_MS[m.group(2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="k1")
    ap.add_argument("--remote-dir", default="/root/mb_k1/bench/dronet_RVV")
    ap.add_argument("--tool", default="/root/mb_k1/tools/iree-benchmark-module")
    ap.add_argument("--modules", default=None,
                    help="comma-separated benchmark vmfb names; default: pick "
                         "the largest few automatically")
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--same-cluster-cpu", type=int, default=1)
    ap.add_argument("--other-cluster-cpu", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--co-module", default=None,
                    help="module the co-runner executes; defaults to a "
                         "different one from the module under test")
    ap.add_argument("--out", default="artifacts/k1_run/contention.json")
    a = ap.parse_args()

    if a.modules:
        mods = [m for m in a.modules.split(",") if m]
    else:
        p = subprocess.run(
            ["ssh", a.host, f"ls {a.remote_dir}/*dispatch_*_benchmark.vmfb"],
            capture_output=True, text=True, timeout=120)
        mods = [l.split("/")[-1] for l in p.stdout.split() if l.strip()]
        mods = mods[:6]
    if not mods:
        print("no benchmark modules found", file=sys.stderr)
        return 1

    results = []
    print(f"{'module':<46}{'solo ms':>10}{'same L2':>10}{'other L2':>10}"
          f"{'same/solo':>11}{'other/solo':>11}")
    for m in mods:
        solo = bench(a.host, a.remote_dir, a.tool, m, a.cpu, a.reps)
        if solo is None or solo <= 0:
            continue
        # A co-runner executing the SAME module shares its weights, so an
        # L2-sharing co-placement can look *helpful*. Use a different module so
        # the interference is competition, not constructive sharing -- which is
        # what concurrent models actually do.
        co = a.co_module or next((x for x in mods if x != m), m)
        same = bench(a.host, a.remote_dir, a.tool, m, a.cpu, a.reps,
                     co_cpu=a.same_cluster_cpu, co_module=co)
        other = bench(a.host, a.remote_dir, a.tool, m, a.cpu, a.reps,
                      co_cpu=a.other_cluster_cpu, co_module=co)
        if same is None or other is None:
            continue
        r = {"module": m, "solo_ms": solo, "same_cluster_ms": same,
             "other_cluster_ms": other,
             "same_ratio": same / solo, "other_ratio": other / solo}
        results.append(r)
        short = m.replace("_embedded_elf_riscv_64_benchmark.vmfb", "")
        print(f"{short:<46}{solo:>10.3f}{same:>10.3f}{other:>10.3f}"
              f"{same/solo:>10.3f}x{other/solo:>10.3f}x", flush=True)

    if results:
        s = statistics.median(r["same_ratio"] for r in results)
        o = statistics.median(r["other_ratio"] for r in results)
        print(f"\n  median slowdown, co-runner on the SAME cluster  : {s:.3f}x")
        print(f"  median slowdown, co-runner on the OTHER cluster : {o:.3f}x")
        print(f"  L2-attributable difference                      : {s/o:.3f}x")
        json.dump({"results": results,
                   "median_same_cluster_ratio": s,
                   "median_other_cluster_ratio": o,
                   "cpu": a.cpu, "same_cluster_cpu": a.same_cluster_cpu,
                   "other_cluster_cpu": a.other_cluster_cpu},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
