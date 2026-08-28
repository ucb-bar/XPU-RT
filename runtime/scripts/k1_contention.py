#!/usr/bin/env python3
"""Measure how much co-runners slow a dispatch down on the K1.

Solo profiles are necessary and not sufficient: the scheduler places several
dispatches at once, and the predicted-vs-actual join on the first real run showed
the large convolutions running 17-25% slower than their solo profile. The
sharded DroNet convs in artifacts/k1_run/baselines/trace_B4.csv are worse still —
62-68% over their solo 4-hart profile — because a solo multi-core profile is not
a valid cost for a shard that has company. That gap is either contention or a
profiling error, and the only way to tell is to create the contention
deliberately.

Design: pin the dispatch under test to one core and run N co-runners on chosen
other cores, then compare against the same dispatch with nothing else running.
The co-runners are real benchmark binaries, so the interference is
representative of the workload rather than a synthetic memory hog. A co-runner
may come from a *different build* (`--co-remote-dir`), which is how "IME kernel
alongside an RVV kernel" gets measured.

Placements worth separating on this part:
  solo            nothing else running
  same cluster    co-runner shares the 512K L2 (cores 0-3, or 4-7)
  other cluster   co-runner has its own L2

WHAT THE FIRST RUN ACTUALLY FOUND (and it is not what this docstring used to
predict): same-cluster co-running costs 1.043x, cross-cluster costs 1.185x. The
shared L2 is not the dominant mechanism; the cross-cluster path (interconnect /
DRAM, and likely the loss of any shared-L2 constructive effects) is worse. So
"spread work across clusters" is the WRONG default on this board. Do not
re-derive the intuition from first principles — re-run the measurement.

Output is a keyed artifact (`xpurt.contention/v2`): every measurement records
which placement it was, which cores the co-runners sat on, how many of them
there were, and which build they came from. It is deliberately a SEPARATE file
from the solo profile — a contention multiplier must never be silently folded
into a solo service time, or the next re-profile double-counts it. The read side
is `xpu-rt/contention_model.py`.

Examples
--------
Default sweep (solo / same cluster / other cluster, one RVV co-runner)::

    k1_contention.py

IME co-runner inside cluster 0, RVV dispatch under test on hart 0::

    k1_contention.py --co-cpus 1 --co-remote-dir /root/mb_k1/bench/dronet_IME \\
        --placement same_cluster_IME_x1 --append

IME under test, co-runner work on the other cluster::

    k1_contention.py --remote-dir /root/mb_k1/bench/dronet_IME \\
        --co-cpus 4 --co-remote-dir /root/mb_k1/bench/dronet_RVV \\
        --placement other_cluster_RVV_vs_IME --append

Three independent RVV kernels sharing cluster 0 with the dispatch under test::

    k1_contention.py --co-cpus 1,2,3 --placement same_cluster_RVV_x3 --append
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

_ROW = re.compile(r"real_time_mean\s+([0-9.]+)\s*(ns|us|ms|s)")
_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}

SCHEMA = "xpurt.contention/v2"


def _esc(m: str) -> str:
    return m.replace("$", r"\$")


def bench(host, rdir, tool, module, cpu, reps, co_cpus=None, co_modules=None,
          co_dir=None, timeout=900):
    """Run one benchmark on `cpu`, optionally with co-runners.

    `co_cpus` is a list: one independent co-runner process is started per cpu,
    taking modules from `co_modules` (cycled) out of `co_dir`.
    """
    run = (f'{tool} --module="{_esc(module)}" --device=local-task '
           f'--task_topology_cpu_ids={cpu} --benchmark_repetitions={reps}')
    co_cpus = list(co_cpus or [])
    if not co_cpus:
        cmd = f"cd {rdir} && {run}"
    else:
        # Start every co-runner first, give them a moment to reach steady
        # state, measure, then stop them all. Their output is discarded.
        co_dir = co_dir or rdir
        co_modules = list(co_modules or [])
        starts = ["rm -f /tmp/co.pids"]
        for i, c in enumerate(co_cpus):
            mod = co_modules[i % len(co_modules)] if co_modules else module
            co = (f'{tool} --module="{co_dir}/{_esc(mod)}" '
                  f'--device=local-task --task_topology_cpu_ids={c} '
                  f'--benchmark_repetitions={reps * 40}')
            starts.append(f"({co} >/dev/null 2>&1 & echo $! >> /tmp/co.pids)")
        cmd = (f"cd {rdir} && " + " && ".join(starts) +
               " && sleep 2 && " + run +
               " ; kill $(cat /tmp/co.pids) 2>/dev/null; true")
    p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    m = _ROW.search(p.stdout)
    if not m:
        return None
    return float(m.group(1)) * _TO_MS[m.group(2)]


def _cluster(cpu: int, cores_per_cluster: int) -> int:
    return int(cpu) // int(cores_per_cluster)


def _auto_placement(cpu, co_cpus, cores_per_cluster, co_dir, n) -> str:
    """Descriptive default key: `<same|other|mixed>_cluster_<BUILD>_x<N>`."""
    mine = _cluster(cpu, cores_per_cluster)
    theirs = {_cluster(c, cores_per_cluster) for c in co_cpus}
    if theirs == {mine}:
        base = "same_cluster"
    elif mine not in theirs:
        base = "other_cluster"
    else:
        base = "mixed_cluster"
    build = os.path.basename(str(co_dir).rstrip("/")) or "co"
    return f"{base}_{build}_x{n}"


def _build_tag(remote_dir: str) -> str:
    """`/root/mb_k1/bench/dronet_IME` -> `IME`."""
    leaf = os.path.basename(str(remote_dir).rstrip("/"))
    return leaf.split("_", 1)[1] if "_" in leaf else leaf


def _list_modules(host, rdir, limit=6):
    p = subprocess.run(
        ["ssh", host, f"ls {rdir}/*dispatch_*_benchmark.vmfb"],
        capture_output=True, text=True, timeout=120)
    mods = [l.split("/")[-1] for l in p.stdout.split() if l.strip()]
    return mods[:limit]


def _short(m: str) -> str:
    return m.replace("_embedded_elf_riscv_64_benchmark.vmfb", "")


def _upgrade_v1(prev: dict) -> dict:
    """Turn the flat v1 artifact into v2 `measurements` so `--append` never
    drops the original same/other-cluster evidence on the floor."""
    out = {}
    for placement, ms_key, ratio_key, cpu_key, med_key in (
        ("same_cluster", "same_cluster_ms", "same_ratio", "same_cluster_cpu",
         "median_same_cluster_ratio"),
        ("other_cluster", "other_cluster_ms", "other_ratio", "other_cluster_cpu",
         "median_other_cluster_ratio"),
    ):
        per_module = {}
        for r in prev.get("results") or []:
            if r.get(ratio_key) is None:
                continue
            per_module[r["module"]] = {"solo_ms": r.get("solo_ms"),
                                       "co_ms": r.get(ms_key),
                                       "ratio": r.get(ratio_key)}
        if not per_module:
            continue
        co_cpu = prev.get(cpu_key)
        out[placement] = {
            "placement": placement,
            "cpu_under_test": prev.get("cpu"),
            "co_cpus": [co_cpu] if co_cpu is not None else [],
            "n_co_runners": 1 if co_cpu is not None else 0,
            "co_runner": {"remote_dir": None, "build": None},
            "per_module": per_module,
            "median_ratio": prev.get(med_key),
            "upgraded_from": "xpurt.contention/v1",
        }
    return out


def _measure(a, mods, co_cpus, co_dir, co_modules_arg, solo_cache):
    """Measure every module in `mods` against one placement. Returns the v2
    measurement dict."""
    per_module = {}
    tag = _build_tag(a.remote_dir)
    for m in mods:
        # Solo baselines are per (build, module): the same dispatch name means
        # a different kernel in the IME build than in the RVV one.
        solo_key = f"{tag}:{m}"
        solo = solo_cache.get(solo_key)
        if solo is None:
            solo = bench(a.host, a.remote_dir, a.tool, m, a.cpu, a.reps)
            if solo is None or solo <= 0:
                continue
            solo_cache[solo_key] = solo
        # A co-runner executing the SAME module shares its weights, so an
        # L2-sharing co-placement can look *helpful*. Default to a different
        # module so the interference is competition, not constructive sharing
        # -- which is what concurrent models actually do.
        if co_modules_arg:
            co_mods = co_modules_arg
        else:
            co_mods = [next((x for x in mods if x != m), m)]
        co = bench(a.host, a.remote_dir, a.tool, m, a.cpu, a.reps,
                   co_cpus=co_cpus, co_modules=co_mods, co_dir=co_dir)
        if co is None or co <= 0:
            continue
        per_module[m] = {"solo_ms": solo, "co_ms": co, "ratio": co / solo,
                         "co_modules": co_mods}
        print(f"  {_short(m):<44}{solo:>10.3f}{co:>10.3f}{co/solo:>10.3f}x",
              flush=True)
    if not per_module:
        return None
    ratios = [v["ratio"] for v in per_module.values()]
    return {
        "cpu_under_test": a.cpu,
        "remote_dir": a.remote_dir,
        "build": _build_tag(a.remote_dir),
        "co_cpus": [int(c) for c in co_cpus],
        "n_co_runners": len(co_cpus),
        "co_runner": {
            "remote_dir": co_dir,
            "build": _build_tag(co_dir),
            "modules": co_modules_arg or "auto (a different dispatch)",
        },
        "reps": a.reps,
        "per_module": per_module,
        "median_ratio": statistics.median(ratios),
        "mean_ratio": statistics.fmean(ratios),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--host", default="k1")
    ap.add_argument("--remote-dir", default="/root/mb_k1/bench/dronet_RVV",
                    help="build the dispatch UNDER TEST comes from")
    ap.add_argument("--tool", default="/root/mb_k1/tools/iree-benchmark-module")
    ap.add_argument("--modules", default=None,
                    help="comma-separated benchmark vmfb names; default: pick "
                         "the largest few automatically")
    ap.add_argument("--cpu", type=int, default=0,
                    help="core the dispatch under test is pinned to")
    ap.add_argument("--co-cpus", default=None,
                    help="comma-separated list of cores to place co-runners "
                         "on; one independent co-runner per core. When "
                         "omitted, runs the default same/other-cluster sweep.")
    ap.add_argument("--co-remote-dir", default=None,
                    help="build the CO-RUNNERS come from (default: same as "
                         "--remote-dir). e.g. /root/mb_k1/bench/dronet_IME")
    ap.add_argument("--co-modules", default=None,
                    help="comma-separated modules for the co-runners, cycled "
                         "across --co-cpus. Default: a different dispatch "
                         "from the one under test.")
    ap.add_argument("--placement", default=None,
                    help="key this measurement is stored under. Default is "
                         "derived from the cluster geometry and co build.")
    ap.add_argument("--cores-per-cluster", type=int, default=4,
                    help="K1: 8 harts in 2 clusters of 4, each with its own "
                         "512K L2")
    ap.add_argument("--same-cluster-cpu", type=int, default=1,
                    help="default-sweep only")
    ap.add_argument("--other-cluster-cpu", type=int, default=4,
                    help="default-sweep only")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing artifact instead of "
                         "replacing it (keeps earlier placements)")
    ap.add_argument("--out", default="artifacts/k1_run/contention.json")
    a = ap.parse_args()

    if a.modules:
        mods = [m for m in a.modules.split(",") if m]
    else:
        mods = _list_modules(a.host, a.remote_dir)
    if not mods:
        print("no benchmark modules found", file=sys.stderr)
        return 1

    co_dir = a.co_remote_dir or a.remote_dir
    co_modules_arg = ([m for m in a.co_modules.split(",") if m]
                      if a.co_modules else None)

    # Which placements to run.
    if a.co_cpus:
        co_cpu_sets = [[int(c) for c in a.co_cpus.split(",") if c.strip()]]
        keys = [a.placement or _auto_placement(
            a.cpu, co_cpu_sets[0], a.cores_per_cluster, co_dir,
            len(co_cpu_sets[0]))]
    else:
        co_cpu_sets = [[a.same_cluster_cpu], [a.other_cluster_cpu]]
        keys = ["same_cluster", "other_cluster"]

    solo_cache: dict[str, float] = {}
    measurements: dict[str, dict] = {}
    for key, co_cpus in zip(keys, co_cpu_sets):
        print(f"\n== {key}  (under test: cpu {a.cpu} / "
              f"{_build_tag(a.remote_dir)};  co-runners: cpus {co_cpus} / "
              f"{_build_tag(co_dir)} x{len(co_cpus)})")
        print(f"  {'module':<44}{'solo ms':>10}{'co ms':>10}{'ratio':>11}")
        meas = _measure(a, mods, co_cpus, co_dir, co_modules_arg, solo_cache)
        if meas is None:
            print("  (no usable measurements)")
            continue
        meas["placement"] = key
        measurements[key] = meas
        print(f"  -> median slowdown {meas['median_ratio']:.3f}x "
              f"over {len(meas['per_module'])} dispatches")

    if not measurements:
        print("nothing measured", file=sys.stderr)
        return 1

    out = {
        "schema": SCHEMA,
        "host": a.host,
        "cores_per_cluster": a.cores_per_cluster,
        "cpu_under_test": a.cpu,
        # Recorded here only as the denominator of the ratios. This artifact is
        # NOT a profile: never merge these into one.
        "solo_ms": dict(solo_cache),
        "measurements": {},
    }
    if a.append and os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("measurements"):
            out["measurements"].update(prev["measurements"])
        if prev.get("solo_ms"):
            merged = dict(prev["solo_ms"])
            merged.update(out["solo_ms"])
            out["solo_ms"] = merged
        elif prev.get("results"):
            # v1 predecessor: upgrade rather than discard. That run is the one
            # that found the cross-cluster inversion.
            out["measurements"].update(_upgrade_v1(prev))
            out["upgraded_from"] = prev.get("schema", "xpurt.contention/v1")
    out["measurements"].update(measurements)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print("\nplacements in artifact:")
    for k, v in sorted(out["measurements"].items()):
        co = v.get("co_runner", {})
        print(f"  {k:<34} {v.get('median_ratio', float('nan')):.3f}x  "
              f"co={co.get('build')} x{v.get('n_co_runners')} "
              f"on cpus {v.get('co_cpus')}")
    if {"same_cluster", "other_cluster"} <= set(out["measurements"]):
        s = out["measurements"]["same_cluster"]["median_ratio"]
        o = out["measurements"]["other_cluster"]["median_ratio"]
        print(f"\n  same/other = {s / o:.3f}x "
              f"({'cross-cluster is WORSE' if o > s else 'same-cluster is worse'})")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
