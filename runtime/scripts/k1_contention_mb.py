#!/usr/bin/env python3
"""Measure how much co-runners slow a dispatch down, on the ModelBlaster path.

WHY A SECOND CONTENTION HARNESS
-------------------------------
`k1_contention.py` measures the same thing and is the source of the numbers in
`artifacts/k1_run/contention.json` -- but it drives `iree-benchmark-module`
over `.vmfb` files, and the IREE path is retired. Every kernel that runs on
this board today comes out of ModelBlaster's curated tree, so a contention
multiplier measured against IREE-compiled kernels is a multiplier for code
nobody runs.

It is also a better measurement on this path, not merely an equivalent one.
IREE needed one process invocation per dispatch, so a 21-dispatch model was 21
separate runs with 21 separate cache states. The ModelBlaster harness runs the
whole model and prints a per-dispatch profile block, so ONE run under
contention gives every dispatch's co-run cost, measured in the cache state the
dispatches actually see each other in.

WHAT IS BEING COMPARED, EXACTLY
-------------------------------
The dispatch under test is pinned to one hart. The co-runners are real model
harnesses looping on other harts -- real kernels touching real weights, not a
synthetic memory hog, so the interference is representative of the workload
rather than of a stress test. Each placement is:

    solo            nothing else running
    same_cluster    co-runner shares the 512K L2 (harts 0-3, or 4-7)
    other_cluster   co-runner has its own L2

THE PRIOR RESULT, WHICH IS COUNTERINTUITIVE AND WORTH RE-TESTING RATHER THAN
INHERITING. On the IREE path the medians were same-cluster 1.043x and
cross-cluster 1.185x -- i.e. sharing an L2 costs LESS than crossing to the
other cluster, so "spread the work across clusters" is the wrong default on
this part. That is the opposite of the shared-L2 intuition, which is exactly
why it should be re-measured on the kernels we actually ship rather than
carried over. It is also consistent with what the multi-core sweep found
independently: DroNet is SLOWER on 8 harts than on 4 (5.32 vs 5.25 ms).

The artifact is deliberately a SEPARATE file from the solo profile. A
contention multiplier folded into a solo service time is double-counted by the
next re-profile, silently. `xpu-rt/contention_model.py` is the read side and
multiplies at duration-lookup time.

Keys are `<model>_dispatch_<id>`, which `contention_model.canonical_key`
already reduces to `<model>:<id>` -- the same key it derives from an IREE
module name, so the read side needs no change and the two artifacts are
comparable row for row.

    k1_contention_mb.py --model dronet
    k1_contention_mb.py --model dronet --co-model yolov8_nano_64x96 --append

Exit 0 on a complete sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, "artifacts", "k1_run", "contention_mb.json")

SCHEMA = "xpurt.contention/v2"

#: rdtime on the K1. Fixed by the hardware, not a calibration -- see
#: runtime/mb_posix_compat.h, which makes the same choice for the same reason.
CLOCK_MHZ = 24.0

#: Iterations the CO-RUNNER performs inside one process. Large enough that
#: process startup is a rounding error on its residency -- see
#: `start_co_runners` for the /proc/stat measurement that fixed this number's
#: predecessor (which was 1, implicitly, and wrong).
CO_RUNNER_ITERS = 100000

_ITER_PROFILE = re.compile(
    r"=== MODELBLASTER_ITER_PROFILE_BEGIN \[(\d+)\] ===\n(.*?)"
    r"=== MODELBLASTER_ITER_PROFILE_END \[\1\] ===", re.S)
_PROFILE = re.compile(
    r"=== MODELBLASTER_PROFILE_BEGIN ===\n(.*?)=== MODELBLASTER_PROFILE_END ===",
    re.S)


def parse_profile(out: str) -> Dict[str, float]:
    """`{dispatch_key: ms}`, median over warm iterations.

    Prefers the per-iteration blocks and DROPS ITERATION 0. A single cold
    sample is not a profile -- two consecutive 4-hart ffn_block runs differed
    by 43% -- and under contention the first iteration is worse still, because
    the co-runners have not reached steady state. Falls back to the final
    block when the harness was run with MODELBLASTER_ITERS unset.
    """
    per: Dict[str, List[float]] = {}
    blocks = _ITER_PROFILE.findall(out)
    if len(blocks) >= 2:
        for it, body in blocks:
            if int(it) == 0:
                continue          # warmup
            for key, ms in _rows(body).items():
                per.setdefault(key, []).append(ms)
        return {k: statistics.median(v) for k, v in per.items() if v}
    m = _PROFILE.search(out)
    return _rows(m.group(1)) if m else {}


def _rows(body: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in body.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 5 or parts[0] == "dispatch_id":
            continue
        try:
            did = int(parts[0])
            ticks = float(parts[-1])
        except ValueError:
            continue
        out[str(did)] = ticks / (CLOCK_MHZ * 1000.0)
    return out


def run_dut(host: str, remote_bin: str, cpu: int, iters: int,
            timeout: float) -> str:
    cmd = (f"MODELBLASTER_CPU={cpu} MODELBLASTER_ITERS={iters} "
           f"{remote_bin}")
    p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode != 0:
        print(f"dut exited {p.returncode}: {p.stderr[-400:]}", file=sys.stderr)
    return p.stdout


def start_co_runners(host: str, remote_bin: str, cpus: List[int]) -> str:
    """Launch a looping co-runner per hart. Returns a token for stopping them.

    A LOOP, because the dispatch under test takes tens of milliseconds and a
    co-runner that finishes early leaves the tail of the measurement
    uncontended -- a blend of two conditions reported as one.

    MANY ITERATIONS INSIDE ONE PROCESS, not one inference per process, and
    this is the difference between a measurement and a mirage. Measured on the
    board with per-CPU /proc/stat sampling (loadavg is useless here: this board
    has a permanent floor of exactly 2.00 from two D-state kernel threads):

        respawn per inference   cpu1 80%   and cpu3 8%, cpu4 4%, cpu5 6%
        ITERS inside one proc   cpu1 100%  and every other hart 0%

    DroNet is 8.3 ms. Re-exec'ing a 3.5 MB static binary around it means most
    of the co-runner's wall time is fork, exec, loader and page faults -- work
    the process does BEFORE `main` calls sched_setaffinity, so it is not
    pinned and lands wherever the scheduler puts it. The co-runner then
    contaminates harts it was never assigned to, in both the same-cluster and
    the other-cluster arm, by the same amount.

    That is not a subtle bias. The first run of this script used the respawn
    form and reported same_cluster 1.011x, other_cluster 1.007x -- a null
    result that looked like "the ModelBlaster kernels do not contend", when
    what it actually measured was two placements sharing one source of
    off-target noise.
    """
    tag = "mb_corun_%d" % os.getpid()
    for c in cpus:
        inner = (f"while :; do MODELBLASTER_CPU={c} "
                 f"MODELBLASTER_ITERS={CO_RUNNER_ITERS} {remote_bin} "
                 f">/dev/null 2>&1; done")
        subprocess.run(
            ["ssh", host,
             f"nohup setsid bash -c {json.dumps(inner)} "
             f"</dev/null >/dev/null 2>&1 & echo {tag}"],
            capture_output=True, text=True, timeout=60)
    return tag


def stop_co_runners(host: str, remote_bin: str) -> None:
    """Kill every co-runner, then VERIFY none survived.

    Not belt-and-braces: a co-runner that outlives its measurement silently
    contaminates the NEXT one, and the next one is usually the solo baseline --
    which would make contention look smaller than it is, in the direction that
    flatters the result.
    """
    base = os.path.basename(remote_bin)
    # The bracket trick: `[d]ronet...` matches the process but not the pgrep
    # command line that contains the pattern. Without it the check counts
    # ITSELF and reports a survivor on every clean run -- which it did, on the
    # first invocation, and the refusal that follows would have been permanent.
    unself = f"[{base[0]}]{base[1:]}"
    subprocess.run(
        ["ssh", host,
         f"pkill -f {base} ; pkill -f 'MODELBLASTER_CPU=' ; sleep 0.3 ; true"],
        capture_output=True, text=True, timeout=60)
    left = subprocess.run(["ssh", host, f"pgrep -cf '{unself}' || true"],
                          capture_output=True, text=True, timeout=60)
    n = (left.stdout or "0").strip()
    if n not in ("", "0"):
        raise SystemExit(
            f"{n} co-runner process(es) still alive after pkill. Refusing to "
            f"continue: a survivor contaminates the next measurement, and the "
            f"next one is the solo baseline.")


def cluster_of(cpu: int, per_cluster: int) -> int:
    return cpu // per_cluster


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("MODELBLASTER_K1_HOST", "k1"))
    ap.add_argument("--remote-root",
                    default=os.environ.get("MODELBLASTER_K1_REMOTE_ROOT",
                                           "/root/mb_k1"))
    ap.add_argument("--model", required=True,
                    help="model under test, e.g. dronet")
    ap.add_argument("--backend", default="rvv_x60")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--co-model", default=None,
                    help="co-runner model (default: the same one)")
    ap.add_argument("--co-backend", default=None)
    ap.add_argument("--cpu", type=int, default=0,
                    help="hart the dispatch under test is pinned to")
    ap.add_argument("--same-cluster-cpu", default="1",
                    help="comma list of harts in the DUT's cluster")
    ap.add_argument("--other-cluster-cpu", default="4",
                    help="comma list of harts in the other cluster")
    ap.add_argument("--suffix", default="",
                    help="appended to both placement names, so a second sweep "
                         "with more co-runners does not overwrite the first")
    ap.add_argument("--cores-per-cluster", type=int, default=4)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing artifact instead of replacing")
    a = ap.parse_args()

    def binary(model: str, backend: str) -> str:
        return (f"{a.remote_root}/bin/"
                f"{model}_{a.quant}_{backend}_harness")

    dut = binary(a.model, a.backend)
    co_model = a.co_model or a.model
    co_bin = binary(co_model, a.co_backend or a.backend)

    def cpulist(spec: str) -> List[int]:
        return [int(x) for x in str(spec).replace(" ", "").split(",") if x]

    same_cpus = cpulist(a.same_cluster_cpu)
    other_cpus = cpulist(a.other_cluster_cpu)
    dut_cluster = cluster_of(a.cpu, a.cores_per_cluster)
    for c in other_cpus:
        if cluster_of(c, a.cores_per_cluster) == dut_cluster:
            raise SystemExit(
                f"--other-cluster-cpu {c} is in the same cluster as --cpu "
                f"{a.cpu}; the 'other_cluster' row would measure the same "
                f"thing as 'same_cluster' under a different name.")
    for c in same_cpus:
        if cluster_of(c, a.cores_per_cluster) != dut_cluster:
            raise SystemExit(
                f"--same-cluster-cpu {c} is NOT in the DUT's cluster.")
        if c == a.cpu:
            raise SystemExit(
                f"--same-cluster-cpu {c} is the DUT's own hart; a co-runner "
                f"there does not contend, it time-slices, which is a "
                f"different phenomenon with a different multiplier.")

    print(f"dut  {dut}  hart {a.cpu}")
    print(f"co   {co_bin}")

    # Make sure nothing is left running from an interrupted earlier sweep.
    stop_co_runners(a.host, co_bin)

    def solo_now(label: str) -> Dict[str, float]:
        p = parse_profile(run_dut(a.host, dut, a.cpu, a.iters, a.timeout))
        if not p:
            raise SystemExit("no profile block from the solo run; is the "
                             "binary deployed? run scripts/run_model_k1.sh "
                             "once first.")
        print(f"  solo ({label}): {sum(p.values()):.3f} ms over "
              f"{len(p)} dispatches")
        return p

    # A PAIRED DESIGN: solo is re-measured immediately before every co-run
    # arm, not once at the start.
    #
    # WHY. Two solo runs of DroNet twenty minutes apart differed by 2.6%
    # (8.328 vs 8.543 ms) with nothing else on the board. The effects being
    # measured here are 1-17%, so at the small end an unpaired design is
    # reporting drift. Two arms measured against DIFFERENT drift also cannot
    # be compared to each other, which is the whole point of running
    # same_cluster beside other_cluster.
    #
    # It doubles the board time. That is the cheaper mistake.
    reference_solo = solo_now("reference")

    measurements: Dict[str, dict] = {}
    for placement, co_cpus in (("same_cluster" + a.suffix, same_cpus),
                               ("other_cluster" + a.suffix, other_cpus)):
        solo = solo_now(placement)
        start_co_runners(a.host, co_bin, co_cpus)
        try:
            co = parse_profile(run_dut(a.host, dut, a.cpu, a.iters, a.timeout))
        finally:
            stop_co_runners(a.host, co_bin)
        if not co:
            raise SystemExit(f"{placement}: no profile block")
        drift = (sum(solo.values()) / sum(reference_solo.values())
                 if reference_solo else 1.0)

        per_module = {}
        ratios = []
        for did, ms in sorted(co.items(), key=lambda kv: int(kv[0])):
            s = solo.get(did)
            if not s:
                continue
            r = ms / s
            per_module[f"{a.model}_dispatch_{did}"] = {
                "solo_ms": round(s, 6), "co_ms": round(ms, 6),
                "ratio": round(r, 6)}
            ratios.append(r)
        med = statistics.median(ratios) if ratios else 1.0
        # The COST-WEIGHTED ratio, alongside the median, because they answer
        # different questions and can disagree sharply. Measured here: one
        # co-runner on the other cluster gave a median of 1.000 and a
        # cost-weighted 1.037 -- the slowdown is concentrated in the few big
        # dispatches, so the median says "nothing happened" about a model that
        # got 3.7% slower. The median is what `contention_model.median_factor`
        # falls back to for an unknown dispatch; the total is what the model's
        # latency actually did.
        tot_solo = sum(v["solo_ms"] for v in per_module.values())
        tot_co = sum(v["co_ms"] for v in per_module.values())
        measurements[placement] = {
            "placement": placement,
            "cpu_under_test": a.cpu,
            "co_cpus": list(co_cpus),
            "n_co_runners": len(co_cpus),
            "co_model": co_model,
            "total_ratio": round(tot_co / tot_solo, 6) if tot_solo else 1.0,
            # How far this arm's own solo sat from the first solo of the
            # sweep. It is the drift the pairing removed, and it belongs in
            # the artifact: an arm whose drift is the size of its effect is
            # not evidence, and a reader cannot tell without this number.
            "solo_drift_vs_reference": round(drift, 6),
            "co_runner": {"remote_dir": os.path.dirname(co_bin),
                          "build": os.path.basename(co_bin)},
            "per_module": per_module,
            "median_ratio": round(med, 6),
        }
        print(f"{placement:<22} median {med:.3f}x  cost-weighted "
              f"{measurements[placement]['total_ratio']:.3f}x  "
              f"({len(ratios)} dispatches, {len(co_cpus)} co-runner(s) "
              f"of {co_model}; this arm's solo drifted {drift:.3f}x from the "
              f"sweep reference)")

    data = {"schema": SCHEMA, "host": a.host,
            "cores_per_cluster": a.cores_per_cluster,
            "cpu_under_test": a.cpu,
            "path": "modelblaster",
            "solo_ms": {f"{a.model}_dispatch_{k}": round(v, 6)
                        for k, v in reference_solo.items()},
            "measurements": measurements}

    if a.append and os.path.exists(a.out):
        prev = json.load(open(a.out))
        prev.setdefault("measurements", {}).update(measurements)
        prev.setdefault("solo_ms", {}).update(data["solo_ms"])
        data = prev
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(data, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
