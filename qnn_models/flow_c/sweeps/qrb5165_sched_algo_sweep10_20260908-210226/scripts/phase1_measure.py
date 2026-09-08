#!/usr/bin/env python3
"""Phase 1c — measure a real cell on the board for every (tile, backend) this
sweep's binding manifests declare, using the GAP methodology.

Why gap and not back-to-back: `qnn_models/flow_c/remeasure_cells.py` documents
that back-to-back cells predicted in-situ tile duration at 0.999x for tiles
>= 1 ms but 1.655x for tiles < 1 ms, missing by a roughly FIXED +0.234 ms,
because the runtime calls each tile ONCE per period from a lane thread that was
asleep until its gate fired. These cells feed a scheduler whose tiles are
invoked exactly that way, so `gap_median_us` is the right statistic and
`loop_median_us` is kept only for comparison.

This is `remeasure_cells.py` with three changes and no others:
  * it reads THIS sweep's bindings/ rather than the shared flow_c/bindings/,
    so it cannot perturb or be perturbed by the shipped manifests;
  * `gpu` is in the backend table (the shipped tool has hta/dsp/cpu only),
    because the gpu lane is part of this sweep's config axis;
  * it also measures the `cpu@int8` context, which no manifest declares, so
    the fp32-on-CPU choice in phase1_bindings.py is auditable.

CPU cells are measured UNMASKED, for the reason the shipped tool gives: the
lane's exec mask binds only the lane thread, while the QNN CPU op package
builds its thread pool at bringup with full-machine affinity, so a
`taskset -c 4-5` measurement does not describe how the runtime executes the
tile.

Three independent PASSES over every pair, not one. The first single-pass run
produced `dronet_sc/dronet_sc_full@dsp` at 2.666 ms against 0.673 and 0.646 ms
for the rungs either side of it, with a gap/loop ratio of 4.9x where every
other cell sat at 1.0-1.5x. That is the bimodal power-collapse the opsweep
documents for the accelerators ("they power-collapse mid-loop even at zero gap,
so repeat passes scatter 4x"), and a single pass cannot tell it from a real
cost. Rather than hand-pick which cells to redo, every pair is measured three
times and `build_cost_model.py` takes the per-cell MEDIAN across passes and
records the spread, so the repeatability of each cell is part of the record.

    python3 phase1_measure.py [--iters 40] [--gap-us 3000] [--passes 3]
                              [--only net,net] [--resume]
"""
from __future__ import annotations

import argparse, json, os, statistics as st, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
BOARD = os.environ.get("QNN_BOARD_HOST", "root@qrb5165")
CTX_DIR = "/root/qnn_runtime_ctx"
BOARD_DIR = "/root/flowc_s10_measure"
SDK = "/root/qairt"
LIB = {"hta": "libQnnHta.so", "dsp": "libQnnDsp.so", "cpu": "libQnnCpu.so",
       "gpu": "libQnnGpu.so", "cpu@int8": "libQnnCpu.so"}


def board(script, timeout=900):
    """One board interaction behind the shared lock. `ssh -n` matters: without
    it ssh eats stdin inside the `flock -c` chain and the result is silently
    empty."""
    q = script.replace("'", "'\\''")
    t0 = time.time()
    p = subprocess.run(
        ["timeout", "-s", "KILL", str(timeout + 60), "ssh", "-n",
         "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
         f"flock -w 900 /tmp/qnn_board.lock -c '{q}'"],
        capture_output=True, text=True, timeout=timeout + 120)
    return p, round(time.time() - t0, 2)


def lock_wait_s(timeout=900):
    t0 = time.time()
    r = subprocess.run(["timeout", "-s", "KILL", str(timeout + 30), "ssh", "-n",
                        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", BOARD,
                        f"flock -w {timeout} /tmp/qnn_board.lock -c 'true'"],
                       capture_output=True, text=True)
    return round(time.time() - t0, 2), r.returncode


def collect(only=None):
    todo = []
    bdir = os.path.join(SWEEP, "bindings")
    compose = {}
    # phase1_state.json is written after EVERY model, so the cpu@int8 audit
    # contexts are visible while a build is still in flight; phase1_compose.json
    # is the flat record written only when the build finishes.
    for cpath in (os.path.join(SWEEP, "results", "phase1_compose.json"),
                  os.path.join(SWEEP, "results", "phase1_state.json")):
        if not os.path.exists(cpath):
            continue
        d = json.load(open(cpath))
        for r in (d.values() if isinstance(d, dict) else d):
            if isinstance(r, dict) and r.get("compose"):
                compose.setdefault(r["id"], r)
    for fn in sorted(os.listdir(bdir)):
        if not fn.endswith(".json"):
            continue
        man = json.load(open(os.path.join(bdir, fn)))
        net = man["network"]
        if only and net not in only:
            continue
        for b in man["bindings"]:
            for kind, spec in (b.get("backends") or {}).items():
                todo.append(dict(cell=f'{net}/{b["name"]}', backend=kind,
                                 ctx=spec["ctx"], graph=spec.get("graph"),
                                 precision=spec.get("precision"),
                                 declared=True, manifest=fn))
            # the undeclared int8-CPU context, for audit
            v = ((compose.get(net) or {}).get("compose") or {}).get("cpu@int8") or {}
            if v.get("status") == "ok":
                todo.append(dict(cell=f'{net}/{b["name"]}', backend="cpu@int8",
                                 ctx=v["ctx"],
                                 graph=(compose[net].get("dlc_graph_name")),
                                 precision="int8", declared=False,
                                 manifest=fn))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--gap-us", type=int, default=3000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=os.path.join(SWEEP, "measurements",
                                                 "qrb5165_v66_s10port_raw.json"))
    ap.add_argument("--passes", type=int, default=3,
                    help="independent repeats of the whole sweep; the cost "
                         "model takes the per-cell median across them")
    ap.add_argument("--resume", action="store_true",
                    help="keep results already in --out and only measure the rest")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    base = collect(only)
    todo = [dict(t, pass_i=i) for i in range(1, a.passes + 1) for t in base]
    print(f"{len(base)} (cell, backend) pairs "
          f"({sum(1 for t in base if t['declared'])} declared, "
          f"{sum(1 for t in base if not t['declared'])} audit-only)"
          f" x {a.passes} pass(es) = {len(todo)} measurements")

    have = {}
    if a.resume and os.path.exists(a.out):
        for r in json.load(open(a.out)).get("results", []):
            if r.get("status") == "ok":
                have[(r["cell"], r["backend"], r.get("pass_i", 1))] = r
        print(f"  resuming: {len(have)} already measured")

    # governor: save, force performance, restore in the finally below
    g, _ = board("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    prev_gov = (g.stdout or "").strip().splitlines()[-1] if g.stdout else "?"
    print(f"  governor was: {prev_gov}")
    board("for c in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; "
          "do echo performance > $c; done; "
          "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

    results = []
    try:
        src = os.path.join(REPO, "qnn_models", "runtime", "profile_segments.cpp")
        subprocess.run(["ssh", "-n", BOARD, f"mkdir -p {BOARD_DIR}"], check=True)
        subprocess.run(["scp", "-q", src, f"{BOARD}:{BOARD_DIR}/"], check=True)
        r, _ = board(f"cd {BOARD_DIR} && g++ -std=c++2a -O2 -pthread "
                     f"-I{SDK}/include -I{SDK}/include/QNN profile_segments.cpp "
                     f"-o profile_seg -ldl && echo BUILT $(stat -c%s profile_seg)")
        print("  " + ((r.stdout or "").strip() or (r.stderr or "").strip()[-300:]))
        if "BUILT" not in (r.stdout or ""):
            sys.exit("harness build failed")

        env = (f'LD_LIBRARY_PATH={SDK}/lib/target '
               f'ADSP_LIBRARY_PATH="{SDK}/lib/hexagon-v66/unsigned;'
               f'{SDK}/lib/hexagon-v66;/dsp/cdsp;/dsp" ')
        for i, t in enumerate(todo, 1):
            key = (t["cell"], t["backend"], t["pass_i"])
            if key in have:
                results.append(have[key])
                continue
            ctx = f'{CTX_DIR}/{t["ctx"]}'
            cmd = (f"cd {BOARD_DIR} && test -f {ctx} && {env}./profile_seg {ctx} "
                   f'{LIB[t["backend"]]} {a.iters} --gap-us {a.gap_us} '
                   f"|| echo '{{\"status\":\"missing_or_failed\"}}'")
            wait, _ = lock_wait_s()
            p, dt = board(cmd, timeout=900)
            line = ""
            for ln in (p.stdout or "").splitlines():
                if ln.strip().startswith("{"):
                    line = ln.strip()
            try:
                js = json.loads(line) if line else {"status": "no_output"}
            except json.JSONDecodeError:
                js = {"status": "unparsable", "raw": line[:300]}
            # profile_seg's JSON has its own `backend` key (the .so it was
            # handed), which silently overwrote the registry KIND this record
            # is about -- every cell came back keyed "libQnnDsp.so" and the
            # cost model had no lane names at all. Let the record win, keep
            # the harness value under a different name.
            rec = {**js, **t, "harness_backend": js.get("backend"),
                   "lock_wait_s": wait, "board_s": dt}
            results.append(rec)
            tag = f'{t["cell"]}@{t["backend"]} p{t["pass_i"]}'
            if js.get("status") == "ok":
                print(f"  [{i:3}/{len(todo)}] {tag:<42} loop "
                      f"{js['median_us']/1000:9.3f}  gap {js['gap_median_us']/1000:9.3f}"
                      f"  delta {js['gap_minus_loop_us']/1000:+8.3f} ms"
                      f"  lock {wait}s")
            else:
                print(f"  [{i:3}/{len(todo)}] {tag:<42} {js.get('status')}"
                      f"  {(p.stderr or '')[-100:] if not line else ''}")
            if i % 10 == 0:
                dump(a, results, prev_gov)
    finally:
        board(f"for c in /sys/devices/system/cpu/cpu[0-7]/cpufreq/scaling_governor; "
              f"do echo {prev_gov} > $c; done; "
              f"cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        dump(a, results, prev_gov)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{ok}/{len(results)} measured -> {a.out}")
    return 0


def dump(a, results, prev_gov):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({
        "_comment": ("Two-phase re-measurement for the sched_algo_sweep10 port. "
                     "loop_* is the back-to-back methodology; gap_* inserts an "
                     "idle gap before each execute so the measurement matches "
                     "how the scheduled runtime invokes a tile. Cost cells are "
                     "built from gap_median_us -- see scripts/build_cost_model.py."),
        "target": "qrb5165_v66", "iters": a.iters, "gap_us": a.gap_us,
        "passes": a.passes,
        "harness": "qnn_models/runtime/profile_segments.cpp",
        "conditions": {"governor": "performance on all 8 cores (forced by this "
                                   f"script; restored to {prev_gov!r} after)",
                       "cpu_affinity": "UNMASKED — the lane exec mask binds only "
                                       "the lane thread, not QnnCpu's own pool"},
        "results": results,
    }, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
