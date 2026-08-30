#!/usr/bin/env python3
"""Re-measure every cost cell on the board in TWO phases, and report the gap.

The sweep (sweeps/qrb5165_20260829-200620) found that cost cells predicted
in-situ tile duration at 0.999x for tiles >= 1 ms but 1.655x for tiles < 1 ms,
missing by a roughly FIXED +0.234 ms. The cause is a methodology mismatch: the
cells were captured as the mean of 50 back-to-back executes (fully warm, per-
invocation setup amortised away), while the runtime calls each tile ONCE per
period from a lane thread that was asleep until its gate fired.

profile_segments.cpp now measures both:
    loop_*  back-to-back, the old methodology
    gap_*   each execute preceded by an idle gap, matching in-situ invocation

This drives it over every (binding, backend) pair named by the binding
manifests, using the context binaries already staged on the board.

    python3 remeasure_cells.py [--iters 40] [--gap-us 3000] [--out FILE]
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")
CTX_DIR = "/root/qnn_runtime_ctx"
BOARD_DIR = "/root/flowc_remeasure"
SDK = "/root/qairt"
LIB = {"hta": "libQnnHta.so", "dsp": "libQnnDsp.so", "cpu": "libQnnCpu.so"}
# CPU cells are measured UNMASKED. The lane's exec mask binds only the lane
# thread; the QNN CPU op package builds its thread pool at bringup with
# full-machine affinity, so a `taskset -c 4-5` measurement does not describe
# how the runtime actually executes the tile (see the
# `cpu_cells_must_be_unmasked` note in measurements/qrb5165_v66.json --  the
# `conditions` block there still claimed the mask and was stale).
MASK = {}


def board(script, timeout=900):
    q = script.replace("'", "'\\''")
    return subprocess.run(
        ["timeout", "-s", "KILL", str(timeout + 60), "ssh", "-o", "ConnectTimeout=20",
         "-o", "BatchMode=yes", BOARD, f"flock -w 900 /tmp/qnn_board.lock -c '{q}'"],
        capture_output=True, text=True, timeout=timeout + 120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--gap-us", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "measurements", "qrb5165_v66_remeasured.json"))
    ap.add_argument("--only", default=None, help="comma list of network names")
    a = ap.parse_args()

    # 1. collect every (cell, backend, ctx, graph) the manifests name
    want = set(a.only.split(",")) if a.only else None
    todo = []
    for fn in sorted(os.listdir(os.path.join(HERE, "bindings"))):
        if not fn.endswith(".json"):
            continue
        man = json.load(open(os.path.join(HERE, "bindings", fn)))
        net = man["network"]
        if want and net not in want:
            continue
        for b in man["bindings"]:
            for kind, spec in (b.get("backends") or {}).items():
                if kind not in LIB:
                    continue
                todo.append({"cell": f"{net}/{b['name']}", "backend": kind,
                             "ctx": spec["ctx"], "graph": spec.get("graph"),
                             "manifest": fn})
    print(f"{len(todo)} (cell, backend) pairs from {len(set(t['manifest'] for t in todo))} manifests")

    # 2. push + build the harness
    src = os.path.join(REPO, "qnn_models", "runtime", "profile_segments.cpp")
    subprocess.run(["ssh", BOARD, f"mkdir -p {BOARD_DIR}"], check=True)
    subprocess.run(["scp", "-q", src, f"{BOARD}:{BOARD_DIR}/"], check=True)
    r = board(f"cd {BOARD_DIR} && g++ -std=c++2a -O2 -pthread -I{SDK}/include "
              f"-I{SDK}/include/QNN profile_segments.cpp -o profile_seg -ldl "
              f"&& echo BUILT $(stat -c%s profile_seg)")
    print(r.stdout.strip() or r.stderr.strip()[-400:])
    if "BUILT" not in r.stdout:
        sys.exit("harness build failed")

    # 3. measure, one board round trip per pair (keeps the lock short)
    # ADSP_LIBRARY_PATH is semicolon-separated, so it MUST be quoted -- unquoted
    # the shell reads each ';' as a command separator and the run silently
    # becomes a series of no-ops.
    env = (f'LD_LIBRARY_PATH={SDK}/lib/target '
           f'ADSP_LIBRARY_PATH="{SDK}/lib/hexagon-v66/unsigned;{SDK}/lib/hexagon-v66;'
           f'/dsp/cdsp;/dsp" ')
    results = []
    for i, t in enumerate(todo, 1):
        ctx = f"{CTX_DIR}/{t['ctx']}"
        cmd = (f"cd {BOARD_DIR} && test -f {ctx} && {env}{MASK.get(t['backend'],'')}"
               f"./profile_seg {ctx} {LIB[t['backend']]} {a.iters} --gap-us {a.gap_us} "
               f"|| echo '{{\"status\":\"missing_or_failed\"}}'")
        r = board(cmd, timeout=600)
        line = ""
        for ln in (r.stdout or "").splitlines():
            if ln.strip().startswith("{"):
                line = ln.strip()
        try:
            js = json.loads(line) if line else {"status": "no_output"}
        except json.JSONDecodeError:
            js = {"status": "unparsable", "raw": line[:200]}
        rec = dict(t, **js)
        results.append(rec)
        if js.get("status") == "ok":
            print(f"  [{i:2}/{len(todo)}] {t['cell']+'@'+t['backend']:<38} "
                  f"loop {js['median_us']/1000:8.3f}  gap {js['gap_median_us']/1000:8.3f}  "
                  f"delta {js['gap_minus_loop_us']/1000:+7.3f} ms")
        else:
            print(f"  [{i:2}/{len(todo)}] {t['cell']+'@'+t['backend']:<38} {js.get('status')}"
                  f"  {str(r.stderr)[-120:] if not line else ''}")

    out = {"_comment": ("Two-phase re-measurement. loop_* reproduces the original "
                        "methodology (back-to-back executes); gap_* inserts an idle "
                        "gap before each execute so the measurement matches how the "
                        "scheduled runtime invokes a tile. Build cost cells from "
                        "gap_median_us."),
           "target": "qrb5165_v66", "iters": a.iters, "gap_us": a.gap_us,
           "harness": "qnn_models/runtime/profile_segments.cpp",
           "conditions": {"governor": "performance on all 8 cores",
                          "cpu_affinity": "taskset -c 4-5 for cpu cells"},
           "results": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{ok}/{len(results)} measured -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
