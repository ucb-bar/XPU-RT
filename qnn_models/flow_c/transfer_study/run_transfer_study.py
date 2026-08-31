#!/usr/bin/env python3
"""Measure boundary-handoff (transfer) cost directly, per entry.

Why this needs measuring rather than reasoning about:

  * The MILP models transfers as FREE. scripts/run_xpurt_schedule.py:244 builds
    `transfer_times = np.zeros((n_cores, n_cores))`, so no cross-lane edge ever
    contributes to a predicted start time.
  * The runtime pays a real cost. Every tile's outputs are memcpy'd into a
    global mutex-guarded cache after execute, and every tile's inputs are
    searched for in that cache before execute. Both loops sit OUTSIDE the
    trace's actual_start..actual_end window, so the per-tile cost model cannot
    see them either.

emit_runtime.py now times both loops per entry and records bytes moved. This
re-emits runtimes from the sweep's ALREADY-SOLVED schedules -- no re-solve, so
placements are identical to the recorded sweep and the only change is the
instrumentation -- then runs them and collects the traces.

    python3 transfer_study/run_transfer_study.py [--reps 3] [--only a,b]
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWC = os.path.abspath(os.path.join(HERE, ".."))
S = os.path.join(FLOWC, "sweeps", "qrb5165_20260829-200620")
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")

# Points chosen to cover every multi-tile network and both lane-pair regimes.
# dronet and mlp_control are single-tile (no internal boundary); yolov8n has
# backbone->head, fused_split has two conv branches -> tail, vint has
# encoders -> decoder.
POINTS = [
    ("main", "baseline_seed0"),   # yolov8n x4, dronet x2
    ("main", "baseline_seed3"),   # yolov8n x2, smallest
    ("main", "fused_seed0"),      # fused_split + yolov8n
    ("main", "fused_seed1"),      # fused_split x5 + yolov8n x5
    ("main", "fused_vint_seed0"), # + vint encoders->decoder
    ("main", "fused_vint_seed1"),
    ("add",  "fused_seed7"),      # periodic-only, uncontended lanes
]
ROOT = {"main": S, "add": os.path.join(S, "addendum_periodic_only")}


def sh(cmd, cwd=None, log=None, timeout=1800):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            f.write("+ " + " ".join(cmd) + f"\n(cwd={cwd})\n\n"
                    + (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else None

    todo = [(k, pt) for k, pt in POINTS if not want or pt in want]
    state = {}
    sp = os.path.join(HERE, "state.json")
    if os.path.exists(sp):
        state = json.load(open(sp))

    for kind, pt in todo:
        root = ROOT[kind]
        spec = os.path.join(root, "workloads", f"{pt}.flowc.json")
        sched = None
        for cand in (f"scheduled_{pt}_profiled.json",
                     f"scheduled_{pt}_greedy_periodic_profiled.json"):
            c = os.path.join(root, "schedules", cand)
            if os.path.exists(c):
                sched = c
                break
        if not sched or not os.path.exists(spec):
            print(f"  {pt:<20} SKIP (spec or schedule missing)")
            continue
        out_dir = os.path.join(HERE, "runtimes", pt)
        p = sh([sys.executable, "flow_c.py", "runtime", "--workload", spec,
                "--tag", f"xfer_{pt}", "--lane-mode", "kind-network",
                "--schedule", sched, "--out-dir", out_dir],
               cwd=FLOWC, log=os.path.join(HERE, "logs", f"runtime_{pt}.log"))
        if p.returncode != 0:
            print(f"  {pt:<20} RUNTIME EMIT FAILED")
            continue
        rec = state.setdefault(pt, {"kind": kind, "reps": {}})
        for rep in range(1, a.reps + 1):
            key = f"rep{rep}"
            if rec["reps"].get(key, {}).get("ok"):
                continue
            log_dir = os.path.join(HERE, "runs", pt, key)
            os.makedirs(log_dir, exist_ok=True)
            t0 = time.time()
            r = sh([sys.executable, "flow_c.py", "run", "--workload", spec,
                    "--tag", f"xfer_{pt}", "--tuned", "--out-dir", out_dir,
                    "--log-dir", log_dir, "--board", BOARD,
                    "--board-dir", "/root/flowc_xfer"],
                   cwd=FLOWC, log=os.path.join(log_dir, "driver.log"))
            rl = os.path.join(log_dir, "run.log")
            ok, hand, wall = False, None, None
            if os.path.exists(rl):
                for ln in open(rl):
                    if "[summary]" in ln:
                        ok = True
                        for tok in ln.split():
                            if tok.startswith("wall="):
                                wall = float(tok[5:].rstrip("ms"))
                        if "handoffs=" in ln:
                            hand = int(ln.split("handoffs=")[1].split()[0].strip(", "))
            rec["reps"][key] = {"ok": ok, "wall_ms": wall, "handoffs": hand,
                                "rc": r.returncode, "s": round(time.time() - t0, 1)}
            print(f"  {pt:<20} {key}  wall {wall}  handoffs {hand}  ok={ok}")
            json.dump(state, open(sp, "w"), indent=1)
    print(f"\nstate -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
