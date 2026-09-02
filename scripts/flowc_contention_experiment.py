#!/usr/bin/env python3
"""Tune the QRB5165 cost model against a CONCURRENT multi-model workload.

WHY THIS RUN IS THE MISSING ONE. `flowc_residual_feedback.py` learns a
per-backend correction from a trace, and on the four traces committed today it
is worth 27% within a configuration. But every one of those 440 dispatches ran
SOLO -- co-runner count is zero throughout, because the smolVLA workloads are
serial chains. So what has been measured so far is calibration bias, not
contention, and the contention half of the model has never been tested against
anything. This script produces the trace that would test it.

THE LOOP. Each round is: schedule -> generate runtime -> stage to the board ->
run under the board lock -> capture the trace -> fit a correction from what
actually happened -> schedule the NEXT round with the corrected costs.

  round 0   uncontended solo profiles, as shipped
  round 1   + per-backend correction fitted on round 0
  round 2   + correction conditioned on CO-RUNNER COUNT, which only round 1's
            trace can supply, because you need a concurrent run to observe it

Success is not "the makespan got smaller" -- a correction changes the estimate,
not the silicon. It is: the PREDICTION ERROR falls, and the schedule chosen
under corrected costs beats the one chosen under solo costs when both are
measured on the board. Both are reported per round.

BOARD SHARING. Another agent runs sweeps on this board. Every board-touching
step goes through `flock` on the same path deploy_and_run.sh uses, so the two
serialise rather than corrupt each other, and `--wait-for-board` blocks until
the lock is free instead of failing. Nothing here runs on the board without
holding that lock.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

BOARD_LOCK = os.environ.get("BOARD_LOCK", "/tmp/qnn_board.lock")
GRAPH = "gen/qnn_vmfb/{m}/qrb5165_flowc/{b}/{m}.int8/{m}.int8_dispatch_graph.json"
PROFILE = ("gen/profile/{b}/qrb5165_flowc/{m}/{m}.int8/"
           "{m}_qrb5165_flowc_{b}_{m}.int8/topo_0/results.csv")

LANES3 = ({"cpu_p": 1, "cpu_e": 1, "cpu_x": 1},
          {"cpu_p": "HTA", "cpu_e": "DSP", "cpu_x": "CPU"})


# ----------------------------------------------------------------- board lock
def board_free() -> bool:
    """True when nothing else holds the shared board lock."""
    try:
        r = subprocess.run(["flock", "-n", BOARD_LOCK, "-c", "true"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def wait_for_board(poll_s: int, max_wait_s: int) -> bool:
    """Block until the board lock frees. Returns False on timeout.

    Polls rather than blocking inside flock so the wait is visible and
    interruptible -- a silent multi-hour block inside flock looks like a hang.
    """
    t0 = time.time()
    n = 0
    while time.time() - t0 < max_wait_s:
        if board_free():
            if n:
                print(f"  board free after {time.time()-t0:.0f}s")
            return True
        if n % 10 == 0:
            print(f"  board busy ({time.time()-t0:.0f}s waited) — "
                  f"another agent holds {BOARD_LOCK}", flush=True)
        n += 1
        time.sleep(poll_s)
    return False


# ------------------------------------------------------------------- workload
def coverage(models, lanes_kinds) -> list[str]:
    missing = []
    for m in models:
        for b in lanes_kinds:
            if not os.path.exists(os.path.join(REPO, PROFILE.format(m=m, b=b))):
                missing.append(f"{m}@{b}")
        if not os.path.exists(os.path.join(REPO, GRAPH.format(m=m, b="CPU"))):
            missing.append(f"{m}:graph")
    return missing


def build_spec(nets: dict, out: str, gen_root: str = "gen") -> str:
    """A workload built to actually OVERLAP.

    The earlier sweeps mostly produced serial schedules because one heavy chain
    dominated. Here the periodic contenders are given periods short enough that
    they must be in flight while the heavy model runs -- that is the whole
    point, since a schedule with no overlap teaches a contention model nothing.
    """
    machines, profile_hw = LANES3
    out_nets = {}
    for i, (name, cfg) in enumerate(nets.items()):
        e = {"id": i, "identifier": name,
             "dispatch_deps_path": GRAPH.format(m=name, b="CPU")}
        if cfg.get("period"):
            e.update({"period": cfg["period"], "window_duration": cfg["period"],
                      "num_instances": cfg["instances"]})
        out_nets[name] = e
    spec = {
        "_comment": ("Concurrent multi-model contention probe. Instances are "
                     "PINNED and prune_periodic is off: the horizon is derived "
                     "from the heavy model's completion, so any improvement "
                     "would otherwise shrink the instance count and make two "
                     "rounds incomparable."),
        "hardware": {"machines": machines, "profile_hw": profile_hw,
                     "profile": {"target": "qrb5165_flowc", "topo_tag": "topo_0",
                                 "topo_tag_override": True, "gen_root": gen_root},
                     "p_core_speedup": 1.0},
        "scheduler": {"prune_periodic": False},
        "networks": out_nets,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(spec, open(out, "w"), indent=1)
    return out


# -------------------------------------------------------------------- actions
def schedule(spec: str, py: str, tag: str) -> str | None:
    base = os.path.basename(spec)[:-5]
    r = subprocess.run([py, os.path.join(HERE, "run_xpurt_schedule.py"),
                        "--networks-json", spec, "--solver", "greedy",
                        "--profiled", "--max-periodic-iters", "1"],
                       cwd=REPO, capture_output=True, text=True, timeout=1800)
    p = os.path.join(REPO, "schedules", f"scheduled_{base}_greedy_profiled.json")
    if r.returncode != 0 or not os.path.exists(p):
        print(f"  [{tag}] schedule FAILED: {(r.stderr or r.stdout)[-300:]}")
        return None
    return p


def predicted_overlap(sched: str) -> dict:
    """How much concurrency the schedule PREDICTS -- the precondition."""
    s = json.load(open(sched))
    iv = sorted((float(v["start_time"]),
                 float(v["start_time"]) + float(v["duration"]))
                for v in s["dispatches"].values())
    span = max(e for _, e in iv) if iv else 0.0
    busy = sum(e - b for b, e in iv)
    ov, cur = 0.0, 0.0
    for b, e in iv:
        if b < cur:
            ov += min(e, cur) - b
        cur = max(cur, e)
    return {"span_ms": round(span, 3), "sum_busy_ms": round(busy, 3),
            "overlap_ms": round(ov, 3),
            "concurrency": round(busy / span, 3) if span else 0.0,
            "n_dispatches": len(iv)}


def run_on_board(sched: str, tag: str, out_dir: str, dry: bool) -> str | None:
    """runtime -> stage -> run, all under the shared board lock."""
    log_dir = os.path.join(REPO, "runs", tag)
    cmd = [sys.executable, os.path.join(REPO, "qnn_models", "flow_c", "flow_c.py"),
           "all", "--schedule", sched, "--lane-mode", "kind-network",
           "--tag", tag, "--log-dir", log_dir]
    if dry:
        print(f"  [{tag}] DRY RUN, would execute:\n      {' '.join(cmd)}")
        return None
    env = dict(os.environ, BOARD_LOCK=BOARD_LOCK)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       timeout=5400, env=env)
    trace = os.path.join(log_dir, "trace.csv")
    open(os.path.join(out_dir, f"{tag}.board.log"), "w").write(
        (r.stdout or "") + "\n===stderr===\n" + (r.stderr or ""))
    if not os.path.exists(trace):
        print(f"  [{tag}] no trace at {trace}; see {tag}.board.log")
        return None
    return trace


def fit_from(trace: str, conditioned: bool) -> dict:
    import flowc_residual_feedback as F
    rows = F.read_trace(trace)
    model = F.fit(rows, conditioned)
    model["error"] = {"before": F.error(rows, None), "after": F.error(rows, model)}
    model["observed_corunners"] = sum(1 for r in rows if r["co"] > 0)
    model["n"] = len(rows)
    return model


def apply_correction(model: dict, out_root: str) -> str:
    """Write a corrected profile tree; the solo tree on disk is never edited."""
    import flowc_residual_feedback as F
    mp = os.path.join(out_root, "model.json")
    os.makedirs(out_root, exist_ok=True)
    json.dump(model, open(mp, "w"), indent=1)
    ns = argparse.Namespace(model=mp, gen_root=os.path.join(REPO, "gen"),
                            out_root=out_root, target="qrb5165_flowc")
    F.cmd_apply(ns)
    return out_root


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="results/flowc_contention")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="everything host-side; print the board commands")
    ap.add_argument("--wait-for-board", action="store_true")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-wait-h", type=float, default=12.0)
    a = ap.parse_args()
    out = os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)

    # A workload chosen to overlap: three networks, three lanes, periods short
    # enough that the contenders are in flight during the heavy model.
    # Chosen from the measured per-lane costs so the four models WANT different
    # lanes and therefore actually co-run. On the three lanes available without
    # GPU (dronet and yolov8n have no GPU cell):
    #     vint        CPU 121.99   (DSP 3796, HTA 12198 -- CPU is the only sane lane)
    #     yolov8n     DSP  28.64
    #     dronet      HTA   2.03
    #     mlp_control CPU   0.11   (deliberately on vint's lane: real contention)
    # The first attempt used vint + mlp + dronet and predicted 0.86 ms of
    # overlap in an 82.7 ms span -- serial, and worthless for this purpose.
    nets = {
        "vint":        {},                                  # heavy, aperiodic, CPU
        "yolov8n":     {"period": 40.0, "instances": 3},    # heavy periodic, DSP
        "dronet":      {"period": 5.0,  "instances": 24},   # light periodic, HTA
        "mlp_control": {"period": 2.0,  "instances": 60},   # contends on CPU
    }
    gap = coverage(list(nets), ["HTA", "DSP", "CPU"])
    if gap:
        print(f"  ABORT: missing profile cells {gap}")
        return 2
    print(f"  workload: {', '.join(nets)}  on HTA/DSP/CPU")

    if not a.dry_run:
        if board_free():
            print("  board lock is free")
        elif a.wait_for_board:
            print(f"  board is BUSY — waiting (poll {a.poll_s}s, max {a.max_wait_h}h)")
            if not wait_for_board(a.poll_s, int(a.max_wait_h * 3600)):
                print("  timed out waiting for the board; nothing was run")
                return 3
        else:
            print("  board is BUSY. Re-run with --wait-for-board, or --dry-run "
                  "to prepare everything host-side.")
            return 3

    rounds = []
    gen_root, model = "gen", None
    for rnd in range(a.rounds):
        tag = f"contend_r{rnd}"
        spec = build_spec(nets, os.path.join(REPO, "data", "toplevel", f"{tag}.json"),
                          gen_root=gen_root)
        sched = schedule(spec, a.python, tag)
        if not sched:
            return 1
        pred = predicted_overlap(sched)
        rec = {"round": rnd, "gen_root": gen_root, "predicted": pred}
        print(f"  [{tag}] predicted span={pred['span_ms']} ms  "
              f"concurrency={pred['concurrency']}x  overlap={pred['overlap_ms']} ms")
        frac = pred["overlap_ms"] / pred["span_ms"] if pred["span_ms"] else 0.0
        rec["overlap_fraction"] = round(frac, 4)
        if frac < 0.10:
            print(f"  WARNING: only {frac*100:.1f}% of the span has two "
                  "dispatches in flight. A near-serial schedule teaches a "
                  "contention model nothing — this is the precondition, not a "
                  "detail. Retune the periods or lanes before spending board "
                  "time on it.")

        trace = run_on_board(sched, tag, out, a.dry_run)
        if trace:
            shutil.copy(trace, os.path.join(out, f"{tag}.trace.csv"))
            model = fit_from(trace, conditioned=(rnd >= 1))
            rec["measured"] = {
                "n": model["n"],
                "dispatches_with_corunners": model["observed_corunners"],
                "logerr_before": model["error"]["before"]["logerr_median"],
                "logerr_after": model["error"]["after"]["logerr_median"],
                "mae_before_ms": model["error"]["before"]["mae_ms"],
                "mae_after_ms": model["error"]["after"]["mae_ms"],
                "factors": {k: v["factor"] for k, v in model["backends"].items()},
            }
            print(f"  [{tag}] measured: {model['n']} dispatches, "
                  f"{model['observed_corunners']} with co-runners; "
                  f"logerr {rec['measured']['logerr_before']} -> "
                  f"{rec['measured']['logerr_after']}")
            gen_root = os.path.relpath(
                apply_correction(model, os.path.join(REPO, f"gen_corrected_r{rnd}")),
                REPO)
        rounds.append(rec)
        json.dump(rounds, open(os.path.join(out, "rounds.json"), "w"), indent=1)

    print(f"\n  wrote {out}/rounds.json")
    if a.dry_run:
        print("  DRY RUN: specs built and scheduled; no board step was taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
