#!/usr/bin/env python3
"""Stage, run and collect the ROS 2 pinning baseline on the QRB5165.

    stage    plans/*.json      -> configs/<cell>__<aid>.cfg   (+ push to board)
    floor    measure the SingleThreadedExecutor empty-callback floor (L3)
    run      execute the staged configs on the board, 3 reps each
    collect  logs/ -> measured.json

Board discipline: every board interaction is a `timeout -s KILL` around an
`ssh -n` whose remote command sits inside `flock -w 900 /tmp/qnn_board.lock -c`.
The board is shared, so the lock wait is measured and recorded per call. The
CPU governor is saved before the campaign and restored after it; the frozen
cost model both sides are scored against was captured at `performance`.

Assignments are batched per ssh call (BATCH) so the lock is taken once for
several runs rather than once per run, without holding it for a whole large
cell.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

BOARD = os.environ.get("BOARD", "root@qrb5165")
LOCK = "/tmp/qnn_board.lock"
LOCK_WAIT = 900
BOARD_DIR = "/root/rospin"
SSH_OPTS = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8"]

CONFIGS = os.path.join(SWEEP, "configs")
LOGS = os.path.join(SWEEP, "logs")
PLANS = os.path.join(SWEEP, "plans")
STATE = os.path.join(SWEEP, "results", "state.json")

REPS = 3
WARM = 1            # discarded pass before each measured pass, matching the
                    # XPU-RT sweep's FLOWC_ITERATIONS=2
TIMEOUT_MS = 60000
BATCH = 4           # assignments per ssh/flock session


# ------------------------------------------------------------------ board
def board_run(cmd, timeout_s, capture=True):
    """One `timeout -s KILL <t> ssh -n BOARD "flock ... -c '<cmd>'"` call."""
    remote = f"flock -w {LOCK_WAIT} {LOCK} -c {shquote(cmd)}"
    argv = ["timeout", "-s", "KILL", str(timeout_s), "ssh", *SSH_OPTS, "-n",
            BOARD, remote]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=capture, text=True)
    return dict(rc=p.returncode, wall_s=round(time.time() - t0, 2),
                out=(p.stdout or ""), err=(p.stderr or ""))


def shquote(s):
    return "'" + s.replace("'", "'\\''") + "'"


ROS_ENV = ("source /opt/ros/foxy/setup.bash; "
           "source /root/ros2_ws/install/setup.bash; "
           "export LD_LIBRARY_PATH=/root/qairt/lib/target:$LD_LIBRARY_PATH; "
           'export ADSP_LIBRARY_PATH="/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp"; '
           "export RCUTILS_LOGGING_BUFFERED_STREAM=0; ")

HARNESS = "/root/ros2_ws/install/ros_qnn_baseline/lib/ros_qnn_baseline/pin_harness"


def gov(action):
    if action == "save":
        c = ("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor "
             "> /tmp/rospin_prev_governor; cat /tmp/rospin_prev_governor")
    elif action == "perf":
        c = ("for c in $(seq 0 7); do echo performance > "
             "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor "
             "2>/dev/null; done; "
             "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    else:
        c = ("g=$(cat /tmp/rospin_prev_governor 2>/dev/null || echo schedutil); "
             "for c in $(seq 0 7); do echo $g > "
             "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor "
             "2>/dev/null; done; "
             "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    r = board_run(c, 90)
    return r["out"].strip()


# ------------------------------------------------------------------ stage
def cfg_text(plan, a):
    """The line-oriented config pin_harness reads, for one assignment."""
    L = [f"# {plan['cell']} / {a['id']} / {a['label']}",
         f"# predicted makespan {a['predicted_makespan_ms']} ms, "
         f"np {a['predicted_np_makespan_ms']} ms",
         f"reps {REPS}", f"warm {WARM}", f"timeout_ms {TIMEOUT_MS}"]
    for n in a["nodes"]:
        per = n["period_ms"] if n["period_ms"] is not None else -1.0
        win = n["window_ms"] if n["window_ms"] is not None else -1.0
        L.append(f"net {n['network']} {n['backend']} {n['backend_lib']} "
                 f"{n['num_instances']} {per} {win}")
        for t in n["tiles"]:
            L.append(f"tile {t['ctx']} {t['graph']}")
    for e in plan["edges"]:
        L.append(f"edge {e['from']} {e['to']}")
    return "\n".join(L) + "\n"


def cmd_stage(args):
    os.makedirs(CONFIGS, exist_ok=True)
    n = 0
    todo = []
    for fn in sorted(os.listdir(PLANS)):
        plan = json.load(open(os.path.join(PLANS, fn)))
        for a in plan["assignments"]:
            if not a.get("measure"):
                continue
            tag = f'{plan["cell"]}__{a["id"]}'
            with open(os.path.join(CONFIGS, tag + ".cfg"), "w") as f:
                f.write(cfg_text(plan, a))
            todo.append(tag)
            n += 1
    print(f"staged {n} configs into {CONFIGS}")
    if args.push:
        subprocess.run(["timeout", "-s", "KILL", "300", "bash", "-c",
                        f"tar czf - -C {shquote(SWEEP)} configs | "
                        f"ssh {' '.join(SSH_OPTS)} {BOARD} "
                        f"\"flock -w {LOCK_WAIT} {LOCK} -c 'mkdir -p {BOARD_DIR} && "
                        f"rm -rf {BOARD_DIR}/configs && tar xzf - -C {BOARD_DIR} && "
                        f"ls {BOARD_DIR}/configs | wc -l'\""], check=True)
    with open(os.path.join(SWEEP, "results", "staged.json"), "w") as f:
        json.dump(todo, f, indent=1)
    return 0


# ------------------------------------------------------------------ floor
def cmd_floor(args):
    """L3: the executor floor, measured rather than assumed."""
    out = {}
    for per in (0.5, 1.0, 2.0, 5.0):
        c = (ROS_ENV + f"{HARNESS} --floor {per} --floor-iters 1500 2>/dev/null")
        r = board_run(c, 180)
        line = [l for l in r["out"].splitlines() if l.startswith("[floor]")]
        print(r["out"].strip() or r["err"][-400:])
        if line:
            m = dict(re.findall(r"(\w+)=([\d.]+)", line[0]))
            out[str(per)] = {k: float(v) for k, v in m.items()}
    p = os.path.join(SWEEP, "results", "floor.json")
    with open(p, "w") as f:
        json.dump({"_comment":
                   "Harness limit L3: measured empty-callback period of one "
                   "rclcpp SingleThreadedExecutor on this board, so a declared "
                   "period below the floor is reported per cell rather than "
                   "silently absorbed. p50/p95/max are of the fire-to-fire gap.",
                   "board": BOARD, "floors": out}, f, indent=1)
    print(f"wrote {p}")
    return 0


# -------------------------------------------------------------------- run
SUMMARY = re.compile(r"\[summary\] rep=(\d+) executed=(\d+)/(\d+) "
                     r"makespan=([\d.]+) np_makespan=([\d.]+)")


def parse_log(txt):
    out = {"passes": [], "ok": False}
    for m in SUMMARY.finditer(txt):
        out["passes"].append(dict(rep=int(m.group(1)), executed=int(m.group(2)),
                                  total=int(m.group(3)),
                                  makespan_ms=float(m.group(4)),
                                  np_makespan_ms=float(m.group(5))))
    tr = re.search(r"ROS_PINSWEEP_TRACE_BEGIN[^\n]*\n(.*?)\n[^\n]*"
                   r"ROS_PINSWEEP_TRACE_END", txt, re.S)
    rows = []
    if tr:
        rows = list(csv.DictReader(io.StringIO(tr.group(1).strip())))
    out["trace_rows"] = len(rows)
    out["rows"] = rows
    if out["passes"]:
        out["ok"] = all(p["executed"] == p["total"] for p in out["passes"]) \
                    and len(out["passes"]) == REPS
    return out


def cmd_run(args):
    todo = json.load(open(os.path.join(SWEEP, "results", "staged.json")))
    if args.only:
        pats = args.only.split(",")
        todo = [t for t in todo if any(p in t for p in pats)]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    os.makedirs(LOGS, exist_ok=True)
    pending = [t for t in todo
               if args.force or not (st.get(t, {}).get("ok"))]
    print(f"{len(pending)} of {len(todo)} assignments pending")
    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        parts = []
        for tag in chunk:
            parts.append(
                f'echo "=== RUN {tag} ==="; '
                f'{HARNESS} --config {BOARD_DIR}/configs/{tag}.cfg 2>&1; '
                f'echo "=== END {tag} rc=$? ==="')
        cmd = ROS_ENV + " ".join(parts)
        r = board_run(cmd, args.timeout * len(chunk) + 120)
        blob = r["out"] + "\n" + r["err"]
        for tag in chunk:
            m = re.search(rf"=== RUN {re.escape(tag)} ===\n(.*?)=== END "
                          rf"{re.escape(tag)} rc=(\d+) ===", blob, re.S)
            body = m.group(1) if m else blob
            rc = int(m.group(2)) if m else -1
            with open(os.path.join(LOGS, tag + ".log"), "w") as f:
                f.write(body)
            info = parse_log(body)
            info.pop("rows", None)
            st[tag] = dict(ok=info["ok"], rc=rc, lock_wall_s=r["wall_s"],
                           passes=info["passes"], trace_rows=info["trace_rows"])
            flag = "ok " if info["ok"] else "BAD"
            ms = [p["makespan_ms"] for p in info["passes"]]
            print(f"  {flag} {tag:52s} rc={rc} "
                  f"makespans={[round(x, 2) for x in ms]}")
        with open(STATE, "w") as f:
            json.dump(st, f, indent=1)
    return 0


# ---------------------------------------------------------------- collect
def cmd_collect(args):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    plans = {}
    for fn in sorted(os.listdir(PLANS)):
        p = json.load(open(os.path.join(PLANS, fn)))
        plans[p["cell"]] = p
    xrt = load_xpurt()
    out = {}
    for tag, rec in sorted(st.items()):
        cell, aid = tag.split("__")
        plan = plans[cell]
        a = next(x for x in plan["assignments"] if x["id"] == aid)
        log = os.path.join(LOGS, tag + ".log")
        if not os.path.exists(log):
            continue
        info = parse_log(open(log, errors="replace").read())
        walls = [p["makespan_ms"] for p in info["passes"]]
        nps = [p["np_makespan_ms"] for p in info["passes"]]
        if not walls:
            continue
        per_net = summarize_rows(info["rows"], plan)
        starved = [n for n, v in per_net.items() if v["instances_run"] == 0]
        rec2 = dict(
            cell=cell, family=plan["family"], config=plan["config"],
            assignment=aid, label=a["label"], assign=a["assign"],
            rank=a["rank"], np_rank=a["np_rank"],
            n_legal=plan["n_legal"], measure_mode=plan["measure_mode"],
            verdict=plan["verdict"], notes=plan["notes"],
            predicted_makespan_ms=a["predicted_makespan_ms"],
            predicted_np_makespan_ms=a["predicted_np_makespan_ms"],
            predicted_missed_instances=a["predicted_missed_instances"],
            ok=info["ok"], reps=len(walls),
            makespan_median_ms=round(statistics.median(walls), 4),
            makespan_spread_ms=round(max(walls) - min(walls), 4),
            makespan_reps_ms=walls,
            np_median_ms=round(statistics.median(nps), 4),
            np_spread_ms=round(max(nps) - min(nps), 4),
            np_reps_ms=nps,
            per_net=per_net,
            starved=starved, n_starved=len(starved),
            missed_instances=sum(v["missed"] for v in per_net.values()),
            missed_nets=[n for n, v in per_net.items() if v["missed"]],
            lock_wall_s=rec.get("lock_wall_s"),
        )
        rec2["measured_over_predicted"] = (
            round(rec2["makespan_median_ms"] / a["predicted_makespan_ms"], 4)
            if a["predicted_makespan_ms"] else None)
        out[tag] = rec2

    doc = {
        "_comment": (
            "Measured ROS 2 whole-network-pinning baseline. One record per "
            "(cell, assignment). Medians are over 3 reps and the spread is "
            "quoted with them; no single-rep number appears as a result. "
            "`makespan` is the wall clock from t0 to the last instance of any "
            "network completing -- the same quantity XPU-RT's runtime prints "
            "as `[summary] wall=`. `np` is the makespan over NON-PERIODIC "
            "networks only, which is the objective the sweep10 solver "
            "comparison actually ranks on. Window feasibility (`missed_*`) is "
            "reported separately and is never folded into the makespan."),
        "board": BOARD, "reps": REPS, "warm_passes_per_rep": WARM,
        "xpurt_sweep": os.path.relpath(
            os.path.join(FLOWC, "sweeps",
                         "qrb5165_sched_algo_sweep10_20260908-210226"), REPO),
        "runs": out,
        "xpurt": xrt,
    }
    p = os.path.join(SWEEP, "measured.json")
    with open(p, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {p}: {len(out)} measured assignments")
    return 0


def summarize_rows(rows, plan):
    """Per-network instance counts, latency and window misses from the trace."""
    per = {}
    for n, spec in plan["networks"].items():
        per[n] = dict(instances_declared=spec["num_instances"],
                      instances_run=0, exec_p50_ms=None, exec_max_ms=None,
                      last_end_ms=None, missed=0,
                      period_ms=spec["period_ms"], window_ms=spec["window_ms"])
    if not rows:
        return per
    # only the measured passes of the LAST rep-set; keep every measured pass
    dur = {}
    for r in rows:
        if r.get("warm") != "0":
            continue
        n = r["network"]
        if n not in per:
            continue
        d = float(r["end_ms"]) - float(r["start_ms"])
        dur.setdefault(n, []).append(d)
        per[n]["instances_run"] = max(
            per[n]["instances_run"], int(r["instance"]) + 1)
        e = float(r["end_ms"])
        per[n]["last_end_ms"] = max(per[n]["last_end_ms"] or 0.0, e)
        w = float(r["window_ms"])
        if w > 0 and e > float(r["release_ms"]) + w + 1e-9:
            per[n]["missed"] += 1
    for n, v in dur.items():
        per[n]["exec_p50_ms"] = round(statistics.median(v), 4)
        per[n]["exec_max_ms"] = round(max(v), 4)
        per[n]["n_exec_samples"] = len(v)
    # misses were counted over every measured rep; normalise to per-rep
    reps = max(1, len({r["rep"] for r in rows if r.get("warm") == "0"}))
    for n in per:
        per[n]["missed"] = round(per[n]["missed"] / reps, 2)
    return per


def load_xpurt():
    """Measured XPU-RT makespans per cell, from the sweep's phase4 results."""
    p = os.path.join(FLOWC, "sweeps",
                     "qrb5165_sched_algo_sweep10_20260908-210226",
                     "results", "phase4_results.json")
    rows = json.load(open(p))
    out = {}
    for r in rows:
        if not r.get("measured_median_ms"):
            continue
        cell = r["workload"]
        e = out.setdefault(cell, {"solvers": {}})
        e["solvers"][r["solver"]] = dict(
            makespan_median_ms=r["measured_median_ms"],
            makespan_spread_ms=r.get("measured_spread_ms"),
            makespan_reps_ms=r.get("measured_reps_ms"),
            np_median_ms=r.get("measured_np_median_ms"),
            np_spread_ms=r.get("measured_np_spread_ms"),
            np_reps_ms=r.get("measured_np_reps_ms"),
            predicted_ms=r.get("table_predicted_makespan_ms"),
            misses=r.get("misses"), sched_hash=r.get("sched_hash"))
    for cell, e in out.items():
        s = e["solvers"]
        e["best_makespan_ms"] = min(v["makespan_median_ms"] for v in s.values())
        e["best_makespan_solver"] = min(
            s, key=lambda k: s[k]["makespan_median_ms"])
        nps = {k: v["np_median_ms"] for k, v in s.items() if v["np_median_ms"]}
        if nps:
            e["best_np_ms"] = min(nps.values())
            e["best_np_solver"] = min(nps, key=nps.get)
        e["greedy_makespan_ms"] = s.get("greedy", {}).get("makespan_median_ms")
        e["greedy_np_ms"] = s.get("greedy", {}).get("np_median_ms")
        e["n_solvers_measured"] = len(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.set_defaults(fn=cmd_stage)
    s.add_argument("--push", action="store_true")
    s = sub.add_parser("floor"); s.set_defaults(fn=cmd_floor)
    s = sub.add_parser("run"); s.set_defaults(fn=cmd_run)
    s.add_argument("--only", default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--timeout", type=int, default=240)
    s = sub.add_parser("collect"); s.set_defaults(fn=cmd_collect)
    s = sub.add_parser("gov"); s.set_defaults(fn=lambda a: print(gov(a.action)))
    s.add_argument("action", choices=["save", "perf", "restore"])
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
