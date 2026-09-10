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
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offsets

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))

BOARD = os.environ.get("BOARD", "root@10.44.120.201")
LOCK = "/tmp/qnn_board.lock"
LOCK_WAIT = 900
BOARD_DIR = "/root/rospin"
SSH_OPTS = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8"]

CONFIGS = os.path.join(SWEEP, "configs")
LOGS = os.path.join(SWEEP, "logs")
PLANS = os.path.join(SWEEP, "plans")
PLANS3 = os.path.join(SWEEP, "plans3net")   # the 3net arm
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


def plan_dirs(arm="all"):
    d = []
    if arm in ("all", "main"):
        d.append(PLANS)
    if arm in ("all", "3net") and os.path.isdir(PLANS3):
        d.append(PLANS3)
    return d


def cfg_hash(plan, a):
    """Content hash of what the harness actually reads, for dedupe.

    The provenance comments name the cell and the assignment id, so they are
    stripped: they would make every config unique by construction and the
    dedupe would never fire. What is left is the network list, the backend
    each is pinned to, the tiles, the instance counts, the periods, the
    windows and the edges -- i.e. the whole experiment.

    Duplicates are real and they are CROSS-CELL. The harness is never told
    which lanes the cell declares, only where each network goes, so a `quad`
    placement that happens to use no HTA is byte-identical to the `dc`
    placement of the same family, and the board would be asked to run the
    same experiment twice. Measuring it once and pointing the other tag at
    that log is the same discipline Phase 4 and the 3net arm use on
    schedules.
    """
    body = "\n".join(l for l in cfg_text(plan, a).splitlines()
                     if not l.startswith("#"))
    return hashlib.sha256(body.encode()).hexdigest()


def cmd_stage(args):
    os.makedirs(CONFIGS, exist_ok=True)
    n = 0
    todo, hashes = [], {}
    for d in plan_dirs(args.arm):
      for fn in sorted(os.listdir(d)):
        plan = json.load(open(os.path.join(d, fn)))
        for a in plan["assignments"]:
            if not a.get("measure"):
                continue
            tag = f'{plan["cell"]}__{a["id"]}'
            with open(os.path.join(CONFIGS, tag + ".cfg"), "w") as f:
                f.write(cfg_text(plan, a))
            hashes[tag] = cfg_hash(plan, a)
            todo.append(tag)
            n += 1
    with open(os.path.join(SWEEP, "results", "cfg_hashes.json"), "w") as f:
        json.dump({"_comment":
                   "sha256 of each staged config with its provenance comments "
                   "stripped -- the dedupe key. Tags that share a hash are the "
                   "same experiment and are measured once; see drive.py "
                   "cfg_hash and `duplicate_of` in results/state.json.",
                   "hashes": hashes}, f, indent=1)
    print(f"staged {n} configs into {CONFIGS} "
          f"({len(set(hashes.values()))} distinct harness inputs)")
    if args.push:
        subprocess.run(["timeout", "-s", "KILL", "300", "bash", "-c",
                        f"tar czf - -C {shquote(SWEEP)} configs | "
                        f"ssh {' '.join(SSH_OPTS)} {BOARD} "
                        f"\"flock -w {LOCK_WAIT} {LOCK} -c 'mkdir -p {BOARD_DIR} && "
                        f"rm -rf {BOARD_DIR}/configs && tar xzf - -C {BOARD_DIR} && "
                        f"ls {BOARD_DIR}/configs | wc -l'\""], check=True)
    name = "staged.json" if args.arm == "all" else f"staged_{args.arm}.json"
    with open(os.path.join(SWEEP, "results", name), "w") as f:
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
    name = "staged.json" if args.arm == "all" else f"staged_{args.arm}.json"
    todo = json.load(open(os.path.join(SWEEP, "results", name)))
    if args.only:
        pats = args.only.split(",")
        todo = [t for t in todo if any(p in t for p in pats)]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    os.makedirs(LOGS, exist_ok=True)
    pending = [t for t in todo
               if args.force or not (st.get(t, {}).get("ok"))]
    # Content-dedupe before spending board time. A tag whose harness input is
    # byte-identical to one already measured -- or to one earlier in this
    # pass -- records `duplicate_of` and is not run; `collect` reads the
    # canonical tag's log for it. Canonical choice is deterministic: an
    # already-measured tag if there is one, else the lexicographically first.
    hp = os.path.join(SWEEP, "results", "cfg_hashes.json")
    if os.path.exists(hp) and not args.no_dedupe:
        hashes = json.load(open(hp))["hashes"]
        groups = {}
        for t in todo:
            groups.setdefault(hashes.get(t, t), []).append(t)
        keep, n_dup = [], 0
        for t in pending:
            g = sorted(groups.get(hashes.get(t, t), [t]))
            canon = next((x for x in g
                          if os.path.exists(os.path.join(LOGS, x + ".log"))
                          and x != t), None)
            if canon is None:
                canon = next((x for x in g if x not in pending), None)
            if canon is None and g[0] != t:
                canon = g[0]
            if canon is not None and canon != t:
                st[t] = dict(duplicate_of=canon, cfg_hash=hashes.get(t))
                n_dup += 1
                continue
            keep.append(t)
        if n_dup:
            print(f"  {n_dup} assignment(s) resolved as content-duplicates "
                  f"of an existing run -- no board time")
            with open(STATE, "w") as f:
                json.dump(st, f, indent=1)
        pending = keep
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
    for d in plan_dirs("all"):
        for fn in sorted(os.listdir(d)):
            p = json.load(open(os.path.join(d, fn)))
            plans[p["cell"]] = p
    xrt = load_xpurt()
    out = {}
    # Iterate over the LOGS, not the state file: the state is a convenience
    # index and a crashed or restarted campaign can lose entries from it, while
    # a log on disk is the measurement.
    tags = sorted(set(f[:-len(".log")] for f in os.listdir(LOGS)
                      if f.endswith(".log") and "__" in f)
                  | {t for t, v in st.items() if v.get("duplicate_of")})
    for tag in tags:
        rec = st.get(tag, {})
        cell, aid = tag.split("__")
        plan = plans.get(cell)
        if plan is None:
            continue
        a = next((x for x in plan["assignments"] if x["id"] == aid), None)
        if a is None or not a.get("measure"):
            continue
        # A content-duplicate carries its OWN cell and assignment metadata and
        # the canonical tag's measurement; `measured_via` says which run it is,
        # so no number here is silently shared between two rows.
        via = rec.get("duplicate_of")
        log = os.path.join(LOGS, (via or tag) + ".log")
        if not os.path.exists(log):
            continue
        info = parse_log(open(log, errors="replace").read())
        walls = [p["makespan_ms"] for p in info["passes"]]
        nps = [p["np_makespan_ms"] for p in info["passes"]]
        if not walls:
            continue
        per_net = summarize_rows(info["rows"], plan)
        starved = [n for n, v in per_net.items() if v["instances_run"] == 0]
        # The start-barrier correction: every non-periodic makespan is also
        # re-derived from that rep's OWN first dispatch, so this side is not
        # charged for a startup delay the other side is not charged for
        # either. Raw and corrected are both kept; nothing is replaced. See
        # scripts/offsets.py for what it is and where it does not apply.
        aper = {n for n, v in plan["networks"].items()
                if v["period_ms"] is None}
        oreps = offsets.ros_reps(log, aper)
        o_med, o_spr, o_v = offsets.summarize(oreps, "corrected_ms")
        f_med, f_spr, f_v = offsets.summarize(oreps, "offset_ms")
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
            measured_via=via,
            np_corrected_median_ms=o_med,
            np_corrected_spread_ms=o_spr,
            np_corrected_reps_ms=o_v,
            offset_median_ms=f_med,
            offset_max_ms=(max(f_v) if f_v else None),
            offset_reps_ms=f_v,
            release_bound_reps=sum(1 for r in oreps if r["release_bound"]),
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


def xpurt_aperiodic(cell):
    """Non-periodic networks of a cell. `cell` already carries the
    `networks_` prefix -- phase4_results.json's `workload` field does."""
    p = os.path.join(REPO, "data", "toplevel", "s10port", cell + ".json")
    if not os.path.exists(p):
        return set()
    nets = json.load(open(p))["networks"]
    return {k for k, v in nets.items() if v.get("period") is None}


def xpurt_corrected(xsweep, point, aperiodic):
    """The start-barrier correction on the XPU-RT side, from its own run logs.

    Same rule as the ROS side (scripts/offsets.py): per rep, the end of the
    last aperiodic operation minus the start of that run's first dispatch,
    then the median over reps. A point whose logs are not on disk returns
    nothing rather than a guess -- the raw number stands alone and the
    analysis says so.
    """
    reps = []
    for rep in (1, 2, 3):
        log = os.path.join(xsweep, "runs", point, f"rep{rep}", "run.log")
        r = offsets.xpurt_rep(log, aperiodic)
        if r:
            reps.append(r)
    if not reps:
        return {}
    c_med, c_spr, c_v = offsets.summarize(reps, "corrected_ms")
    o_med, o_spr, o_v = offsets.summarize(reps, "offset_ms")
    return dict(np_corrected_median_ms=c_med, np_corrected_spread_ms=c_spr,
                np_corrected_reps_ms=c_v, offset_median_ms=o_med,
                offset_max_ms=(max(o_v) if o_v else None), offset_reps_ms=o_v,
                release_bound_reps=sum(1 for r in reps if r["release_bound"]),
                headstart_evidence=dict(
                    first_gate_ms=reps[0]["first_gate_ms"],
                    first_dep_wait_ms=reps[0]["first_dep_wait_ms"],
                    last_pred_start_ms=reps[0]["last_pred_start_ms"],
                    last_actual_start_ms=reps[0]["last_actual_start_ms"]))


def load_xpurt():
    """Measured XPU-RT makespans per cell, from the sweep's phase4 results."""
    xsweep = os.path.join(FLOWC, "sweeps",
                          "qrb5165_sched_algo_sweep10_20260908-210226")
    p = os.path.join(xsweep, "results", "phase4_results.json")
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
            misses=r.get("misses"), sched_hash=r.get("sched_hash"),
            measured_via=r.get("measured_via"),
            # `measured_via` is the point whose logs this row's numbers come
            # from: Phase 4 deduped identical schedules, so a solver can carry
            # a measurement without owning a run directory. Following it is
            # what makes the correction available on `cpsat:warmbest` for the
            # five cells where it deduped onto `greedy`.
            **xpurt_corrected(xsweep, r.get("measured_via") or r["point"],
                              xpurt_aperiodic(cell)))
    for cell, e in out.items():
        s = e["solvers"]
        e["best_makespan_ms"] = min(v["makespan_median_ms"] for v in s.values())
        e["best_makespan_solver"] = min(
            s, key=lambda k: s[k]["makespan_median_ms"])
        nps = {k: v["np_median_ms"] for k, v in s.items() if v["np_median_ms"]}
        if nps:
            e["best_np_ms"] = min(nps.values())
            e["best_np_solver"] = min(nps, key=nps.get)
        # the same oracle, on the corrected quantity. Taken over the corrected
        # numbers rather than re-using the raw winner: the correction can
        # reorder two solvers whose raw medians differed by less than one of
        # their head starts, and carrying the raw winner over would hide that.
        cnp = {k: v["np_corrected_median_ms"] for k, v in s.items()
               if v.get("np_corrected_median_ms")}
        if cnp:
            e["best_np_corrected_ms"] = min(cnp.values())
            e["best_np_corrected_solver"] = min(cnp, key=cnp.get)
        e["greedy_makespan_ms"] = s.get("greedy", {}).get("makespan_median_ms")
        e["greedy_np_ms"] = s.get("greedy", {}).get("np_median_ms")
        e["n_solvers_measured"] = len(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.set_defaults(fn=cmd_stage)
    s.add_argument("--push", action="store_true")
    s.add_argument("--arm", default="all", choices=["all", "main", "3net"])
    s = sub.add_parser("floor"); s.set_defaults(fn=cmd_floor)
    s = sub.add_parser("run"); s.set_defaults(fn=cmd_run)
    s.add_argument("--arm", default="all", choices=["all", "main", "3net"])
    s.add_argument("--no-dedupe", action="store_true",
                   help="measure every marked assignment even where two are "
                        "the same experiment; the default resolves those "
                        "against the existing run")
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
