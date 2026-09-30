#!/usr/bin/env python3
"""Catalogue every ROS 2 deployment we have run on the K1, from the runs themselves.

A hand-written table of arms drifts: 22 of the 52 arms measured on this board were missing from
docs/Baselines/ros_baseline_reproduction.md, including ones the figure and the effort ladder use. So this reads
each run's own manifest -- what the processes, pinning, pool, executor, QoS and control mode ACTUALLY
were -- and the measured cadence and latency beside them, and writes the catalogue. Regenerate it
whenever a new arm is run; nothing here is typed by hand.

    scripts/ros_arms_catalog.py --out docs/Baselines/ros_arms_catalog.md
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from flight_quarantine import flight_rows   # noqa: E402

RES = os.path.join(REPO, "results", "codesign_feedback")


def procs_of(man):
    """The per-process view, whether the run recorded one process or several."""
    p = man.get("processes")
    return p if isinstance(p, dict) and p else {"(single process)": man}


def describe(man):
    ps = procs_of(man)
    # the control fields belong to whichever process runs the control node: in the pinned layouts the
    # perception process still carries a default ctrl_mode, which labelled the chained arms "timer"
    any_p = next((v for v in ps.values() if "control" in str(v.get("nodes", ""))), next(iter(ps.values())))
    parts = []
    for name, v in ps.items():
        bits = [f"{name}: {v.get('nodes','?')}"]
        # Worker pools are read from whatever the manifest recorded rather than from a list of
        # pools this file knows about: any `<what>_pool` key with a width is reported, and its hart
        # list is the matching `<what>_harts` (the YOLO pool writes its list as plain `pool_harts`).
        # A node given a new pool therefore shows up here without this file learning its name. An
        # empty hart list is the unpinned case -- the pool exists and the OS places its workers.
        for key in sorted(k for k in v if k.endswith("_pool") and v.get(k)):
            what = key[:-len("_pool")]
            harts = v.get(f"{what}_harts") or (v.get("pool_harts") if what == "yolo" else "")
            label = "YOLO" if what == "yolo" else what     # an acronym, not a per-pool rule
            bits.append(f"{label} pool {v[key]} on harts {harts}" if harts
                        else f"{label} pool {v[key]}, unpinned")
        mask = str(v.get("affinity_mask", ""))
        if mask and mask != "0xff":
            bits.append(f"pinned {mask}")
        parts.append(", ".join(bits))
    # cameras/alternate belong to the process that RUNS the camera: downstream processes record how
    # many topics they subscribe to, so reading them called the one-sensor alternating arm "2 cameras"
    cam_p = next((v for v in ps.values() if re.fullmatch(r"camera\d*", str(v.get("nodes", "")).split(",")[0] or "")),
                 next(iter(ps.values())))
    return {
        "layout": " | ".join(parts),
        "processes": len(ps),
        "executor": any_p.get("executor"),
        "ctrl_mode": any_p.get("ctrl_mode"),
        "ctrl_hz": any_p.get("ctrl_hz"),
        "qos_depth": any_p.get("qos_depth"),
        "cameras": cam_p.get("cameras"),
        "alternate": cam_p.get("alternate"),
        "extra": (any_p.get("extra") or "").strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "Baselines", "ros_arms_catalog.md"))
    a = ap.parse_args()

    summary = {}
    p = os.path.join(RES, "ros_traced", "summary.csv")
    for r in csv.DictReader(open(p)):
        summary[r["tag"]] = r

    flights = collections.Counter(); flown = collections.Counter()
    for f in glob.glob(os.path.join(RES, "campaign*", "campaign*.csv")):
        for r in flight_rows(f):
            t = (r.get("ctrl_trace") or "").split("/")[-1]
            if t:
                flights[t] += 1; flown[t] += r["outcome"] == "success"

    arms = collections.defaultdict(dict)          # arm -> hz -> [tags]
    aborted = []                                  # runs that recorded no goals
    for d in sorted(glob.glob(os.path.join(RES, "ros_traced", "*_r[0-9]*"))):
        tag = os.path.basename(d)
        m = re.match(r"(\d+)_(.+?)_r(\d+)$", tag)
        mf = os.path.join(d, "manifest.json")
        if not m or not os.path.exists(mf):
            continue
        # A run that recorded no goals never completed a perception->navigation chain, so it carries
        # no camera->goal time and no control cadence. Listing it as an arm invents a deployment out
        # of an aborted run, so it is counted and named below instead.
        if summary.get(tag) and not (summary[tag].get("n_goals") or "0").strip("0 "):
            aborted.append(tag)
            continue
        arms[m.group(2)].setdefault(int(m.group(1)), []).append((tag, mf))

    out = ["# Every ROS 2 deployment measured on the K1",
           "",
           "Generated by `scripts/ros_arms_catalog.py` from each run's own manifest and the per-run",
           "roll-up, so it says what the processes, pinning, pool, executor, QoS and control mode",
           "actually were. Regenerate after running a new arm; do not edit by hand.",
           "",
           f"{len(arms)} arms, {sum(len(v) for hz in arms.values() for v in hz.values())} runs.",
           ""]
    if aborted:
        out += ["Not listed: " + ", ".join(f"`{t}`" for t in sorted(aborted))
                + " -- recorded no goals, so the run produced no chain timing to report.", ""]
    for arm in sorted(arms):
        rates = arms[arm]
        first = describe(json.load(open(next(iter(rates.values()))[0][1])))
        out.append(f"## `{arm}`")
        out.append("")
        out.append(f"* layout: {first['layout']}")
        out.append(f"* {first['processes']} process(es), {first['executor']} executor, "
                   f"control {first['ctrl_mode']}"
                   + (f" at {first['ctrl_hz']} Hz" if first["ctrl_mode"] == "timer" else " (chained to the goal)")
                   + f", QoS depth {first['qos_depth']}"
                   + (f", {first['cameras']} camera(s)" if first["cameras"] else "")
                   + (" (one sensor, frames alternating between the perception instances)"
                      if first["alternate"] else "")
                   + (f", extra nodes {first['extra']}" if first["extra"] else ""))
        out.append("")
        out.append("| camera | runs | control | camera→goal | flights (completed/flown) |")
        out.append("|---|---|---|---|---|")
        for hz in sorted(rates):
            tags = rates[hz]
            gaps = [float(summary[t]["gap_mean_ms"]) for t, _ in tags
                    if t in summary and summary[t].get("gap_mean_ms")]
            e2e = [float(summary[t]["e2e_goal_med_ms"]) for t, _ in tags
                   if t in summary and summary[t].get("e2e_goal_med_ms")]
            tr = f"ros_{arm}{hz}.csv"
            n = flights.get(tr, 0)
            out.append(f"| {hz} Hz | {len(tags)} | "
                       + (f"{1000/statistics.mean(gaps):.1f} Hz" if gaps else "—") + " | "
                       + (f"{statistics.median(e2e):.1f} ms" if e2e else "—") + " | "
                       + (f"{flown.get(tr,0)}/{n}" if n else "not flown") + " |")
        out.append("")
    open(a.out, "w").write("\n".join(out) + "\n")
    print(f"wrote {a.out}: {len(arms)} arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
