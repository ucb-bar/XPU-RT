#!/usr/bin/env python3
"""The ROS 2 effort ladder: what each unit of expert work buys the baseline, measured.

Every rung is a real deployment of the same 3-network chain on the K1, differing from the one above
it by ONE change a ROS 2 engineer would make: a worker pool for YOLO, a wider pool, keep-last-1
queues, a control timer, hand-pinning, a second model instance. For each rung this prints what the
board measured (harts doing work, control cadence, camera-to-goal latency) and how the flights that
replay those measurements came out, so the reader can see where the figure's baseline sits on the
ladder and how far the rungs above it get.

Nothing here is a new measurement: the timing comes from scripts/measured_timing.py (which
re-derives every row from the board traces under --verify) and the flights from the campaign CSVs
through flight_quarantine, so a reviewer can re-run this and get the table in the paper.

    scripts/ros_effort_ladder.py [--json out.json]
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
import measured_timing as MT   # noqa: E402
from flight_quarantine import flight_rows   # noqa: E402

RES = os.path.join(REPO, "results", "codesign_feedback")

# (arm@camera, what this rung adds, the ctrl trace its flights replay)
LADDER = [
    ("vanilla@45",        "ROS 2 as it ships: serial YOLO, one process, unpinned", "ros_vanilla45.csv"),
    ("vanilla4@45",       "+ a 4-hart worker pool for YOLO",                        "ros_vanilla445.csv"),
    ("vanilla8@45",       "+ the pool widened to all 8 harts",                      "ros_vanilla845.csv"),
    ("vanilla4_q1@45",    "+ keep-last-1 queues (one QoS argument)",                "ros_vanilla4_q145.csv"),
    ("vanilla4tm@45",     "+ control on its own 100 Hz timer",                      "ros_vanilla4tm45.csv"),
    ("p3@45",             "hand-pinned: 3 processes over 6 cores",                  "ros_p345.csv"),
    ("p3_q1@45",          "hand-pinned + keep-last-1",                              "ros_p3_q145.csv"),
    ("vanilla4x2@45",     "two model instances across both clusters (all 8 cores)", "ros_vanilla4x245.csv"),
    ("vanilla4x2@36",     "two instances at the 36 Hz camera the figure draws",     "ros_vanilla4x236.csv"),
    ("vanilla4x2tm@45",   "two instances + control on its own timer",               "ros_vanilla4x2tm45.csv"),
]


def hart_counts():
    """Harts doing real work per arm: >=20 % busy in the per-core sampler, median over replicates.

    Summed per run first: the OS moves a busy thread between cores, so a per-column median counts
    the same work twice and the arm looks wider than it ran.
    """
    out = collections.defaultdict(list)
    p = os.path.join(RES, "ros_traced", "summary.csv")
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        m = re.match(r"(\d+)_(.+?)_r(\d+)$", r["tag"])
        if not m:
            continue
        busy = [float(r[f"busy_c{c}"]) for c in range(8) if r.get(f"busy_c{c}")]
        if busy:
            out[f"{m.group(2)}@{m.group(1)}"].append(sum(b >= 20 for b in busy))
    return out


# The reference arm the figure draws beside the rungs; it shares the flight column, so it shares the
# cell intersection too.
XPU_TRACE = "xpu_a_cpsat_hard.csv"

# Every cadence trace scored in the flight column. One list, so the table and the figure intersect
# over the same arms and cannot disagree.
def traces():
    return [t for _, _, t in LADDER] + [XPU_TRACE]


# One cell, so the rungs are comparable. Pooling every flight ever launched against a rung's cadence
# trace mixes experiments -- courses a and c, densities 0.2/0.3/0.4, people 1.7 m and 2.4 m, two
# controller gains, people placed across the aisle instead of along it (`walk_cross`), and campaigns
# that replayed cadence only (percep_latency_ms == 0) -- which made `ros_vanilla4` read 86/1068 where
# the matched cell is 5/180.
#
# `walk_cross` is part of the cell even though these flights have walk_speed == 0: the flag changes
# where the people are PLACED, not just whether they move (mdp_obstacles.py -- crossing people are
# pinned to the aisle centre line and span most of its width, patrolling people are scattered across
# it), so a crossing run is a different and harder obstacle field, not a replicate of this one.
CELL = dict(course="a", density=0.30, person_h=2.4, gain=0.0055, walk_speed=0.0, walk_cross=0)
SPEEDS = (1.0, 1.2, 1.4, 1.6, 1.8)


def _num(v, default=0.0):
    """A CSV field as a float. Comparisons here are numeric, so a campaign that writes 0.300 or
    1.00 cannot silently drop out of the cell the way a string compare would let it."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def in_cell(r):
    return ((r.get("course") or "a") == CELL["course"]
            and abs(_num(r.get("prop_density"), 0.3) - CELL["density"]) < 1e-6
            and abs(_num(r.get("person_h"), 2.4) - CELL["person_h"]) < 1e-6
            and abs(_num(r.get("moment_scale")) - CELL["gain"]) < 2e-5
            and _num(r.get("walk_speed")) == CELL["walk_speed"]
            and int(_num(r.get("walk_cross"))) == CELL["walk_cross"]
            and _num(r.get("percep_latency_ms")) > 0      # the arm's measured latency, not cadence only
            and any(abs(_num(r.get("cruise_speed")) - s) < 1e-6 for s in SPEEDS))


def matched_cells(traces_=None):
    """Per cadence trace, the flight rows inside CELL at the (cruise, seed) cells EVERY flown arm shares.

    Restricting to CELL makes the rungs the same experiment; intersecting the cells then makes them
    the same flights, so a rung cannot look better for having been flown at easier speeds or seeds.
    An arm with no flights in the cell is left out rather than shrinking the intersection to nothing.

    This is the single definition behind both the table (`flight_tally`) and the drawn column in
    scripts/ros_ladder_figure.py, which imports it -- they must not disagree in the paper.
    Returns (rows per trace, the shared cruise speeds, the shared seeds, the CSVs read).
    """
    traces_ = list(traces_ if traces_ is not None else traces())
    per = collections.defaultdict(lambda: collections.defaultdict(list))   # trace -> (cruise, seed) -> rows
    files = []
    for f in sorted(glob.glob(os.path.join(RES, "campaign*", "campaign*.csv"))):
        files.append(f)
        for r in flight_rows(f):
            t = (r.get("ctrl_trace") or "").split("/")[-1]
            if t in traces_ and in_cell(r):
                per[t][(round(_num(r["cruise_speed"]), 2), int(_num(r["seed"])))].append(r)
    flown = [t for t in traces_ if per.get(t)]
    common = set.intersection(*(set(per[t]) for t in flown)) if flown else set()
    # every replicate of a shared cell counts: the same cell flown in two campaigns is two flights
    cells = {t: [r for c in common for r in per[t][c]] for t in flown}
    return cells, sorted({c for c, _ in common}), sorted({s for _, s in common}), files


def flight_tally(traces_=None):
    """(completed, flown) per cadence trace over the matched cell, plus the pooled counts.

    The pooled counts are every flight ever launched against the trace, across every campaign; they
    mix experiments and are returned for provenance only, never for comparing rungs.
    """
    cells, _cru, _seeds, _files = matched_cells(traces_)
    k = collections.Counter(); n = collections.Counter()
    for t, rows in cells.items():
        n[t] = len(rows); k[t] = sum(r["outcome"] == "success" for r in rows)
    kp = collections.Counter(); np_ = collections.Counter()
    for f in glob.glob(os.path.join(RES, "campaign*", "campaign*.csv")):
        for r in flight_rows(f):
            t = (r.get("ctrl_trace") or "").split("/")[-1]
            if t:
                np_[t] += 1; kp[t] += r["outcome"] == "success"
    return k, n, kp, np_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the rungs as JSON")
    a = ap.parse_args()
    arms = MT.derive()["ros_arms"]
    harts = hart_counts(); k, n, kp, np_ = flight_tally()
    rows = []
    print(f"{'rung':<56}{'cores':>6}{'ctrl Hz':>9}{'cam->goal':>11}{'completed':>14}")
    print(f"(flights: course {CELL['course']}, density {CELL['density']:.2f}, people {CELL['person_h']} m "
          f"placed along the aisle (walk_cross=0), gain {CELL['gain']}, the arm's measured latency "
          f"replayed, and only the (cruise, seed) cells every flown arm shares)")
    for key, label, trace in LADDER:
        a_ = arms.get(key)
        if not a_ or not a_.get("gap_mean_pooled"):
            print(f"{label:<56}{'—':>6}{'not measured':>9}"); continue
        h = harts.get(key)
        rec = {"arm": key, "rung": label, "cores": (statistics.median(h) if h else None),
               "ctrl_hz": round(1000 / a_["gap_mean_pooled"], 1),
               "camera_to_goal_ms": round(a_["e2e_goal_med_pooled"], 1),
               "flights_matched_cell": [k.get(trace, 0), n.get(trace, 0)],
               "flights_pooled_all_campaigns": [kp.get(trace, 0), np_.get(trace, 0)],
               "cell": {**CELL, "cruise": list(SPEEDS)}, "ctrl_trace": trace}
        rows.append(rec)
        m = rec["flights_matched_cell"]
        flown = f"{m[0]}/{m[1]}" if m[1] else "not flown"
        print(f"{label:<56}{(rec['cores'] or 0):>6.0f}{rec['ctrl_hz']:>9.1f}"
              f"{rec['camera_to_goal_ms']:>11.1f}{flown:>14}")
    if a.json:
        json.dump({"rungs": rows, "note": "timing from scripts/measured_timing.py (--verify re-derives "
                                          "each row from the board traces); flights from the campaign "
                                          "CSVs through flight_quarantine"},
                  open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
