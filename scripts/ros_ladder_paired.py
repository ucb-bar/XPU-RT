#!/usr/bin/env python3
"""Each ROS 2 rung against XPU-RT over the SAME flights: shared cells, equal replicates per cell.

`scripts/ros_effort_ladder.py` restricts every rung to one scene cell (CELL) and to the (cruise,
seed) cells every flown arm shares, but it keeps every replicate a cell happened to receive, so the
rungs' denominators differ (60 to 300) and XPU-RT's 44/300 is not counted over the flights a rung
flew. This compares each rung with XPU-RT on that rung's own cells, taking min(n_ros, n_xpu)
replicates of each, in campaign order, so both arms are counted over identical conditions.

The cell definition (CELL, in_cell), the quarantine filter (flight_rows) and the rung list (LADDER)
are imported from ros_effort_ladder.py, never restated. A second XPU-RT column weights every
XPU-RT replicate of a cell equally rather than taking the first k, so the choice of which k
replicates to keep can be seen not to drive the result.

    scripts/ros_ladder_paired.py [--json results/codesign_feedback/ros_ladder_paired.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ros_effort_ladder as LAD  # noqa: E402
from flight_quarantine import flight_rows  # noqa: E402

XPU = "xpu_a_cpsat_hard.csv"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], summing tables no more likely than the observed."""
    from math import comb
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def cells_by_trace():
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    files = sorted(glob.glob(os.path.join(LAD.RES, "campaign*", "campaign*.csv")))
    for f in files:
        for r in flight_rows(f):
            t = (r.get("ctrl_trace") or "").split("/")[-1]
            if LAD.in_cell(r):
                per[t][(round(LAD._num(r["cruise_speed"]), 2), int(LAD._num(r["seed"])))].append(r)
    return per, files


def compare():
    per, files = cells_by_trace()
    out = []
    for key, label, trace in LAD.LADDER:
        if trace not in per:
            continue
        common = sorted(set(per[trace]) & set(per[XPU]))
        kr = kx = n = 0
        wx = 0.0
        for c in common:
            rr, xx = per[trace][c], per[XPU][c]
            k = min(len(rr), len(xx))
            n += k
            kr += sum(r["outcome"] == "success" for r in rr[:k])
            kx += sum(r["outcome"] == "success" for r in xx[:k])
            wx += k * sum(r["outcome"] == "success" for r in xx) / len(xx)
        if n == 0:
            continue
        out.append({"arm": key, "rung": label, "ros_trace": trace, "cells": len(common), "flights_per_arm": n,
                    "ros_completed": kr, "xpu_completed": kx, "xpu_completed_weighted": round(wx, 2),
                    "ros_wilson95": wilson(kr, n), "xpu_wilson95": wilson(kx, n),
                    "fisher_p_two_sided": round(fisher_two_sided(kr, n - kr, kx, n - kx), 4)})
    return out, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows, files = compare()
    print(f"{'rung':46s} {'cells':>5s}  {'ROS 2':>12s}  {'XPU-RT':>12s}  {'XPU-RT wtd':>10s}  p (2-sided)")
    for r in rows:
        n = r["flights_per_arm"]
        print(f"{r['rung'][:46]:46s} {r['cells']:5d}  {r['ros_completed']:3d}/{n:<4d}{100 * r['ros_completed'] / n:4.1f}%"
              f"  {r['xpu_completed']:3d}/{n:<4d}{100 * r['xpu_completed'] / n:4.1f}%  {r['xpu_completed_weighted']:9.1f}"
              f"  {r['fisher_p_two_sided']:.3f}")
    if a.json:
        json.dump({"generated_by": "scripts/ros_ladder_paired.py", "xpu_trace": XPU, "cell": LAD.CELL,
                   "rule": "per rung: its (cruise, seed) cells shared with XPU-RT, min(n_ros, n_xpu) replicates each, campaign order",
                   "inputs": [os.path.relpath(f, LAD.REPO) if hasattr(LAD, "REPO") else f for f in files],
                   "rows": rows}, open(a.json, "w"), indent=1)
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
