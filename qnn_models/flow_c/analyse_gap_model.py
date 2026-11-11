#!/usr/bin/env python3
"""Test whether a tile's cost is a function of the idle time before its dispatch.

The gap calibration showed small accelerator tiles get slower as the idle gap
before each execute grows, while large tiles are flat. If that is the physical
story -- the accelerator's clock/power state decaying while the lane is idle --
then the right cell for a tile is the one measured at the gap the SCHEDULE
actually leaves in front of it, not a single global constant.

That is testable: the sweep traces record, for every dispatch, when the lane
last finished. This measures each probe tile's real median idle gap and checks
whether the calibration curve evaluated there reproduces the in-situ duration.
"""
import csv, glob, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "sweeps", "qrb5165_20260829-200620")
CAL = json.load(open(os.path.join(HERE, "measurements", "calibration_gap_sweep.json")))

TRACE_NCOL = 20
def trace(path):
    rows, on, pending = [], False, ""
    for ln in open(path):
        if "TRACE_BEGIN" in ln: on = True; continue
        if "TRACE_END" in ln: break
        if not on: continue
        ln = ln.rstrip("\n")
        cut = ln.find("[main] ")
        if cut >= 0:
            ln = ln[:cut]
            if not ln: continue
        cand = pending + ln
        if cand.count(",") == TRACE_NCOL - 1: rows.append(cand); pending = ""
        elif cand.count(",") < TRACE_NCOL - 1: pending = cand
        else: rows.append(cand); pending = ""
    if not rows: return []
    return list(csv.DictReader(rows[1:], fieldnames=rows[0].split(",")))

def ms(r, k):
    v = float(r.get(k) or 0)
    return v / 1000.0 if r.get("time_unit") == "us" else v

# --- per (tile, lane): observed idle gap before each dispatch, and duration ---
gaps, durs = {}, {}
for root in (S, os.path.join(S, "addendum_periodic_only")):
    for p in sorted(glob.glob(os.path.join(root, "runs", "*", "rep*", "run.log"))):
        t = trace(p)
        if not t: continue
        by_lane = {}
        for r in sorted(t, key=lambda r: ms(r, "actual_start_cycles")):
            lane = r["backend"].lower()
            s, e = ms(r, "actual_start_cycles"), ms(r, "actual_end_cycles")
            prev_end = by_lane.get(lane)
            if prev_end is not None:
                key = (r["name"], lane)
                gaps.setdefault(key, []).append(max(0.0, s - prev_end))
                durs.setdefault(key, []).append(e - s)
            by_lane[lane] = e

# --- calibration curve per probe ---
curve = {}
for r in CAL["probes"]:
    tile = r["cell"].split("/", 1)[1]
    curve.setdefault((tile, r["lane"]), {})[r["gap_us"]] = r["gap_median_ms"]

def interp(pts, x_us):
    xs = sorted(pts)
    if x_us <= xs[0]: return pts[xs[0]]
    if x_us >= xs[-1]: return pts[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= x_us <= b:
            f = (x_us - a) / (b - a)
            return pts[a] + f * (pts[b] - pts[a])
    return pts[xs[-1]]

print(f"{'tile@lane':<34} {'in-situ':>8} {'median idle':>12} {'curve@idle':>11} "
      f"{'err@idle':>9} {'err@gap0':>9} {'err@10ms':>9}")
e_idle, e_0, e_10 = [], [], []
for key in sorted(curve):
    if key not in gaps: 
        print(f"{key[0]+'@'+key[1]:<34} {'--':>8}  (no in-situ dispatches)")
        continue
    g = st.median(gaps[key]) * 1000.0          # ms -> us
    d = st.median(durs[key])
    pred = interp(curve[key], g)
    p0, p10 = curve[key][min(curve[key])], curve[key][max(curve[key])]
    ei, e0, e1 = abs(pred-d)/d, abs(p0-d)/d, abs(p10-d)/d
    e_idle.append(ei); e_0.append(e0); e_10.append(e1)
    print(f"{key[0]+'@'+key[1]:<34} {d:>8.3f} {g:>10.0f}us {pred:>11.3f} "
          f"{100*ei:>8.1f}% {100*e0:>8.1f}% {100*e1:>8.1f}%")

print(f"\n  median |error| vs in-situ:")
print(f"    cell measured at gap 0        {100*st.median(e_0):6.1f}%")
print(f"    cell measured at gap 10 ms    {100*st.median(e_10):6.1f}%")
print(f"    cell evaluated at the tile's OWN median idle gap   {100*st.median(e_idle):6.1f}%")
