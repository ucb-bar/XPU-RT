#!/usr/bin/env python3
"""Re-derive every number a showdown_improved_<variant> sidecar records, from the CSVs and traces.

The producer (scripts/showdown_improved_figure.py) pairs flights through showdown_v3_figure's
load_campaigns / flight_cells / equalise. This check does not call those: it reads each campaign CSV
through flight_quarantine.flight_rows itself, applies the same cell definition (course a, the row's
prop density, people height and gain, no walking or crossing people, latency replayed), keeps the
(cruise, seed) cells every arm flew, keeps the first k flights of each arm per cell (k = the fewer any
arm flew, in campaign-file then row order), and recounts. Wilson and Newcombe intervals are computed
here from their formulas; the gate-difference bootstrap is figure_candidates.boot_gate_diff on the
re-derived pairs. Board numbers come from measured_timing.derive(), the replay registry and the
measured-Gantt sidecars; the display pair from its figure_data.npz and the scene census episodes.

    scripts/verify_showdown_improved.py [--metrics <sidecar> ...] [-v]

With no --metrics, every refined/showdown_improved_*_metrics.json is checked. Exit 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts"))
from flight_quarantine import flight_rows   # noqa: E402
from figure_candidates import boot_gate_diff   # noqa: E402
import figure_constants as FC   # noqa: E402
import measured_timing as MT   # noqa: E402

FAILS = 0
PASSES = 0
VERBOSE = False


def check(ok, msg):
    global FAILS, PASSES
    if ok:
        PASSES += 1
        if VERBOSE:
            print("PASS  " + msg)
    else:
        FAILS += 1
        print("FAIL  " + msg)
    return ok


def close(a, b, tol=1e-3):
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2); d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def gates(r):
    return 4.0 if r["outcome"] == "success" else float(r.get("gates_passed") or 0)


# ------------------------------------------------------------------------------------------ flights
_ROWS = {}


def campaign_rows(camp):
    if camp not in _ROWS:
        p = os.path.join(RES, camp, "campaign.csv")
        _ROWS[camp] = flight_rows(p) if os.path.exists(p) else []
    return _ROWS[camp]


def all_campaigns():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RES, "campaign_*", "campaign.csv")))


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def in_cell(r, dens, gain, person_h):
    return ((r.get("course") or "a") == "a" and abs(_f(r.get("prop_density"), 0.3) - dens) < 1e-6
            and abs(_f(r.get("moment_scale")) - gain) <= 2e-5 and _f(r.get("walk_speed")) == 0.0
            and int(_f(r.get("walk_cross"))) == 0 and abs(_f(r.get("person_h"), 2.4) - person_h) < 1e-6
            and _f(r.get("percep_latency_ms")) > 0)


def matched(traces, camps, gain, person_h=2.4, dens=0.30, censor=False):
    """{trace: {(cruise, seed): [rows]}} on the cells every trace flew, first k per cell."""
    per = {t: collections.defaultdict(list) for t in traces}
    for camp in sorted(camps):
        for r in campaign_rows(camp):
            t = os.path.basename(r.get("ctrl_trace") or "")
            if t in per and in_cell(r, dens, gain, person_h):
                if censor and r["outcome"] == "timeout":
                    continue
                per[t][(round(_f(r["cruise_speed"]), 2), int(_f(r["seed"])))].append(r)
    common = set.intersection(*(set(per[t]) for t in traces)) if all(per[t] for t in traces) else set()
    out = {t: {} for t in traces}
    for c in sorted(common):
        k = min(len(per[t][c]) for t in traces)
        for t in traces:
            out[t][c] = per[t][c][:k]
    return out, common


# ------------------------------------------------------------------------------------------ checks
def check_census(side):
    recs = side.get("census") or []
    check(len(recs) > 0, f"the sidecar records {len(recs)} paired censuses")
    rule = side.get("pairing_rule", {}).get("gate_diff_bootstrap", {})
    for rec in recs:
        tag = f"census {rec['camera_hz']} Hz {rec['ros_trace']} vs {rec['xpu_trace']}"
        xt, rt = rec["xpu_trace"], rec["ros_trace"]
        m, common = matched((xt, rt), rec["campaigns"], rec["gain"])
        check(len(common) == rec["cells"], f"{tag}: {rec['cells']} shared (cruise, seed) cells re-derive ({len(common)})")
        X = [r for c in sorted(m[xt]) for r in m[xt][c]]; R = [r for c in sorted(m[rt]) for r in m[rt][c]]
        check(len(X) == len(R) == rec["flights_per_arm"], f"{tag}: the same {rec['flights_per_arm']} flights per arm ({len(X)} / {len(R)})")
        kx = sum(r["outcome"] == "success" for r in X); kr = sum(r["outcome"] == "success" for r in R)
        check(rec["xpu_completed"] == [kx, len(X)], f"{tag}: XPU-RT completed {rec['xpu_completed']} re-derives ({kx}/{len(X)})")
        check(rec["ros_completed"] == [kr, len(R)], f"{tag}: ROS 2 completed {rec['ros_completed']} re-derives ({kr}/{len(R)})")
        check(close(rec["xpu_wilson95"], [round(v, 4) for v in wilson(kx, len(X))[1:]], 2e-4), f"{tag}: XPU-RT Wilson interval {rec['xpu_wilson95']}")
        check(close(rec["ros_wilson95"], [round(v, 4) for v in wilson(kr, len(R))[1:]], 2e-4), f"{tag}: ROS 2 Wilson interval {rec['ros_wilson95']}")
        check(rec["xpu_timeouts"] == sum(r["outcome"] == "timeout" for r in X), f"{tag}: XPU-RT timeouts {rec['xpu_timeouts']}")
        check(rec["ros_timeouts"] == sum(r["outcome"] == "timeout" for r in R), f"{tag}: ROS 2 timeouts {rec['ros_timeouts']}")
        pairs = [(gates(a), gates(b)) for c in sorted(common) for a, b in zip(m[xt][c], m[rt][c])]
        check(close(rec["xpu_mean_gates"], round(float(np.mean([a for a, _ in pairs])), 3)), f"{tag}: XPU-RT mean gates {rec['xpu_mean_gates']}")
        check(close(rec["ros_mean_gates"], round(float(np.mean([b for _, b in pairs])), 3)), f"{tag}: ROS 2 mean gates {rec['ros_mean_gates']}")
        d, lo, hi = boot_gate_diff(pairs, n=int(rule.get("resamples", 4000)), seed=int(rule.get("seed", 11)))
        check(close(rec["gate_diff"], [round(d, 3), round(lo, 3), round(hi, 3)]), f"{tag}: gate difference {rec['gate_diff']} (bootstrap)")
        pf = {"xpu_more": sum(a > b for a, b in pairs), "tie": sum(a == b for a, b in pairs), "xpu_fewer": sum(a < b for a, b in pairs)}
        check(rec["per_flight"] == pf, f"{tag}: per-flight gate comparison {rec['per_flight']}")
        cd = newcombe(kx, len(X), kr, len(R))
        check(close(rec["completion_diff_newcombe95"], [round(v, 4) for v in cd], 2e-4), f"{tag}: completion difference Newcombe {rec['completion_diff_newcombe95']}")
        want = "XPU-RT ahead" if lo > 0 else ("ROS 2 ahead" if hi < 0 else "tie")
        check(rec["verdict"] == want, f"{tag}: verdict '{rec['verdict']}' follows from the gate interval ({want})")
        check(close(rec["ros_ctrl_hz"], round(1000.0 / FC.ctrl_gap_ms(rt), 1), 0.05), f"{tag}: ROS 2 control {rec['ros_ctrl_hz']} Hz from the board registry")
        check(close(rec["xpu_ctrl_hz"], round(1000.0 / FC.ctrl_gap_ms(xt), 1), 0.05), f"{tag}: XPU-RT control {rec['xpu_ctrl_hz']} Hz from the board registry")
        a = FC.arm_for(rt)
        check(a.cam_hz == rec["camera_hz"], f"{tag}: the ROS 2 board run was at a {rec['camera_hz']} Hz camera ({a.cam_hz})")
        check(FC.arm_for(xt).cam_hz == rec["camera_hz"], f"{tag}: the XPU-RT board run was at the same camera rate ({FC.arm_for(xt).cam_hz})")
    return recs


def check_headline(side, recs):
    H = side.get("headline")
    if not H:
        return
    v = {k: sum(r["verdict"] == k for r in recs) for k in ("XPU-RT ahead", "tie", "ROS 2 ahead")}
    check(H.get("verdicts") == v, f"headline verdict counts {H.get('verdicts')} re-count from the censuses ({v})")
    if "ros_completes_more" in H:
        more = [[r["camera_hz"], r["ros_trace"], r["xpu_trace"]] for r in recs if r["ros_completed"][0] > r["xpu_completed"][0]]
        check(H["ros_completes_more"] == more, f"headline: the censuses where ROS 2 completes more ({len(more)}) re-derive")
    if H.get("text"):
        t1 = H["text"][0]
        check(f"ahead in {v['XPU-RT ahead']} of {len(recs)}" in t1 and f"level in {v['tie']}" in t1 and f"behind in {v['ROS 2 ahead']}" in t1,
              "headline text states the re-counted verdicts")


def check_rates(side):
    C = side.get("control_rate") or {}
    ra = MT.derive()["ros_arms"]
    for arm, rec in C.items():
        if arm == "xpu":
            for h, runs in rec.get("runs", {}).items():
                for key, hz in runs:
                    g = MT.SOLVER_ARMS.get(key, {}).get("ctrl_gap_mean_ms")
                    check(g is not None and close(hz, round(1000.0 / g, 2), 0.01), f"panel C: XPU-RT {key} at {h} Hz camera controls at {hz} Hz")
            continue
        want = sorted((int(k.split("@")[1]), round(1000.0 / v["gap_mean_pooled"], 2)) for k, v in ra.items()
                      if k.split("@")[0] == arm and v.get("gap_mean_pooled"))
        got = [tuple(p) for p in rec["points"]]
        check(close(got, want, 0.01), f"panel C: ROS 2 {arm} rate curve ({len(got)} points) re-derives from the board runs")


def check_display(side):
    A = side.get("A")
    if not A:
        return
    X = np.load(os.path.join(REPO, A["xpu_dir"], "figure_data.npz"), allow_pickle=True)
    R = np.load(os.path.join(REPO, A["ros_dir"], "figure_data.npz"), allow_pickle=True)
    check(A["episode_seed"] == {"xpu": int(X["seed"]), "ros": int(R["seed"])}, f"panel A: episode seeds {A['episode_seed']}")
    check(A["layout_seed"] == {"xpu": int(X["layout_seed"]), "ros": int(R["layout_seed"])} and A["layout_seed"]["xpu"] == A["layout_seed"]["ros"],
          f"panel A: both flights on one layout ({A['layout_seed']})")
    check(A["xpu_gates"] == int(X["gates_passed"]) and str(X["outcome"]) == "success", f"panel A: XPU-RT flight completes, {A['xpu_gates']} gates")
    check(A["ros_gates"] == int(R["gates_passed"]) and A["ros_outcome"] == str(R["outcome"]), f"panel A: ROS 2 flight {A['ros_outcome']} after {A['ros_gates']} gates")
    check(close(A["xpu_eff_hz"], round(float(X["eff_cmd_hz"]), 1), 0.05) and close(A["ros_eff_hz"], round(float(R["eff_cmd_hz"]), 1), 0.05),
          f"panel A: in-flight control {A['xpu_eff_hz']} / {A['ros_eff_hz']} Hz")
    check(os.path.basename(str(X["ctrl_trace"])) == A["xpu_trace"] and os.path.basename(str(R["ctrl_trace"])) == A["ros_trace"],
          f"panel A: the flights replay {A['xpu_trace']} and {A['ros_trace']}")
    check(close(float(X["cruise_speed"]), A["display_cruise"]) and close(float(R["cruise_speed"]), A["display_cruise"]), f"panel A: cruise {A['display_cruise']} m/s")
    for arm, sub in (("xpu", os.path.splitext(A["xpu_trace"])[0]), ("ros", os.path.splitext(A["ros_trace"])[0])):
        eps = sorted(glob.glob(os.path.join(REPO, A["scene_dir"], sub, "ep*.npz")))
        runs = []
        for f in eps:
            z = np.load(f, allow_pickle=True)
            runs.append({"seed": int(z["seed"]), "outcome": str(z["outcome"]), "gates": int(z["gates_passed"])})
            check(int(z["layout_seed"]) == A["layout_seed"][arm] and os.path.basename(str(z["ctrl_trace"])) == A[f"{arm}_trace"],
                  f"panel A: scene run {os.path.basename(f)} ({arm}) is the display layout and arm")
        check(A["scene_runs"][arm] == runs, f"panel A: the {len(runs)} {arm} scene runs drawn re-derive")
        k = sum(r["outcome"] == "success" for r in runs)
        check(A["scene_completed"][arm] == [k, len(runs)], f"panel A: {arm} scene tally {A['scene_completed'][arm]}")
        mg = round(float(np.mean([4 if r["outcome"] == "success" else r["gates"] for r in runs])), 3)
        check(close(A["scene_mean_gates"][arm], mg), f"panel A: {arm} scene mean gates {A['scene_mean_gates'][arm]}")
        # the census cell is the same one the scene campaign table records
        rows = [r for r in flight_rows(os.path.join(REPO, A["scene_dir"], "campaign.csv")) if os.path.basename(r.get("ctrl_trace", "")) == A[f"{arm}_trace"]]
        kk = sum(r["outcome"] == "success" for r in rows)
        check([kk, len(rows)] == [k, len(runs)], f"panel A: {arm} scene tally agrees with the scene's campaign.csv ({kk}/{len(rows)})")


def check_gantt(side):
    I = side.get("I")
    if not I:
        return
    pre = os.path.join(RES, "refined", "navshard36", "measured_gantt_navshard36")
    for row in ("xpu", "ros"):
        m = json.load(open(f"{pre}_{row}_metrics.json"))
        for k in ("chain_ms_median", "ctrl_gap_mean_ms", "frames_late", "frames_checked"):
            check(close(I[row].get(k), m.get(k)), f"panel I: {row} {k} = {I[row].get(k)} (Gantt sidecar)")
        for k in ("source", "cpu_source", "manifest"):
            p = os.path.join(REPO, m[k])
            check(os.path.exists(p) and sha(p) == m[f"{k}_sha256"], f"panel I: {row} board {k} is the traced file ({os.path.basename(m[k])})")


def check_envelope_D(side):
    D = side.get("D")
    if not D:
        return
    cell = {}
    for r in flight_rows(os.path.join(RES, "hil_ablation.csv")):
        h = round(float(r["eff_cmd_hz"])); k, n = cell.get(h, (0, 0)); cell[h] = (k + (r["outcome"] == "success"), n + 1)
    want = {str(h): list(cell[h]) for h in sorted(cell)}
    check(D.get("k_n_per_hz") == want, f"panel D: rate-injected k/n per control rate re-derive ({want})")
    check(D.get("flights") == sum(1 for _ in open(os.path.join(RES, "hil_ablation.csv"))) - 1, f"panel D: {D.get('flights')} flights in hil_ablation.csv")
    A = side.get("A") or {}
    check(close(D.get("xpu_hz"), A.get("xpu_eff_hz"), 0.06) and close(D.get("ros_hz"), A.get("ros_eff_hz"), 0.06),
          "panel D marks the display arms' in-flight rates")


def check_envelope_variant(side):
    E = side.get("envelope")
    if not E:
        return
    for key, rec in E.items():
        arms = rec["arms"]
        if key == "std45":
            m, common = matched(arms, all_campaigns(), 0.0055, 2.4)
        else:
            m, common = matched(arms, ("campaign_break",), 0.0055, 1.7, censor=True)
        check(len(common) == rec["cells"], f"envelope {key}: {rec['cells']} shared cells re-derive ({len(common)})")
        for t in arms:
            by = collections.defaultdict(list)
            for (c, s), v in m[t].items():
                by[c] += v
            pts = []
            for c in sorted(by):
                v = by[c]; k = sum(r["outcome"] == "success" for r in v); _, lo, hi = wilson(k, len(v))
                pts.append({"cruise": c, "completed": k, "flown": len(v), "wilson95": [round(lo, 4), round(hi, 4)],
                            "mean_gates": round(float(np.mean([gates(r) for r in v])), 3)})
            got = rec["series"][t]["points"]
            ok = len(got) == len(pts) and all(g["cruise"] == p["cruise"] and g["completed"] == p["completed"] and g["flown"] == p["flown"]
                                              and close(g["wilson95"], p["wilson95"], 2e-4) and close(g["mean_gates"], p["mean_gates"]) for g, p in zip(got, pts))
            check(ok, f"envelope {key}: {t} per-speed completion re-derives")
            tk = sum(p["completed"] for p in pts); tn = sum(p["flown"] for p in pts)
            check(rec["series"][t]["completed"] == [tk, tn], f"envelope {key}: {t} total {rec['series'][t]['completed']}")
    # the two-arm 1.7 m pairing is the one the final figure's breaking-point panel records
    fin = os.path.join(RES, "refined", "warehouse_showdown_final_metrics.json")
    if "tall17_two_arm" in E and os.path.exists(fin):
        ib = json.load(open(fin)).get("I_break", {})
        for lab, tr in (("XPU-RT · CP-SAT", "xpu_a_cpsat_hard.csv"), ("ROS 2 vanilla (4-hart YOLO)", "ros_vanilla445.csv")):
            want = [sum(v[0] for v in ib.get(lab, {}).values()), sum(v[1] for v in ib.get(lab, {}).values())]
            check(E["tall17_two_arm"]["series"][tr]["completed"] == want, f"envelope tall17_two_arm: {tr} agrees with the final figure's I_break ({want})")


def check_inputs(side):
    inp = side.get("inputs") or {}
    stale = [p for p, h in inp.items() if not os.path.exists(os.path.join(REPO, p)) or sha(os.path.join(REPO, p)) != h]
    check(not stale, f"the {len(inp)} inputs the render read are unchanged" + (f" ({len(stale)} moved, e.g. {stale[0]})" if stale else ""))
    check(not side.get("fallbacks_used"), "the render used no fallback")
    for camp in {c for r in side.get("census", []) for c in r["campaigns"]}:
        check(f"results/codesign_feedback/{camp}/campaign.csv" in inp, f"{camp}/campaign.csv is among the hashed inputs")
    fig = os.path.join(REPO, side.get("figure", ""))
    check(os.path.exists(fig) and os.path.exists(fig[:-4] + ".pdf"), f"the figure exists ({side.get('figure')} + .pdf)")


def verify(path):
    side = json.load(open(path))
    print(f"=== {os.path.relpath(path, REPO)}  ({side.get('variant')})")
    recs = check_census(side)
    check_headline(side, recs)
    check_rates(side)
    check_display(side)
    check_gantt(side)
    check_envelope_D(side)
    check_envelope_variant(side)
    check_inputs(side)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(); VERBOSE = a.verbose
    paths = a.metrics or sorted(glob.glob(os.path.join(RES, "refined", "showdown_improved_*_metrics.json")))
    if not paths:
        print("no showdown_improved sidecars"); return 1
    for p in paths:
        verify(p)
    print(f"{PASSES} PASS, {FAILS} FAIL")
    print(f"{FAILS} FAIL")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
