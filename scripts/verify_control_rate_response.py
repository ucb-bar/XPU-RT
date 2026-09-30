#!/usr/bin/env python3
"""The control-rate response figures, re-derived from the campaign censuses they read.

`control_rate_response.py` draws one point per (cadence trace, perception hold) configuration that
has at least `MIN_N` flights at the figure's gain, and a highlighted `controlled_sweep` where the
camera rate is the only thing that moves. Its sidecar records every one of those points -- n,
completions, mean gates and the cadence the arm actually replayed -- and this recomputes every one.

Rather than restate the grouping and the quarantine rule, this imports `collect()` from the renderer
and compares what it returns with what the sidecar recorded, point by point. The two therefore cannot
drift: a change to which flights count, or to how a point is formed, moves both sides together and
this check still fails if the committed figure no longer matches.

    scripts/verify_control_rate_response.py [--stem NAME ...] [-v]

A stem whose recorded input hashes no longer match the files on disk cannot re-derive to the numbers
it drew -- the census grew under it. That is reported as a stale input rather than as agreement, and
the point comparison is skipped for that stem, because comparing against a different population
would be a check that cannot fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
sys.path.insert(0, os.path.join(REPO, "scripts"))

STEMS = ["control_rate_response", "control_rate_response_cp3", "control_rate_response_v2"]
FAILED = 0


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1
    return ok


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    stems = a.stem or STEMS

    import control_rate_response as C
    print("=== the population every stem is drawn from")
    pts, seen = C.collect()
    by_key = {(p["trace"], round(p["hold_ms"], 1)): p for p in pts}
    check(len(seen) > 0, f"the campaign censuses load through flight_quarantine ({len(seen)} files)")
    check(len(pts) > 0, f"collect() forms {len(pts)} points at gain {C.GAIN}, min n {C.MIN_N}")

    for stem in stems:
        side = os.path.join(REF, stem + "_metrics.json")
        print(f"\n=== {stem}")
        if not check(os.path.exists(side), "the figure's sidecar exists"):
            continue
        s = json.load(open(side))
        check(not s.get("fallbacks_used"),
              f"the render used no fallback ({s.get('fallbacks_used') or 'none'})")
        check(s.get("gain") == C.GAIN, f"the gain the sidecar records is the renderer's ({s.get('gain')})")
        check(s.get("min_n") == C.MIN_N, f"the minimum n is the renderer's ({s.get('min_n')})")

        stale = []
        for rel, want in (s.get("inputs") or {}).items():
            p = os.path.join(REPO, rel)
            if not os.path.exists(p) or sha(p) != want:
                stale.append(rel)
        if stale:
            check(False, f"the {len(s.get('inputs') or {})} censuses the render read are unchanged "
                         f"({len(stale)} have moved since, e.g. {os.path.basename(os.path.dirname(stale[0]))})")
            print(f"      -- point comparison skipped: the population is no longer the one drawn")
            continue
        check(True, f"the {len(s.get('inputs') or {})} censuses the render read are unchanged")

        miss, off = [], []
        for p in s.get("points", []):
            g = by_key.get((p["trace"], round(p["hold_ms"], 1)))
            if g is None:
                miss.append(p["trace"])
                continue
            for k, tol in (("n", 0), ("completed", 0), ("hz", 0.01), ("mean_gates", 0.0005)):
                if abs(float(g[k]) - float(p[k])) > tol:
                    off.append(f"{p['trace']} {k} drawn {p[k]} re-derives {g[k]}")
            if g["kind"] != p["kind"]:
                off.append(f"{p['trace']} kind drawn {p['kind']} re-derives {g['kind']}")
        check(not miss, f"every drawn point is still a configuration on disk ({len(s.get('points', []))} points"
                        + (f"; absent: {', '.join(miss[:3])}" if miss else "") + ")")
        check(not off, f"every drawn point's n, completions, cadence and mean gates re-derive"
                       + (f" ({len(off)} differ: {off[0]})" if off else
                          f" ({len(s.get('points', []))} points)"))

        # the highlighted sweep is a subset of the same points, plus the camera rate it isolates
        sw = s.get("controlled_sweep", [])
        names = {t for _, arms in C.SWEEPS.values() for t, _, _ in arms}
        check(all(p["trace"] in names for p in sw),
              f"the highlighted sweep names only traces the renderer defines ({len(sw)} rungs)")
        cams = [p.get("camera_hz") for p in sw]
        check(cams == sorted(cams) and all(c is not None for c in cams),
              f"the sweep is ordered by camera rate ({' → '.join(str(c) for c in cams)} Hz)")
        rates = [round(float(p["hz"]), 1) for p in sw]
        check(rates == sorted(rates),
              f"the command rate rises with the camera rate ({' → '.join(f'{r:.1f}' for r in rates)} Hz)")

        # the highlighted sweep repeats entries from points/ with a camera rate attached; nothing
        # compared the two, so the sweep could highlight numbers the panel does not draw
        pts = {(p["trace"], round(p["hold_ms"], 1)): p for p in s.get("points", [])}
        sw_off = []
        for e in sw:
            base = pts.get((e["trace"], round(e["hold_ms"], 1)))
            if base is None:
                sw_off.append(f"{e['trace']} is highlighted but is not a drawn point")
                continue
            for k in ("n", "completed", "hz", "mean_gates"):
                if abs(float(e[k]) - float(base[k])) > 1e-6:
                    sw_off.append(f"{e['trace']} {k}: sweep {e[k]}, point {base[k]}")
        check(not sw_off, f"every highlighted rung repeats the point it highlights ({len(sw)} rungs)"
                          + (f" -- {sw_off[0]}" if sw_off else ""))
        cams = {t: c for _, arms in C.SWEEPS.values() for t, _, c in arms}
        bad_cam = [e["trace"] for e in sw if cams.get(e["trace"]) != e.get("camera_hz")]
        check(not bad_cam, f"each rung's camera rate is the one the renderer defines for that trace"
                           + (f" ({bad_cam[0]} differs)" if bad_cam else ""))

        tot = sum(p["n"] for p in s.get("points", []))
        check(tot == s.get("flights_total"),
              f"flights_total is the sum over the drawn points ({s.get('flights_total')})")
        # the annotated line is drawn through the points, so it is their unweighted mean -- the
        # greedy arm is held out of it, as it is held out of the xpu series the line belongs to
        xs = [p for p in s.get("points", []) if p["kind"] == "xpu"]
        if xs:
            mg = sum(p["mean_gates"] for p in xs) / len(xs)
            check(abs(mg - float(s["xpu_mean_gates"])) < 0.0005,
                  f"the \"XPU-RT holds {float(s['xpu_mean_gates']):.2f}\" line is the mean over the "
                  f"{len(xs)} XPU-RT points ({s['xpu_mean_gates']})")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
