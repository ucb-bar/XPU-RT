#!/usr/bin/env python3
"""The ROS effort ladder, re-derived by re-running the figure that draws it.

Each rung is a ROS 2 deployment given more of the board than the one below it, and the figure's
claim is the column that does not move: more cores, more processes, more pinning, and the control
rate stays where the camera puts it. Its sidecar records every rung's hart count, control rate,
camera->goal latency and flight tally, and nothing recomputed them.

The rungs come from two places that can drift apart -- `measured_timing.derive()` for the board
timing and the campaign censuses through `flight_quarantine` for the flights -- so rather than
restate either, this re-runs `ros_ladder_figure.py` into a temporary directory and compares the
sidecar it emits with the committed one, rung by rung.

The flight column is pooled over `campaign*/campaign*.csv` by glob, so it grows whenever a campaign
directory is added. Hashing the inputs a render recorded cannot see that: every recorded file is
unchanged while the population the figure would draw today is larger. So this compares the input
*set* as well, and a render whose producer now reads files it did not read then is reported as
drawn against a population that no longer exists rather than as agreement.

    scripts/verify_ros_effort_ladder.py [--stem NAME ...] [-v]

Needs nothing but the repository: the board traces and the campaign CSVs are tracked or archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
# The current render is the default. The figure as first drawn is checked with
# `--stem ros_effort_ladder`, and reports what it is: drawn before nine more campaign directories
# landed, so its flight column is pooled over a population that no longer exists.
STEMS = ["ros_effort_ladder_v2"]
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
    print("=== re-running the figure once, for every stem to be compared against")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ladder")
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/ros_ladder_figure.py"),
                            "--out", out], capture_output=True, text=True, cwd=REPO)
        if not check(r.returncode == 0 and os.path.exists(out + "_metrics.json"),
                     "ros_ladder_figure.py re-runs"
                     + ("" if r.returncode == 0 else f" (exit {r.returncode}: {r.stderr.strip()[-200:]})")):
            print(f"\n{FAILED} failed"); return 1
        got = json.load(open(out + "_metrics.json"))

    for stem in stems:
        if not compare(stem, got):
            continue
    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


def compare(stem, got):
    side = os.path.join(REF, stem + "_metrics.json")
    print(f"\n=== {stem}")
    if not check(os.path.exists(side), "the figure's sidecar exists"):
        return False
    have = json.load(open(side))
    check(not have.get("fallbacks_used"),
          f"the render used no fallback ({have.get('fallbacks_used') or 'none'})")

    stale = [rel for rel, want in (have.get("inputs") or {}).items()
             if not os.path.exists(os.path.join(REPO, rel)) or sha(os.path.join(REPO, rel)) != want]
    check(not stale, f"the {len(have.get('inputs') or {})} inputs the render read are unchanged"
                     + (f" ({len(stale)} have moved, e.g. {stale[0]})" if stale else ""))
    added = sorted(set(got.get("inputs") or {}) - set(have.get("inputs") or {}))
    if not check(not added,
                 f"the producer still reads the same censuses it read then "
                 f"({len(got.get('inputs') or {})} now, {len(have.get('inputs') or {})} then"
                 + (f"; added since: {', '.join(os.path.basename(os.path.dirname(x)) for x in added[:3])}"
                    f"{' and %d more' % (len(added) - 3) if len(added) > 3 else ''}" if added else "") + ")"):
        print("      -- rung comparison skipped: the flight column is pooled by glob, so it is "
              "drawn from a population that no longer exists")
        return False

    check(got.get("cell") == have.get("cell"),
          f"the scene cell the flights are counted in re-derives "
          f"(course {(have.get('cell') or {}).get('course')}, gain {(have.get('cell') or {}).get('gain')})")

    hr, gr = have.get("rungs", []), got.get("rungs", [])
    if not check(len(hr) == len(gr), f"the ladder has the same rungs ({len(hr)} drawn, {len(gr)} re-derived)"):
        return False

    for h, g in zip(hr, gr):
        arm = h.get("arm")
        check(g.get("rung") == h.get("rung"), f"{arm}: the rung is {str(h.get('rung'))[:56]!r}")
        check(g.get("ctrl_trace") == h.get("ctrl_trace"),
              f"{arm}: the cadence trace it replays is {h.get('ctrl_trace')}")
        for key, tol, unit in (("control_hz", 0.05, " Hz"), ("camera_to_goal_ms", 0.05, " ms")):
            hv, gv = h.get(key), g.get(key)
            ok = (hv is None and gv is None) or (
                hv is not None and gv is not None and abs(float(hv) - float(gv)) <= tol)
            check(ok, f"{arm}: {key.replace('_', ' ')} re-derives ({hv}{unit if hv is not None else ''})")
        check(g.get("cores_busy_ge20pct") == h.get("cores_busy_ge20pct"),
              f"{arm}: the harts carrying work re-derive ({h.get('cores_busy_ge20pct')})")
        check(g.get("cores_per_replicate") == h.get("cores_per_replicate"),
              f"{arm}: the per-replicate hart counts re-derive ({h.get('cores_per_replicate')})")
        # Only the solved rung records this one -- every hart the schedule touches, beside the
        # stricter >=20%-busy count above -- and nothing compared it until the mutation sweep asked.
        if h.get("cores_carrying_work") is not None or g.get("cores_carrying_work") is not None:
            check(g.get("cores_carrying_work") == h.get("cores_carrying_work"),
                  f"{arm}: the harts the schedule touches at all re-derive "
                  f"({h.get('cores_carrying_work')})")
        check(g.get("latency_source") == h.get("latency_source"),
              f"{arm}: the latency statistic drawn is {h.get('latency_source')}")
        fm = h.get("flights_matched") or {}
        if fm.get("flown"):
            k_, n_ = fm["completed"], fm["flown"]
            if fm.get("fraction") is not None:
                check(abs(k_ / n_ - float(fm["fraction"])) < 5e-4,
                      f"{arm}: the completion fraction follows from {k_}/{n_} ({fm['fraction']})")
            if fm.get("wilson95"):
                import math
                z = 1.96
                ph = k_ / n_
                c = (ph + z * z / (2 * n_)) / (1 + z * z / n_)
                m = z * math.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
                want_ci = [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]
                got_ci = [round(float(x), 4) for x in fm["wilson95"]]
                check(all(abs(a - b) < 1e-3 for a, b in zip(want_ci, got_ci)),
                      f"{arm}: the 95% interval is Wilson's on {k_}/{n_} "
                      f"([{got_ci[0]:.3f}, {got_ci[1]:.3f}])")
        for key in ("flights_matched", "flights_pooled_all_campaigns"):
            hv, gv = h.get(key) or {}, g.get(key) or {}
            same = all(hv.get(k) == gv.get(k) for k in ("completed", "flown"))
            check(same, f"{arm}: {key.replace('_', ' ')} re-derives "
                        f"({hv.get('completed')}/{hv.get('flown')})")

    # No thesis is asserted here beyond the re-derivation above. The ladder's rungs do not share one
    # ordering to check against: four ROS rungs reach 100 Hz -- the two whose control node runs on its
    # own timer and re-sends a held goal, and the two hand-pinned ones, which genuinely hold 100 Hz --
    # so a check of the form "no ROS rung reaches the solved arm's rate" would be false. What the
    # figure argues from those rungs is the caption's business; what is verifiable is that every
    # number drawn on them comes back from the board traces and the censuses, which it does.
    return True


if __name__ == "__main__":
    sys.exit(main())
