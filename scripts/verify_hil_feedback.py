#!/usr/bin/env python3
"""The feedback-study figures, re-derived by re-running the producer that drew each one.

`hil_feedback_figure.py` draws, per round, the table CP-SAT solved, the same window as the K1
executed it, and the executed-minus-planned start of every dispatch. Its sidecar records each
round's solver, calibration multiplier, per-network window misses, control-gap mean, camera->control
median and the solver's own predicted miss count -- and nothing recomputed any of them, because the
sidecar records no inputs and the figure names no command.

The spec is the one flag that is not recoverable from the tag: the schedules and traces of round *r*
are found from `fb<tag>r<r>` alone, but the periods and windows a miss is counted against come from
`data/toplevel/<spec>.json`. The pairs below are the ones `docs/Evaluation/measurements_and_ablations.md` 1
states, and pairing them wrongly would count misses against the wrong windows -- so the fact that
each re-run reproduces the committed sidecar exactly is what establishes the pair, not an assumption
behind the check.

    scripts/verify_hil_feedback.py [--stem NAME ...] [-v]

Needs nothing but the repository: every round's schedule, trace and calibration file is tracked.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "results/codesign_feedback/refined")
FAILED = 0

# stem -> (producer, flags). The five study figures differ only in which chain and table they draw;
# the close-up draws one round of the 120 Hz chain twice, isolated against calibrated, from defaults.
FIGURES = {
    "hil_feedback_a90":       ("hil_feedback_figure.py",  ["--tag", "a90",   "--spec", "wh_chain90_solve_500"]),
    "hil_feedback_a90h":      ("hil_feedback_figure.py",  ["--tag", "a90h",  "--spec", "wh_chain90_solve_h200"]),
    "hil_feedback_a120h":     ("hil_feedback_figure.py",  ["--tag", "a120h", "--spec", "wh_chain120_solve_h200"]),
    "hil_feedback_a120e":     ("hil_feedback_figure.py",  ["--tag", "a120e", "--spec", "wh_chain120_solve_h200"]),
    "hil_feedback_b5":        ("hil_feedback_figure.py",  ["--tag", "b5",    "--spec", "wh_chain90_rich_solve_500"]),
    "hil_feedback_close_120": ("hil_feedback_closeup.py", []),
}


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1
    return ok


def close(a, b, tol=1e-6):
    """Equal, comparing numbers by tolerance and walking containers, so a float does not fail on its last bit."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    stems = a.stem or sorted(FIGURES)

    for stem in stems:
        producer, flags = FIGURES[stem]
        side = os.path.join(REF, stem + "_metrics.json")
        print(f"\n=== {stem}  ({producer} {' '.join(flags)})")
        if not check(os.path.exists(side), "the figure's sidecar exists"):
            continue
        have = json.load(open(side))

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, stem + ".png")
            r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", producer)]
                               + flags + ["--out", out],
                               capture_output=True, text=True, cwd=REPO)
            got_p = out.replace(".png", "_metrics.json")
            if not check(r.returncode == 0 and os.path.exists(got_p),
                         "the producer re-runs from the recorded flags"
                         + ("" if r.returncode == 0 else f" (exit {r.returncode}: {r.stderr.strip()[-160:]})")):
                continue
            got = json.load(open(got_p))

        if isinstance(have, dict):                       # the close-up: one record, two executions
            check(got.get("spec") == have.get("spec"), f"the spec re-derives ({have.get('spec')})")
            for side_name in ("isolated", "calibrated"):
                h, g = have.get(side_name) or {}, got.get(side_name) or {}
                check(g.get("tag") == h.get("tag"), f"{side_name}: the round drawn is {h.get('tag')}")
                check(close(g.get("span_ms"), h.get("span_ms"), 1e-6),
                      f"{side_name}: the executed span re-derives ({h.get('span_ms', 0):.1f} ms)")
                check(close(g.get("busy"), h.get("busy"), 1e-9),
                      f"{side_name}: hart occupancy re-derives ({100 * (h.get('busy') or 0):.1f} %)")
            continue

        check(len(got) == len(have), f"the sidecar records one entry per round ({len(have)})")
        check([r.get("round") for r in have] == [r.get("round") for r in got],
              f"the rounds are the ones the study ran, in order "
              f"({', '.join(str(r.get('round')) for r in have)})")
        for h, g in zip(have, got):
            n = h.get("round")
            check(g.get("solver") == h.get("solver"),
                  f"round {n}: the table executed is the {h.get('solver')} one")
            check(close(g.get("calibration_aggregate"), h.get("calibration_aggregate"), 1e-9),
                  f"round {n}: the calibration multiplier re-derives "
                  f"({h.get('calibration_aggregate') if h.get('calibration_aggregate') is None else round(h['calibration_aggregate'], 4)})")
            hw = h.get("window_misses") or {}
            check(close(g.get("window_misses"), hw, 1e-6),
                  f"round {n}: every network's window misses re-derive "
                  + ", ".join(f"{k} {v['miss']}/{v['n']}" for k, v in sorted(hw.items())))
            check(close(g.get("gap_mean_ms"), h.get("gap_mean_ms"), 1e-6),
                  f"round {n}: the control-gap mean re-derives ({(h.get('gap_mean_ms') or 0):.3f} ms)")
            check(close(g.get("e2e_median_ms"), h.get("e2e_median_ms"), 1e-6),
                  f"round {n}: the camera→control median re-derives ({(h.get('e2e_median_ms') or 0):.2f} ms)")
            check(g.get("predicted_misses") == h.get("predicted_misses"),
                  f"round {n}: the solver's own predicted miss count re-derives "
                  f"({h.get('predicted_misses')})")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
