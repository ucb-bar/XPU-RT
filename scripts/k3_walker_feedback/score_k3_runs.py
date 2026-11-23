#!/usr/bin/env python3
"""Score K3 walker runs: deadline misses and worst lateness per network, the
median actual-minus-predicted finish, and (with --modelblaster) a bit-exact
check of every network's output against its golden.

    score_k3_runs.py RUN_DIR [RUN_DIR ...] [--spec SPEC.json] [--modelblaster MB_ROOT]

A run directory holds what run_schedule_on_k3.sh copies: the schedule JSON, the
walker's *_stdout.txt and *_trace.csv. Misses use each instance's release
(instance x period) plus its window from the spec; a miss is an instance whose
last dispatch ends after that deadline.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xpu-rt"))
import walker_sim as W  # noqa: E402

DEFAULT_SPEC = ROOT / "data/toplevel/networks_k3_mlp1600_dronet480_yolo64_singletons.json"
NETS = ("mlp_control", "dronet", "yolov8_nano")


def score(run: Path, windows, periods, mb_root: Path | None) -> dict:
    sched = next(run.glob("scheduled_*.json"))
    trace = next(run.glob("*_trace.csv"))
    tab = W.load_table(sched)
    tr = W.read_trace(trace, tab)
    mm = W.instance_misses(tab, [x["f"] for x in tr], windows, periods)
    disc = {n: statistics.median([t["f"] - (e.ps + e.pd) for e, t in zip(tab, tr)
                                  if e.net == n and not t["zero"]]) for n in NETS}
    out = next(run.glob("*_stdout.txt")).read_text()
    mv = re.search(r"dynamic walker .*?moved=(\d+)", out)
    pl = re.search(r"pooled_taken=(\d+)", out)
    res = {"run": run.name, "misses": mm, "median_discrepancy_ms": disc,
           "moved": int(mv.group(1)) if mv else 0, "pooled": int(pl.group(1)) if pl else 0}
    if mb_root:
        sys.path.insert(0, str(mb_root))
        from validation import runner_common as rc  # ModelBlaster
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            res["bit_exact"] = all(
                rc.check_one(rc.parse_output(out, n),
                             str(mb_root / f"build/k3_xpurt/{n}/int8/io.npz"), 0, 0)[0]
                for n in NETS)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--modelblaster", type=Path, help="ModelBlaster root, for the golden check")
    a = ap.parse_args()
    windows, periods = W.spec_windows(a.spec)
    for run in sorted(a.runs):
        if not any(run.glob("*_trace.csv")):
            continue
        r = score(run, windows, periods, a.modelblaster)
        cols = "  ".join(f"{n} {r['misses'][n]['misses']:3d}/{r['misses'][n]['n']:<3d} "
                         f"worst {r['misses'][n]['worst']:+6.2f} ms" for n in NETS)
        disc = " / ".join(f"{r['median_discrepancy_ms'][n]:+.2f}" for n in NETS)
        extra = f"  moved={r['moved']} pooled={r['pooled']}" if r["moved"] or r["pooled"] else ""
        ok = "" if "bit_exact" not in r else ("  bit-exact" if r["bit_exact"] else "  MISMATCH")
        print(f"{r['run']:32s} {cols}  median actual-predicted {disc} ms{extra}{ok}")


if __name__ == "__main__":
    main()
