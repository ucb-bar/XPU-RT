#!/usr/bin/env python3
"""Paper Figure 1 (`fig_loop_overview`), re-derived by re-running its generator.

Same gap as Figure 4 had, and closed the same way. The figure's sidecar records the ablation cell
counts (A/B/C/D), which levers each solver fired, and each evolution stage's deadline misses -- and
it names the one summary it read. Nothing re-derived any of it.

Rather than restate how a cell count is formed, this re-runs `plot_loop_overview.py` against the
summary the sidecar names, into a temporary directory, and compares the sidecar it emits field by
field. That covers both bands and all four stage panels, and it fails if the generator or the summary
change under the committed figure.

Band A is schematic: the sidecar says so in as many words, and this checks that the statement is
still there rather than pretending the mini-Gantts carry a measurement.

    scripts/verify_loop_overview.py [-v]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDE = os.path.join(REPO, "results/codesign_feedback/loop_overview_metrics.json")
FAILED = 0


def check(ok, msg):
    global FAILED
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        FAILED += 1
    return ok


def tracked(rel):
    return subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel],
                          capture_output=True, text=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.parse_args()

    print("=== the inputs the figure names")
    if not check(os.path.exists(SIDE), "the figure's sidecar exists"):
        print(f"\n{FAILED} failed"); return 1
    have = json.load(open(SIDE))
    used = have.get("summaries_used") or []
    check(bool(used), f"the sidecar names the summaries it read ({len(used)})")
    check(not have.get("summaries_missing"),
          f"no summary it wanted was missing ({have.get('summaries_missing') or 'none'})")
    for rel in used:
        check(os.path.exists(os.path.join(REPO, rel)) and tracked(rel),
              f"{rel} is tracked")

    print("\n=== what the figure says about itself")
    check(bool(have.get("band_a")) and "schematic" in have["band_a"],
          "band A is recorded as schematic, carrying no measured number")
    check(bool(have.get("caveat_cell_C")),
          "cell C's caveat is recorded (it is scored on the costs it re-solves against)")
    check(bool(have.get("cell_meaning")),
          f"the four cells are given meanings ({', '.join(f'{k}={v}' for k, v in sorted((have.get('cell_meaning') or {}).items()))})")

    print("\n=== re-running the generator and comparing what it emits")
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, os.path.join(REPO, "scripts/plot_loop_overview.py"),
               "--out-dir", td, "--stem", "lo"]
        for rel in used:
            cmd += ["--summary", os.path.join(REPO, rel)]
        h = have.get("canvas_mm")
        if h and len(h) > 1:
            cmd += ["--height-mm", str(h[1])]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        out = os.path.join(td, "lo_metrics.json")
        if not check(r.returncode == 0 and os.path.exists(out),
                     "the generator re-runs from the recorded summary"
                     + ("" if r.returncode == 0 else f" (exit {r.returncode}: {r.stderr.strip()[-200:]})")):
            print(f"\n{FAILED} failed"); return 1
        got = json.load(open(out))

    check(got.get("canvas_mm") == have.get("canvas_mm"),
          f"canvas re-derives ({have.get('canvas_mm')} mm)")

    ha, ga = have.get("ablation") or {}, got.get("ablation") or {}
    check(sorted(ha) == sorted(ga), f"the same workload groups are drawn ({', '.join(sorted(ha))})")
    for grp in sorted(set(ha) & set(ga)):
        check(ha[grp].get("n_nets") == ga[grp].get("n_nets"),
              f"band B {grp}: {ha[grp].get('n_nets')} networks")
        for solver in ("cpsat", "greedy"):
            hc = (ha[grp].get(solver) or {}).get("cells")
            gc = (ga[grp].get(solver) or {}).get("cells")
            check(hc == gc, f"band B {grp} {solver}: cell counts re-derive "
                            f"({', '.join(f'{k}={v}' for k, v in sorted((hc or {}).items()))})")
            hl = (ha[grp].get(solver) or {}).get("levers")
            gl = (ga[grp].get(solver) or {}).get("levers")
            check(hl == gl, f"band B {grp} {solver}: levers re-derive ({', '.join(hl or []) or 'none'})")

    he, ge = have.get("evolution") or [], got.get("evolution") or []
    check(len(he) == len(ge), f"band C draws the same number of stages ({len(he)})")
    for h_, g_ in zip(he, ge):
        st = h_.get("stage")
        for key in ("title", "has_shard", "has_ime", "dispatch_widths", "implementations"):
            check(h_.get(key) == g_.get(key), f"band C stage {st} {key}: {str(h_.get(key))[:48]!r}")
        check(h_.get("instance_misses") == g_.get("instance_misses"),
              f"band C stage {st}: {h_.get('instance_misses')} deadline miss(es) re-derive")
        check(h_.get("missed_by_network") == g_.get("missed_by_network"),
              f"band C stage {st}: which networks miss re-derives ({h_.get('missed_by_network') or 'none'})")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
