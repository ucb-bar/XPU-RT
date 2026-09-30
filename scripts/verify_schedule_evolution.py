#!/usr/bin/env python3
"""Paper Figure 4 (`fig_schedule_evolution`), re-derived by re-running the composer.

Every other figure in this tree is checked by re-deriving its drawn numbers from the files the render
read. This one had no check at all: its sidecar records four panels' makespans and deadline-miss
counts but not which schedule each panel came from, so nothing could trace a number back -- and the
counts it draws are deliberately *not* the schedules' own (`displayed_instance_deadline_misses` counts
the instances inside the drawn window, `deadline_miss_count` counts all of them), so the difference
cannot be read as a defect or a match without re-doing the work.

Rather than restate the composer's deadline arithmetic -- which happens inside the plotting function
and would drift -- this re-runs `compose_schedule_evolution.py` into a temporary directory from the
recorded panel list and compares the sidecar it emits, field by field, with the committed one. That
covers every panel and every number, and it fails if the composer, the panel schedules or the spec
change under the committed figure.

Two sibling renders, `refined/schedule_evolution_{short,tall}`, are the same four panels at another
geometry and record the same numbers, so the one re-run is compared against each of them.

    scripts/verify_schedule_evolution.py [-v]

Needs nothing but the repository: the spec, the panel list and all four panel schedules are tracked.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/codesign_feedback")
# the paper's render, then the two sibling geometries under refined/ that draw the same panels
FIGS = [os.path.join(RES, "schedule_evolution_mega"),
        os.path.join(RES, "refined/schedule_evolution_short"),
        os.path.join(RES, "refined/schedule_evolution_tall")]
PANELS = os.path.join(RES, "sensor_evo/panels.json")
SPEC = "data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json"
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
    a = ap.parse_args()

    print("=== the inputs the figure names")
    if not check(os.path.exists(PANELS), f"the panel list {os.path.relpath(PANELS, REPO)} exists"):
        print(f"\n{FAILED} failed"); return 1
    panels = json.load(open(PANELS))
    check(tracked(os.path.relpath(PANELS, REPO)), f"the panel list is tracked ({len(panels)} panels)")
    check(os.path.exists(os.path.join(REPO, SPEC)) and tracked(SPEC),
          f"the workload spec is tracked ({os.path.basename(SPEC)})")
    for e in panels:
        p = e.split("|")[2]
        m = p.replace(".json", "_metrics.json")
        check(os.path.exists(os.path.join(REPO, p)) and tracked(p),
              f"panel schedule {os.path.basename(p)} is tracked")
        check(os.path.exists(os.path.join(REPO, m)) and tracked(m),
              f"panel metrics {os.path.basename(m)} is tracked")

    print("\n=== the committed sidecars")
    sides = {}
    for fig in FIGS:
        name, side = os.path.basename(fig), fig + "_metrics.json"
        if check(os.path.exists(side), f"{name}: the figure's sidecar exists"):
            sides[name] = json.load(open(side))
    if not sides:
        print(f"\n{FAILED} failed"); return 1
    for name, h in sides.items():
        check(len(h.get("panels", [])) == len(panels),
              f"{name}: the sidecar records one entry per panel ({len(h.get('panels', []))})")
    have = sides[os.path.basename(FIGS[0])]
    # The composer's geometry is a flag, not a measurement, and the field naming it has changed
    # (authored_figure_size_in -> figure_size_in + height_arg). Recover the height from whichever the
    # sidecar carries, so the re-run reproduces the committed figure rather than the default one.
    size = have.get("figure_size_in") or have.get("authored_figure_size_in")
    height = have.get("height_arg") or (size[1] if size and len(size) > 1 else None)
    check(height is not None, f"the figure's height is recoverable from the sidecar ({height} in)")

    print("\n=== re-running the composer and comparing what it emits")
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "evo")
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts/compose_schedule_evolution.py"),
                            "--spec", SPEC, "--panels-json", PANELS, "--layout",
                            have.get("layout", "grid")]
                           + (["--height", str(height)] if height else [])
                           + ["--out", out],
                           capture_output=True, text=True, cwd=REPO)
        if not check(r.returncode == 0 and os.path.exists(out + "_metrics.json"),
                     "the composer re-runs from the recorded panel list"
                     + ("" if r.returncode == 0 else f" (exit {r.returncode}: {r.stderr.strip()[-200:]})")):
            print(f"\n{FAILED} failed"); return 1
        got = json.load(open(out + "_metrics.json"))

    g_size = got.get("figure_size_in") or got.get("authored_figure_size_in")
    for name, h in sides.items():
        h_size = h.get("figure_size_in") or h.get("authored_figure_size_in")
        check(got.get("layout") == h.get("layout"), f"{name}: layout re-derives ({h.get('layout')})")
        check(g_size == h_size, f"{name}: figure size re-derives at that height ({h_size})")
        for i, (hp, g) in enumerate(zip(h.get("panels", []), got.get("panels", [])), 1):
            for key in ("stage", "title", "driver", "cost_source"):
                check(g.get(key) == hp.get(key), f"{name}: panel {i} {key}: {str(hp.get(key))[:44]!r}")
            hm, gm = hp.get("makespan_ms"), g.get("makespan_ms")
            check(hm is not None and gm is not None and abs(float(hm) - float(gm)) < 1e-6,
                  f"{name}: panel {i} makespan {hm} ms re-derives from "
                  f"{os.path.basename(panels[i-1].split('|')[2])}")
            check(g.get("displayed_instance_deadline_misses") == hp.get("displayed_instance_deadline_misses"),
                  f"{name}: panel {i} drawn deadline misses re-derive "
                  f"({hp.get('displayed_instance_deadline_misses')} inside the drawn window)")
            check(g.get("missed_instances_by_network") == hp.get("missed_instances_by_network"),
                  f"{name}: panel {i} which networks miss re-derives "
                  f"({hp.get('missed_instances_by_network') or 'none'})")

    print(f"\n{FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
