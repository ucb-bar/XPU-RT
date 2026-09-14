#!/usr/bin/env python3
"""Shared reduction for the in-flight 44-arm plane sweep (plane3/).

One place to answer "what did the plane measure", so the progress figure and the
sanity-check figure cannot disagree with each other. Reads the per-cell
energy2.json scalars mirrored locally by g5grid/fetch_plane3.sh.

The sweep is SEED-MAJOR: it walks all 176 (task, arm) pairs at seed 300, then
again at 301, and so on to 309. That ordering is why a partial sweep is already
usable -- the whole plane is present at low n, rather than part of the plane at
full n. Every number here is therefore provisional in its ERROR BAR, not in its
coverage.
"""
from __future__ import annotations
import json, glob, re, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SRC = HERE.parent / "g5grid" / "plane3_runs"
ARMS_TSV = HERE.parent / "g5grid" / "arms.tsv"
TASKS = [("egg", "eggplant in basket", "widowx, 40 ms grid"),
         ("spoon", "spoon on towel", "widowx, 40 ms grid"),
         ("coke", "pick coke can", "google, 333 ms grid"),
         ("drawer", "close drawer", "google, 333 ms grid")]
# The energy channel. NOT get_qf(): that was gravity + Coriolis and blind to
# contact and stall by construction (ENERGY_AUDIT.md). This is the PD drive
# torque PhysX actually applies, integrated per substep, skipping the 4 substeps
# after each setpoint step so the integral measures work and not command size.
ECH = "t2_drive_arm_sus"


def arm_plane() -> dict[str, tuple[float, float]]:
    """arm name -> (latency_ms, issue_period_ms) as handed to trace_eval2.py."""
    out = {}
    for line in open(ARMS_TSV):
        f = line.split()
        if f:
            out[f[0]] = (float(f[1]), float(f[2]))
    return out


def load() -> dict[tuple[str, str], dict[int, dict]]:
    """(task, arm) -> {seed: energy2.json}. Deduplicated by cell name across hosts."""
    D = collections.defaultdict(dict)
    for f in glob.glob(str(SRC / "*" / "*" / "energy2.json")):
        m = re.match(r"(egg|spoon|coke|drawer)_(g\d+_\d+)_rng(\d+)$", Path(f).parent.name)
        if m:
            D[(m.group(1), m.group(2))][int(m.group(3))] = json.load(open(f))
    return D


def reduce_cell(seeds: dict[int, dict]) -> dict:
    """Per-arm scalars from its seed replicates.

    success  -- mean over seeds of the seed's own 24-episode rate.
    energy   -- mean over seeds of the seed's MEDIAN episode, matching
                fig_metrics3.py. The median resists the one flailing episode that
                would otherwise carry an arm on its own; the outer mean is what
                the CI is taken over.
    mission  -- pooled over SUCCESSFUL episodes only. A failure has no completion
                time, so this is conditioned and biases slow arms DOWNWARD.
    """
    sr = [100 * d["n_success"] / d["n_episodes"] for d in seeds.values()]
    en = [float(np.median([e[ECH] for e in d["episodes"]])) for d in seeds.values()]
    mt = [e["duration_s"] for d in seeds.values() for e in d["episodes"] if e["success"]]
    return dict(n_seeds=len(sr), n_ep=sum(d["n_episodes"] for d in seeds.values()),
                success=float(np.mean(sr)), success_seeds=sr,
                energy=float(np.mean(en)), energy_seeds=en,
                mission=float(np.mean(mt)) if mt else float("nan"), n_success_ep=len(mt))


def table() -> dict[str, dict[str, dict]]:
    """task -> arm -> reduced scalars, for arms that have at least one seed."""
    D = load()
    out = collections.defaultdict(dict)
    for (task, arm), seeds in D.items():
        out[task][arm] = reduce_cell(seeds)
    return dict(out)


if __name__ == "__main__":
    t = table()
    for task, _, _ in TASKS:
        a = t[task]
        ns = [v["n_seeds"] for v in a.values()]
        print(f"{task:7s} arms={len(a):2d}  seeds/arm {min(ns)}-{max(ns)}  "
              f"episodes={sum(v['n_ep'] for v in a.values()):5d}")
