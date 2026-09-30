#!/usr/bin/env python3
"""STRUCTURAL resolution audit of the 39-arm plane, derived from the harness's own code.

finegrain_eval.py builds the schedule as integer TICK indices:
    snap_tick[j]  = ceil(j*PERIOD / TICK_MS)          (line 330)
    apply_tick[j] = ceil((j*PERIOD + LAT) / TICK_MS)  (line 334)
and the robot consumes a result only on its ACTUATION grid (every tick on widowx,
every 9 ticks on google_robot, whose act_ms is 333.3 vs a 37.037 ms tick).

So the rollout is fully determined by, at each actuation, WHICH inference is freshest and
WHEN that inference's observation was snapped. Two arms with the same such sequence are
the SAME EXPERIMENT by construction -- the design cannot resolve them, no matter how far
apart their nominal latencies are on paper.

This is a DESIGN-side statement, not a claim about measured outcomes: the harness is not
run-to-run deterministic (see NONDETERMINISM.md), so structurally-identical arms can and
do produce different success vectors. That divergence is itself useful -- it measures
harness nondeterminism at scale, on arms that are guaranteed to be the same experiment.
"""
import math, collections, json, sys
from pathlib import Path

HERE = Path(__file__).parent
def ct(t, tick): return int(math.ceil(t / tick - 1e-9))

ARMS = []
for ln in open(HERE / "arms.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 4: ARMS.append((p[0], float(p[1]), float(p[2])))

# task -> (tick_ms, actuation stride in ticks, horizon_ms = max_episode_steps * native period)
TASKS = {"coke":  (1000 / 27.0, 9, 80 * (1000 / 3)),
         "egg":   (40.0,        1, 120 * 200.0),
         "spoon": (40.0,        1, 60 * 200.0)}


def groups(task):
    tick, every, horizon = TASKS[task]
    nt = int(round(horizon / tick))
    sig = collections.defaultdict(list)
    for name, lat, per in ARMS:
        snap, app, j = [], [], 0
        while j * per <= horizon:
            snap.append(ct(j * per, tick)); app.append(ct(j * per + lat, tick)); j += 1
        used, k = [], -1
        for a in range(0, nt, every):
            while k + 1 < len(app) and app[k + 1] <= a: k += 1
            used.append(snap[k] if k >= 0 else -1)
        sig[tuple(used)].append(name)
    return list(sig.values())


out = {}
for task in TASKS:
    g = groups(task)
    out[task] = {"n_design_arms": len(ARMS), "n_distinct_experiments": len(g),
                 "collapsing_groups": [v for v in g if len(v) > 1]}
    print(f"\n{task}: {len(ARMS)} design arms -> {len(g)} DISTINCT experiments")
    for v in sorted(x for x in g if len(x) > 1):
        d = {n: l for n, l, _ in ARMS if n in v}
        print("   ", " == ".join(f"{n}[{d[n]:.1f} ms]" for n in v))
json.dump(out, open(HERE / "resolution_audit.json", "w"), indent=1)
print(f"\n[ok] {HERE / 'resolution_audit.json'}")
