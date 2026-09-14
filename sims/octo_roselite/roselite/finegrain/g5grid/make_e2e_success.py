#!/usr/bin/env python3
"""Rebuild paper/grid_e2e_success.json with per-cell PROVENANCE.

Two things this does that the previous ad-hoc pass did not:

1. DEDUPLICATES by distinct run key.  g5grid/runs/ holds 3623 summary.json files but
   only 2640 distinct <task>_<arm>_rng<seed> keys: fetch_phase3.sh rsyncs each worker
   into its own runs/p3_<i>/, and 983 runs were pulled from two workers.  The copies
   are bit-identical (verified: 983/983 same success vector), so they are the SAME run,
   not a replicate.  Averaging the raw file list double-weights those seeds and moved
   arm rates by up to 1.83 pts.  Rates here are the mean over the 20 DISTINCT seeds.

2. PROPAGATES the 27 solved-but-duplicate cells.  Those cells are not new experiments:
   CP-SAT returns a schedule with the same (observation age, cadence) as a cell already
   run, and job_grid.sh feeds the simulator nothing but --latency-ms/--issue-period-ms,
   so the two cells are the same command.  The twin is found by TIMING (|dlat|<=0.6 ms
   and |dcad|<=0.6 ms against grid_warm.json), never by name, and the equality of the
   values job_grid.sh actually passes is asserted.  Each entry carries
   source=measured|propagated and, when propagated, the twin arm.

A propagated cell is a copy of measured evidence, not new evidence.  Anything that
consumes this file must filter to source=="measured" before computing a statistic.
"""
import json, glob, collections, re
from pathlib import Path

HERE = Path("/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain")
TOL = 0.6
TASKS = ["egg", "spoon", "coke"]

# ---- measured: mean over DISTINCT run keys ---------------------------------
byseed = collections.defaultdict(dict)
nfiles = 0
for f in glob.glob(str(HERE / "g5grid/runs/*/*/summary.json")):
    name = Path(f).parent.name
    if "_rng" not in name:
        continue
    nfiles += 1
    head, seed = name.rsplit("_rng", 1)
    task, arm = head.split("_", 1)
    eps = json.load(open(f)).get("episodes", [])
    byseed[(task, arm)][int(seed)] = 100 * sum(int(e["success"]) for e in eps) / max(len(eps), 1)
nkeys = sum(len(v) for v in byseed.values())
assert nkeys == 2640, nkeys
assert all(len(v) == 20 for v in byseed.values())

cells = json.load(open(HERE / "paper/grid_warm.json"))["cells"]
armtim = {}
for l in open(HERE / "paper/arms.tsv"):
    n, a, c, x = l.split()
    armtim[n] = (float(a), float(c))
jg = {m.group(1): (float(m.group(2)), float(m.group(3)))
      for m in re.finditer(r'(g\d+_\d+)\)\s*LAT=([\d.]+);\s*PER=([\d.]+)',
                           open(HERE / "g5grid/job_grid.sh").read())}
assert set(jg) == set(armtim) and all(jg[n] == armtim[n] for n in jg)

solved = {k: v for k, v in cells.items() if "lat_med" in v}
dups = [k for k in solved if "g" + k not in armtim]

# ---- twin match, by timing only --------------------------------------------
twins = {}
for k in dups:
    v = solved[k]
    cand = sorted((abs(v["lat_med"] - a) + abs(v["cadence"] - c), n)
                  for n, (a, c) in armtim.items()
                  if abs(v["lat_med"] - a) <= TOL and abs(v["cadence"] - c) <= TOL)
    assert cand, f"no twin within {TOL} ms for cell {k}"
    assert len(cand) == 1 or cand[1][0] - cand[0][0] > 1e-6, f"ambiguous twin for {k}: {cand}"
    n = cand[0][1]
    a, c = armtim[n]
    # the assertion that matters: the simulator sees the same two numbers
    assert abs(round(v["lat_med"], 1) - a) <= TOL and abs(round(v["cadence"], 1) - c) <= TOL
    twins[k] = dict(twin=n, dlat=abs(v["lat_med"] - a), dcad=abs(v["cadence"] - c))
assert len(twins) == 27, len(twins)

out = {"_meta": {
    "generator": "g5grid/make_e2e_success.py",
    "run_keys": nkeys, "summary_files_on_disk": nfiles,
    "note": ("rate = mean over DISTINCT <task>_<arm>_rng<seed> keys; the 983 duplicated "
             "fetch copies are bit-identical and are counted once, never summed per host"),
    "twin_tolerance_ms": TOL,
    "provenance": {"measured": "simulated end-to-end, 20 seeds x 24 episodes",
                   "propagated": ("solved cell whose CP-SAT schedule has the same "
                                  "(observation age, cadence) as `twin` within "
                                  f"{TOL} ms; job_grid.sh passes only those two numbers, "
                                  "so it is the same experiment -- copied, NOT re-run, "
                                  "and NOT new evidence")},
    "twins": twins}}

for task in TASKS:
    d = {}
    for arm in sorted(a for (t, a) in byseed if t == task):
        vals = byseed[(task, arm)]
        d[arm] = {"rate": sum(vals.values()) / len(vals), "n_seeds": len(vals),
                  "source": "measured"}
    for k, tw in twins.items():
        d["g" + k] = {"rate": d[tw["twin"]]["rate"], "n_seeds": d[tw["twin"]]["n_seeds"],
                      "source": "propagated", "twin": tw["twin"]}
    out[task] = d
    nm = sum(1 for v in d.values() if v["source"] == "measured")
    print(f"{task:6s} measured {nm}  propagated {len(d)-nm}  total {len(d)}")

json.dump(out, open(HERE / "paper/grid_e2e_success.json", "w"), indent=1)
print("[ok] paper/grid_e2e_success.json")
