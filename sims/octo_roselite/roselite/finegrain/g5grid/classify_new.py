#!/usr/bin/env python3
"""Classify the cells re-solved at 3600 s: newly SOLVED vs still UNKNOWN vs INFEASIBLE,
and for each newly solved cell, NEW operating point vs duplicate of an arm already run.

Input : gx_status.txt   ("p=P w=W -> STATUS", one line per cell)
        gx.json         (extract_gx.py output: per-cell lat_med/lat_max/cadence/makespan)
Output: prints the classification; writes new_arms.tsv for anything genuinely new.

"Duplicate" is decided ONLY by the two numbers job_grid.sh passes to the simulator
(--latency-ms = lat_med, --issue-period-ms = cadence), matched within 0.6 ms.  Never
by name.
"""
import json, sys, re
from pathlib import Path
HERE = Path(__file__).parent
TOL = 0.6
arms = {}
for l in open(HERE / "arms.tsv"):
    n, a, c, x = l.split()
    arms[n] = (float(a), float(c))
status = {}
for l in open(HERE / "gx_status.txt"):
    m = re.match(r"p=(\d+) w=(\d+) -> (\w+)", l.strip())
    if m: status[f"{m.group(1)}_{m.group(2)}"] = m.group(3)
tim = json.load(open(HERE / "gx.json"))

solved, unk, inf, dup, new = [], [], [], [], []
for k, st in sorted(status.items(), key=lambda kv: (int(kv[0].split("_")[0]), int(kv[0].split("_")[1]))):
    if st in ("OPTIMAL", "FEASIBLE"):
        solved.append(k)
        v = tim.get(k)
        if v is None:
            print(f"  !! {k} status={st} but no schedule extracted"); continue
        cand = sorted((abs(v["lat_med"] - a) + abs(v["cadence"] - c), n)
                      for n, (a, c) in arms.items()
                      if abs(v["lat_med"] - a) <= TOL and abs(v["cadence"] - c) <= TOL)
        if cand:
            dup.append((k, cand[0][1], v))
        else:
            new.append((k, v))
    elif st == "UNKNOWN":
        unk.append(k)
    elif st == "INFEASIBLE":
        inf.append(k)
    else:
        print(f"  !! {k} unexpected status {st}")

print(f"cells re-solved at 3600 s : {len(status)}")
print(f"  newly SOLVED            : {len(solved)}   {solved}")
print(f"  still UNKNOWN           : {len(unk)}   {unk}")
print(f"  now proven INFEASIBLE   : {len(inf)}   {inf}")
print()
for k, n, v in dup:
    a, c = arms[n]
    print(f"  DUPLICATE  {k:8s} lat_med={v['lat_med']:8.3f} cad={v['cadence']:7.3f} "
          f"-> twin {n} (dlat={abs(v['lat_med']-a):.3f} dcad={abs(v['cadence']-c):.3f})")
for k, v in new:
    print(f"  NEW POINT  {k:8s} lat_med={v['lat_med']:8.3f} cad={v['cadence']:7.3f} "
          f"lat_max={v['lat_max']:8.3f}")
if new:
    with open(HERE / "new_arms.tsv", "w") as f:
        for k, v in new:
            f.write(f"g{k}\t{v['lat_med']:.1f}\t{v['cadence']:.1f}\t{v['lat_max']:.1f}\n")
    print(f"\n[ok] wrote new_arms.tsv ({len(new)} arms)")
else:
    print("\nno genuinely new operating point -- nothing to simulate")
