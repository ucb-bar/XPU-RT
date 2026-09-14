#!/usr/bin/env python3
"""Merge the 3600 s re-solve of the 13 UNKNOWN cells into paper/grid_warm.json.

Only cells whose status IMPROVES are written: UNKNOWN -> OPTIMAL/FEASIBLE/INFEASIBLE.
A cell that is still UNKNOWN keeps its record and its `resolved_at` is bumped to record
the larger budget it survived -- UNKNOWN is a budget outcome, so the budget it was tried
at is part of the claim.

Asserts monotonicity in the window afterwards: a larger window is a strictly looser
deadline, so a cell solvable at a tighter window can never be INFEASIBLE at a looser one.
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).parent
GRID = HERE.parent / "paper/grid_warm.json"
g = json.load(open(GRID))
cells = g["cells"]
status = {}
for l in open(HERE / "gx_status.txt"):
    m = re.match(r"p=(\d+) w=(\d+) -> (\w+)", l.strip())
    if m: status[f"{m.group(1)}_{m.group(2)}"] = m.group(3)
tim = json.load(open(HERE / "gx.json"))
assert len(status) == 13, f"expected 13 re-solved cells, got {len(status)}"

changed = []
for k, st in status.items():
    c = cells[k]
    assert c["status"] == "UNKNOWN", f"{k} was {c['status']}, not UNKNOWN"
    if st == "UNKNOWN":
        c["resolved_at"] = "still UNKNOWN after a 3600 s deep pass"
        continue
    c["status"] = st
    c["resolved_at"] = "3600s deep pass"
    for f in ("lat_med", "lat_max", "cadence", "makespan"):
        c.pop(f, None)
    if st in ("OPTIMAL", "FEASIBLE"):
        v = tim[k]
        c.update({f: v[f] for f in ("lat_med", "lat_max", "cadence", "makespan")})
    changed.append((k, st))

PERIODS = [100,110,120,130,140,150,165,180,200,220,250,283]
WINDOWS = [240,260,275,290,305,320,350,400,450]
for p in PERIODS:
    seen_solved = False
    for w in WINDOWS:
        st = cells[f"{p}_{w}"]["status"]
        if st in ("OPTIMAL", "FEASIBLE"): seen_solved = True
        elif st == "INFEASIBLE" and seen_solved:
            sys.exit(f"MONOTONICITY VIOLATED: p={p} solvable at a tighter window but "
                     f"INFEASIBLE at w={w}")
print("monotonicity in the window: OK")

tal = {}
for c in cells.values(): tal[c["status"]] = tal.get(c["status"], 0) + 1
g["tally"] = tal
g["n_solved"] = tal.get("OPTIMAL", 0) + tal.get("FEASIBLE", 0)
pts = {(round(c["lat_med"], 1), round(c["cadence"], 1))
       for c in cells.values() if "lat_med" in c}
g["n_distinct_operating_points"] = len(pts)
g["source"]["deep_pass"] = (g["source"].get("deep_pass", "") +
    "; the 13 cells still UNKNOWN after that were re-solved at 3600 s "
    "(heatmap_deeper.sh, 13-way at 7 CP-SAT workers each)")
json.dump(g, open(GRID, "w"), indent=1)
print(f"changed: {changed}")
print(f"tally {tal}  n_solved {g['n_solved']}  distinct points {g['n_distinct_operating_points']}")
