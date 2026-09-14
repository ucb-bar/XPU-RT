# The 27 propagated plane cells — why they are not re-run

`arms_prop.tsv` lists the 27 cells of the CP-SAT grid whose value in the plane
heatmaps is copied from a twin rather than simulated. The question was whether
that copy is an assumption worth removing by measurement. It is not, and the
reason is exact rather than approximate.

## They are the same command

`trace_eval2.py` is parameterised by exactly two numbers: `--latency-ms` and
`--issue-period-ms`. Comparing each propagated cell's `(lat_med, cadence)` from
`paper/grid_warm.json` against its twin's:

* **26 of 27 are bit-identical at full float precision.**
* **1 of 27** — `g150_400`, 319.012 ms against `g150_350`'s 318.970 ms — differs
  by 0.042 ms. The arm tables the job scripts read format latency to 0.1 ms, so
  both become `319.0` and even that cell is the same command as executed.
* **0 of 27 are bit-different as the pipeline actually runs them.**

The 27 collapse to five distinct commands: `g220_260` (x8), `g250_260` (x8),
`g283_260` (x8), `g200_400` (x2), `g150_350` (x1).

Simulating them would have been 1,080 runs, ~364 lane-hours, ~$370, and would
have produced 27 differently-labelled samples of harness nondeterminism.

## Why the grid produces duplicates at all

Past roughly a 220 ms release period the deadline window stops binding: the
schedule already finishes inside it, so CP-SAT returns the same schedule whatever
window is requested. The duplicates are a property of the search space, not an
artefact of the sweep.

## The one thing that WAS arbitrary

`make_e2e_success.py` folds a cell into a twin when
`|dlat| + |dcad| <= 0.6 ms`. That threshold is bookkeeping, not physics — the
simulator tick is 40 ms. It is why `g150_400` (0.042 ms away) was folded while
`g150_450` (1.348 ms away) was kept as its own operating point, even though at a
40 ms tick neither difference can change a rollout. The dedup decides which
representative gets measured; it does not affect any value, because the
representative's measurement applies to every cell that shares its command.

## Still open, by choice

12 cells are `UNKNOWN`: CP-SAT found no schedule within its budget. That is a
reported outcome of the search, not a gap in the plane, and they sit at SHORT
periods (100-130 ms) — the aggressive corner, not the dominated one.
`cand_c16_plane_complete.png` marks them `?` and the 25 `INFEASIBLE` cells `x`,
because "no solution found in budget" and "no solution exists" are different
claims.
