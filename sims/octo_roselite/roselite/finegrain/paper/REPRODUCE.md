# Reproducing the figures

All ten scripts run clean and each is **byte-deterministic** across consecutive runs
(verified; the two that jitter points seed their RNG). What they need to run varies, and
that is the part that matters for anyone other than the author.

```bash
python fig_<name>.py        # writes fig_<name>.png beside itself
```

## Dependency tiers

**Tier 1 — reproduces from committed data alone.** Inputs are small JSON/TSV that live
beside the script.

| figure | inputs |
|---|---|
| `fig_schmoo.py` | `grid_warm.json`, `arms.tsv` |
| `fig_e2e_heatmap.py` | `grid_e2e_success.json`, `grid_warm.json` |
| `fig_energy.py` | `energy_cache.json` |

`fig_energy` used to need the 700 MB `traces_torque2/` tree. Both quantities it plots
integrate to ONE SCALAR per episode, so `make_energy_cache.py` reduces the tree to a
39 KB cache. Verified equal to the full computation on all 36 cells at rtol 1e-12. The
script prefers the cache and falls back to the traces when it is absent.

**Tier 2 — needs the run summaries** (`../g5fine/runs/`, ~13 MB of `summary.json`; no
arrays). Regenerate with `g5fine/job.sh` / `job_torque2.sh`.

`fig_metrics.py`, `fig_operating_points.py`, `fig_pareto.py`, and panel A of
`fig_grasp_failure.py`.

**Tier 3 — needs full per-tick traces** (`../traces_torque2/`, ~700 MB). These draw
positions over time, so they cannot be reduced to scalars.

`fig_trajectory.py`, and panels B/C of `fig_grasp_failure.py`. Regenerate with
`g5fine/job_torque2.sh` (36 cells x 24 episodes; a few GPU-hours).

**Tier 4 — needs board run logs.** `fig_schedule.py` and `fig_lane_balance.py` read the
QRB5165 runtime logs. Point them anywhere with

```bash
export OCTO_REPRO_RUNS=/path/to/repro_runs      # default: the author's absolute path
```

## Not reproducible from the repo alone

Tiers 2-4 live in `octo_work/`, which is a **sibling of the XPU-RT checkout, not inside
it**, and is untracked. Only the `fig_schmoo` inputs are committed (under
`qnn_models/octo/schmoo/`). Anyone reproducing tiers 2-4 must either be given that tree
or regenerate it from the job scripts, which needs GPU workers and the SimplerEnv
install described in `REPRODUCE_ARM_EXPERIMENTS.md`.

Making tiers 1-2 fully self-contained would mean committing ~13 MB of summaries; tier 3
cannot reasonably be committed at 700 MB.

## Regenerating derived inputs

```bash
python make_energy_cache.py                 # traces_torque2/ -> energy_cache.json
python ../g5grid/analyze_merged.py          # the sweep tables the captions quote
python ../g5grid/resolution_audit.py        # tick-grid collapse -> resolution_audit.json
```

```bash
python ../g5grid/make_e2e_success.py        # g5grid/runs/ -> grid_e2e_success.json
```

`grid_e2e_success.json` is per-cell mean success over `g5grid/runs/`, and carries a
`source` flag per cell. It is built by `g5grid/make_e2e_success.py`, which does two
things by hand-rolled analysis rather than a plain glob:

* **deduplicates by run key.** `g5grid/runs/` holds 3623 `summary.json` files but only
  2640 distinct `<task>_<arm>_rng<seed>` keys -- `fetch_phase3.sh` rsyncs each worker
  into its own `runs/p3_<i>/` and 983 runs were pulled from two workers. The copies are
  bit-identical (asserted), so they are one run, not a replicate. Averaging the raw file
  list double-weights those seeds and moved arm rates by up to 1.83 pts.
* **propagates structural duplicates.** 27 of the 71 solved grid cells return a CP-SAT
  schedule whose `(lat_med, cadence)` matches a cell already simulated. `job_grid.sh`
  passes the simulator only `--latency-ms` and `--issue-period-ms`, so those cells are
  the same command; they are marked `source: "propagated"` with their `twin`, matched by
  TIMING within 0.6 ms and never by name. A propagated cell is a copy of measured
  evidence: consumers must filter to `source == "measured"` before computing a statistic,
  and `fig_e2e_heatmap.py` asserts that it has.

`grid_warm.json` comes from the scheduler host via `qnn_models/octo/schmoo/` (see its
README).

## Caveat on the captions

Several captions quote numbers computed elsewhere (noise floors, trend statistics,
BH-adjusted p-values) rather than recomputing them at draw time. They are correct as of
the sweep in `g5grid/runs/`, and `analyze_merged.py` is the source. If that sweep is
extended, the captions must be refreshed by hand -- they will NOT update themselves.
