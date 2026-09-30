# HIL flight-envelope data — what it is, how to reproduce the figure, and its scope

This directory holds the raw per-flight data behind the warehouse HIL figure (the flight-envelope
error-bar panel and the single-flight showdown). Every plotted number is regenerated from these
CSVs by one script — nothing in the figure is hand-entered.

## Reproduce every plotted number
```
scripts/measured_timing.py --verify            # every board number, re-derived from its trace
scripts/campaign_select.py                      # the showdown cell, chosen by the rule written before the runs
scripts/render_showdown_measured.sh             # the composite, then scripts/verify_showdown_figure.py
<env_isaaclab>/python scripts/hil_story_figure.py   # envelope / generalization / mechanism
```

## The second-form showdown (`campaign_v2/`)
The flight figure's arms replay the control-output cadence measured on the K1 for each runtime
(`ctrl_traces/*.csv`, from `scripts/ctrl_trace_from_board.py`): XPU-RT's CP-SAT table and greedy
table of the same spec, and the out-of-the-box ROS 2 graph (`ros_vanilla4`: one process per node,
unpinned, the model's 4-hart YOLO, control in the goal callback; `ros_vanilla4t` timer-driven;
`ros_vanilla` serial YOLO). Ten cruise speeds × 12 seeds per arm under the fixed gain (0.0055) and,
where the calibrated policy is drawn, each arm at 0.5 / its replayed rate. `scripts/campaign_select_v2.py`
names the displayed cell by the rule written before the runs; `campaign_v2/display/` holds every
display attempt (video + figure data from one flight each) and `campaign_v2/display_v2*.log` the
attempt log.

## The two data sources (they are DIFFERENT experiments — read this before comparing them)
Every flight is one real headless Isaac-Lab run of `sims/scripts/sweep_rate_demo.py` (RL/MLP
controller `nav_fused_v12_cnn.pt`, YOLOv8n perception, fixed 4-gate warehouse course). What differs:

| | **Envelope** (`hil_ablation.csv`, `gain_controlled/`) | **Showdown** (`campaign/campaign.csv`, `mean_gap/`) |
|---|---|---|
| what it is | the error-bar panel | the two displayed flights and their cell's k/n |
| control rate set by | clean ZOH **decimation** (`sched_latency_ms` 8/18/28/38 → 100/50/33/25 Hz) | the arm's measured control cadence on the K1 as **`sched_latency_ms`** (`scripts/measured_timing.py`); the simulator holds the command for ceil(cadence / 10 ms) control steps, so XPU-RT's 10.00 ms is the 100 Hz tick and the ROS 2 pool arm's 29.9 ms the 33 Hz tick |
| controller gain | fixed `moment_scale=0.0055` (`hil_ablation.csv`) and calibrated `0.5/eff_hz` (`gain_controlled/`), both drawn | **fixed** `moment_scale=0.0055`, so control cadence is the only difference between the arms |
| grid | 5 speeds × 4 rates × 12 seeds | 6 speeds × 4 arms × 12 seeds; the displayed cell (cruise 1.4) pooled with the earlier same-seed runs |

**The displayed flights are one draw each.** The simulator is not run-to-run deterministic (a
seed that crashed mid-course in the campaign can fly clean when re-dumped), so the figure
shows one flight per arm from the selected cell and carries the cell's k/n in the legend;
every dump attempt is appended to `campaign/display/display.csv`, and
`scripts/verify_showdown_figure.py` checks the displayed dumps' outcomes and gain. **Do not pool
the two sources or read them cell-for-cell.**

## Data versions (Sept 2026)
- `hil_ablation_v1_120flights_fixedgain.csv` — the original **120-flight** envelope (6 seeds/cell),
  fixed gain, produced by the earlier sim. **Archived permanently** (sha `706fd92…`); never deleted.
- A **fresh full grid at 12 seeds/cell (240 flights)** is being regenerated from the *canonical*
  repo with one consistent sim version (→ `hil_ablation_v2.csv`); once verified it **supersedes**
  `hil_ablation.csv` as the figure source. `reproduce_hil_figure.py` recomputes whatever is current,
  so the printed numbers (and the caption) update with the new data. v1 stays for provenance.
- `hil_ablation_courseB.csv` — the same sweep on a *second gate course* (`WAREHOUSE_COURSE=b`), for
  cross-course generalization. **Never pool it with course A** (different layout).

## Why the rate axis has a floor — the measured cadences (the bridge)
The envelope shows *that* success needs a control-rate floor; the board traces show *where each
runtime lands* on that axis. Both arms run the same ModelBlaster kernels on the same K1 at a
45 Hz camera; only the orchestrator differs (`docs/Baselines/ros_baseline_reproduction.md`,
`docs/Evaluation/measurements_and_ablations.md` §1.5–1.6; every number re-derives under
`scripts/measured_timing.py --verify`):

| runtime | control-output gap | control rate | camera→control |
|---|---|---|---|
| ROS 2 as deployed (one process, default executor, YOLO on a 4-hart pool) | 29.9 ms mean | 33 Hz | 119 ms |
| ROS 2 as shipped (serial YOLO, unpinned) | 53 ms mean | 17 Hz | 213 ms |
| ROS 2, multi-threaded executor | 10.00 ms | 100 Hz | 264 ms, 48 % of frames dropped |
| ROS 2, hand-partitioned (a process per node, pinned) | 10.00 ms | 100 Hz | 53 ms |
| XPU-RT (`alt2` schedule) | 10.00 ms mean, 9.98–10.10 | 100 Hz | 40 ms, every frame on time |

Scope the claims separately, because they have different reach:

* the **control-cadence** result — the crash — holds for ROS 2 as deployed: one thread serves
  every callback, so the control timer waits behind YOLO. Control on its own thread or process,
  or the multi-threaded executor, holds 10.00 ms too, and that is measured and shown;
* the **latency and throughput** result holds for every ROS 2 deployment, including the
  hand-partitioned one, and widens under load: with two cameras and the heavier stack it is
  237 ms with 13 % of frames dropped against 74 ms with none, and the multi-threaded executor
  254 ms with 67 % dropped (`refined/deployment_layers.png`).

## Files
- `hil_ablation.csv` — the envelope panel (the committed figure source; v1 today, v2 after the refresh).
- `hil_flights_master.csv` — **435 flights** unioned across every experiment, with `source`, a
  `regime` column, and a `success` flag. Regimes: envelope (120), calibrated-gain grid (96), showdown
  (18), perception-freshness/safety (116, from the canonical `perc_crash/` sweeps), verification (85).
  The `regime` column exists so these are **never silently pooled** — the figure uses only the
  envelope + showdown regimes; `perc_crash` is a separate perception-freshness study. *(Checked:
  within `perc_crash`, perception hold 22 ms vs 65 ms did NOT separate success — fresh22 11/16 vs
  stale65 11/16, xpu_safe 10/16 vs ros_safe 10/16 — a null on that knob. Kept for breadth; no claim
  is built on it.)*
- `crash_verify/new_xpu.csv`, `new_ros50.csv`, `new_ros.csv` — the showdown seeds (6 each; 18 after refresh).
- `crash_verify/run_newshowdown.sh`, `run_ros50.sh` — the exact showdown invocations.
- `scripts/hil_ablation_grid.sh` — the envelope grid driver (regenerates `hil_ablation.csv`).

## Column dictionary (`hil_ablation.csv`; the master adds `source` + `success`)
- `seed` — RNG seed (also drives the randomized obstacle/people/crate field this flight sees).
- `cruise_speed` — commanded forward speed (m/s).
- `sim_dt`, `decimation`, `control_dt_ms` — physics step, control decimation, resulting control period.
- `sched_latency_ms` — worst-case command latency; the motor command is held (ZOH) between refreshes.
- `hold_steps` — control steps per refresh = `ceil(latency / control_dt)`.
- `eff_cmd_hz` — effective command rate = `1000 / (control_dt_ms × hold_steps)` (the x-axis).
- `moment_scale` — controller action→moment gain (0.0055 fixed for the envelope).
- `gates_passed` — gates cleared (of 4); `steps` — sim steps survived; `K` — total gates.
- `outcome` — `success` (all gates) / `crash` / `timeout`.
- master extras: `walk_speed, gust, motor_tau, pipeline_zoh, percep_*` (blank unless that run logged them),
  `crash_type`, `success` (1 iff `outcome==success`).
