# octo_roselite — scheduling a VLA policy on a heterogeneous SoC

Source for the manipulator experiment: an Octo-small-1.5 INT8 policy scheduled
across the CPU/DSP/HTA engines of a Qualcomm QRB5165, with the resulting
(observation latency, release period) replayed inside SIMPLER-env to measure what
the robot actually achieves.

**Read `roselite/finegrain/paper/MANIPULATOR_EXPERIMENT.md` first** — it is the
full writeup: setup, metrics, the three results, and the threats to validity.

## What is here, and what is not

This directory holds **source only**: the harness, the sweep orchestration, the
analysis, and the figure generators. Generated artefacts and raw trace data are
excluded by `.gitignore` — roughly 3 GB of per-episode `.npy`/`.npz`/`.json` plus
rendered `.png`/`.mp4`. They are reproducible from these scripts given GPU time,
and they do not belong in git.

At the time of writing the data lives on the machine that produced it, under
`/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/`:

| dataset | size | what it is |
|---|---|---|
| `finegrain/traces_torque3/` | 145 MB | curated ladder, 30 seeds on eggplant/spoon/drawer |
| `finegrain/g5grid/plane3_archive/` | 508 MB | full per-cell record of the 44-point plane |
| `finegrain/g5grid/plane3_runs/` | 80 MB | the plane's reduced scalars |
| `finegrain/runs_video/` | 18 MB | rendered rollouts used for figure snapshots |

Total simulated evidence: **42,240 episodes** on the plane (44 operating points x
4 tasks x 10 seeds x 24 episodes), plus the curated ladder at 30 seeds x 24
episodes on three tasks.

## Layout

| path | what |
|---|---|
| `roselite/finegrain/trace_eval2.py` | the harness: replays a schedule's latency/period inside SIMPLER-env and logs 20 per-episode channels |
| `roselite/finegrain/g5fine/reduce_torque3.py` | on-worker reduction to per-episode scalars |
| `roselite/finegrain/g5grid/` | fleet orchestration: start/stop, job scripts, shard + launch, monitors, fetch, recovery |
| `roselite/finegrain/paper/` | analysis and the paper figures |
| `roselite/finegrain/paper/layouts/` | figure design exploration, 45 candidates + `LAYOUT_OPTIONS.md` |
| `roselite/finegrain/calib/` | energy calibration to joules from published actuator data |

## Key documents

| file | what it establishes |
|---|---|
| `paper/MANIPULATOR_EXPERIMENT.md` | the full experiment writeup |
| `paper/PLANE3_SANITY.md` | validation of the 1760-cell sweep, and its outstanding caveats |
| `ENERGY_AUDIT.md` | why the original energy metric was invalid and what replaced it |
| `calib/ENERGY_CALIBRATION.md` | conversion to joules, and why the raw torque integral cannot be converted at face value |
| `g5grid/PROPAGATED_CELLS.md` | why 27 of the 108 grid cells are copies rather than measurements |
| `paper/layouts/LAYOUT_OPTIONS.md` | figure design decisions and what was rejected |

## Reproducing

The sweep needs GPU workers; `g5grid/farm_start.sh <tag>` brings up the fleet and
captures current public IPs (they are reassigned on restart, so a stored host
list goes stale). Job scripts take `TASK:ARM:SEED` and write one reduced cell
each. Figures regenerate from the committed `.json`/`.tsv` inputs in `paper/` and
`calib/` plus the trace data above.
