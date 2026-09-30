# Warehouse showdown on the 3-network chain: ROS 2 on all 8 cores across the camera-rate sweep

The paper figure (`scripts/showdown_paper_figure.py`, layout of `fig_hil_showdown`) drawn from one pairing:
XPU-RT against ROS 2 with two model instances pinned across both clusters, at the same camera rate, on the
same scene, each replaying what it measured on the SpaceMiT K1. Every number the figure prints is written to
its sidecar and re-derived by `scripts/verify_showdown_figure.py`.

## What decides a flight

A flight replays two board measurements per arm: the control-output cadence (`scripts/ctrl_trace_from_board.py`)
and the camera-to-control latency. The navigation goal is held for one camera period in both arms, since
perception cannot refresh faster than the camera.

In the default ROS 2 graph control is triggered by the navigation callback, which is triggered by perception,
so the control cadence is the pipeline's throughput: `min(camera, ~62 Hz)` for the two-instance graph. XPU-RT's
schedule runs control at 100 Hz from any camera. Below ~60 ms the latency does not change flight outcomes,
and control rates of roughly 45 Hz and above behave alike, so the camera rate sets whether the baseline's
control rate is inside that region.

## 1. Board measurements (K1)

```bash
# ROS 2, two perception instances (pools P0-3 and E4-7), every camera rate, three replicates
for r in 1 2 3; do RATES="25 30 36 38 40 45 60 75 90 120" scripts/ros_traced_matrix.sh vanilla4x2 $r; done
.venv/bin/python scripts/pull_ros_traced.py

# XPU-RT, greedy placement on the board's per-width costs, at the chosen camera rate (three replicates)
scripts/xpu_greedy_shard_at_rate.sh 36

# XPU-RT, CP-SAT with two rounds of board feedback (30 Hz shown; see "Rates" for 45 Hz)
LIMIT=5400 CPSAT_WORKERS=4 scripts/rate_feedback_loop.sh 30
```

Measured (pooled over three replicates, `scripts/measured_timing.py --verify` re-derives each):

| camera | ROS 2 control | ROS 2 camera→goal | XPU-RT arm | XPU-RT camera→control |
|---|---|---|---|---|
| 30 Hz | 30.0 Hz | 31.4 ms | CP-SAT feedback round 1 | 30.1 ms |
| 36 Hz | 36.0 Hz | 32.2 ms | greedy, measured shard costs | 30.1 ms |
| 38 Hz | 38.0 Hz | 36.4 ms | greedy, measured shard costs | 59.1 ms (start backlog; not flown) |
| 40 Hz | 40.0 Hz | 37.5 ms | greedy, measured shard costs | 39.8 ms |
| 45 Hz | 45.0 Hz | 37.0 ms | — | — |

Both arms keep all eight harts busy (per-core sampler over the run).

## 2. Cadence traces and registry

```bash
.venv/bin/python scripts/ctrl_trace_from_board.py results/codesign_feedback/ros_traced/36_vanilla4x2_r1/ctrl_gaps.csv \
    --out results/codesign_feedback/ctrl_traces/ros_vanilla4x236.csv
.venv/bin/python scripts/ctrl_trace_from_board.py results/codesign_feedback/xpurt_long/trace_w2pg36r1_other_run1.csv \
    --out results/codesign_feedback/ctrl_traces/xpu_w2pg36.csv
```

Each trace has a `ReplayArm` in `scripts/figure_constants.py` whose latency is derived from a row of
`scripts/measured_timing.py` (`ROS_VANILLA[("vanilla4x2", 36)]`, `SOLVER_ARMS["w2pg36r"]`).

## 3. Flight census (GPU)

```bash
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_rate30.sh      # 30 Hz
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_rate3640.sh    # 36 and 40 Hz
```

Standard cell: course a, prop density 0.30, people 2.4 m, gain 0.0055, cruise 1.0-1.8 m/s, twelve seeds.
Flights are admitted one at a time under `results/codesign_feedback/gpu_admit.lock`; a batch whose simulator
log shows a GPU fault is written to `quarantine.txt`, and `results/codesign_feedback/flight_quarantine.csv` lists
the faulted batches every reader drops (`scripts/flight_quarantine.py` re-checks each entry).

Paired over the same (cruise, seed) cells:

| camera | XPU-RT arm | XPU-RT completed | ROS 2 completed | mean gates XPU / ROS | gate difference (bootstrap 95 %) | completion difference (Newcombe 95 %) |
|---|---|---|---|---|---|---|
| 30 Hz | CP-SAT, 55.2 ms | 6/60 | 0/60 | 1.87 / 0.32 | +1.55 [+1.28, +1.83] | +10.0 [+2.0, +20.1] |
| 30 Hz | CP-SAT feedback round 1, 30.1 ms | 6/60 | 0/60 | 2.07 / 0.32 | +1.75 [+1.48, +2.02] | +10.0 [+2.0, +20.1] |
| **36 Hz** | **greedy, measured shard costs, 30.1 ms** | **7/60** | **1/60** | **1.90 / 1.02** | **+0.88 [+0.55, +1.23]** | **+10.0 [+0.7, +20.6]** |
| 40 Hz | greedy, measured shard costs, 39.8 ms | 5/60 | 7/60 | 1.85 / 1.73 | +0.12 [−0.23, +0.48] | −3.3 [−14.9, +8.1] |

The baseline never completes at 30 Hz and matches XPU-RT at 40 Hz; 36 Hz is the rate in between, where it
completes sometimes and less often than XPU-RT (bootstrap over the paired per-cell gate differences, seed 11,
4000 resamples).

## 4. The displayed pair

```bash
scripts/display_pairs_rate36.sh      # the census cells, re-flown with figure data dumped
scripts/display_search_rate36.sh     # every seed per cruise in the display script itself
```

The display script flies one episode with the layout seed equal to the flight seed, so a census cell is not
the same scene; a pair is kept only when, on the scene it flies, XPU-RT completes and ROS 2 crashes before
gate 3. The pair drawn: cruise 1.2 m/s, seed 1000 — XPU-RT 4/4 gates in 1286 steps (100.3 Hz control, 30.1 ms),
ROS 2 clears G1 and hits a gate frame at step 302 (36.1 Hz control, 32.2 ms).

Panel A's per-scene tally: twelve flights per arm on that layout.

```bash
T=results/codesign_feedback/ctrl_traces
CELL=r36_l1000 LAYOUT_SEED=1000 CRUISE=1.2 MAX_SIMS=3 NEED_MB=10000 \
  ARMS="xpu:$PWD/$T/xpu_w2pg36.csv:30.1:27.8 ros8:$PWD/$T/ros_vanilla4x236.csv:32.2:27.8" scripts/scene_runs_pair.sh
```

## 5. Panel D and panel I

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
MAX_SIMS=3 NEED_MB=10000 CONDS="xpu_w2pg36:$T/xpu_w2pg36.csv ros_x2_36:$T/ros_vanilla4x236.csv" CRUISE=1.2 \
  ER=$PWD/results/codesign_feedback/energy_runs_r36 OUTCSV=$PWD/results/codesign_feedback/flight_energy_r36.csv \
  scripts/run_energy_pair.sh

D=results/codesign_feedback
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$D/xpurt_long/trace_w2pg36r1_other_run1.csv:$D/xpurt_long/cpu_w2pg36r1_other_run1.csv:$D/xpurt_long/manifest_w2pg36r1_other_run1.json:schedules/fig_w2pg36_greedy_clamped.json" \
  --arm "ros8:ros:$D/ros_traced/36_vanilla4x2_r1/trace.csv:$D/ros_traced/36_vanilla4x2_r1/cpu.csv:$D/ros_traced/36_vanilla4x2_r1/manifest.json" \
  --spec data/toplevel/wh_chain36_w2p.json --window-ms 160 --out-prefix $D/refined/rate36/measured_gantt_r36
```

Each Gantt row opens its window where the row's own frames have the run's median latency (the sidecar records
both medians). Panel D over six seeds per arm: ROS 2 commands 35.7× the mean |moment| and 18.2× the power of
XPU-RT.

## 6. Render and verify

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_r36.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_rate3640/display36/search_c1.2/xpu_s1000_figdata \
  --ros-dir $R/campaign_rate3640/display36/search_c1.2/ros_s1000_figdata \
  --scene-records $R/campaign_scene/r36_l1000 --display-cruise 1.2 \
  --xpu-trace xpu_w2pg36.csv --ros-trace ros_vanilla4x236.csv \
  --xpu-label "XPU-RT · greedy on measured costs" --ros-label "ROS 2 on all 8 cores" \
  --camera-hz 36 --gantt-prefix $R/refined/rate36/measured_gantt_r36 --gantt-rows xpu,ros8 \
  --out $R/refined/warehouse_showdown_paper_r36

.venv/bin/python scripts/verify_showdown_figure.py --metrics $R/refined/warehouse_showdown_paper_r36_metrics.json
.venv/bin/python scripts/measured_timing.py --verify
.venv/bin/python scripts/flight_quarantine.py
.venv/bin/python -m pytest tests xpu-rt/tests -q
```

## Rates

A CP-SAT table that minimises makespan fills the hyperperiod. When the camera period is close to one YOLO
frame's span (45 Hz: 22.2 ms period, 22-24 ms frame) a ~2 ms per-frame overrun on the board accumulates into a
start backlog that grows through the run; re-costing from the executed traces does not add slack. At 30 Hz
(33.3 ms period) the same loop runs 30.1 ms with no backlog. Per-frame start lag after release, planned span and
executed span locate this (see `docs/Feature/feedback_loop_reference.md`).
