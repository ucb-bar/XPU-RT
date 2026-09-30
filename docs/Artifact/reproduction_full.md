# Reproducing the study end to end — host, simulator, board

One document that says, for every artifact of the showdown study (the composite `refined/warehouse_showdown_v3*`,
the atlas, the final figure, Figure 10 `warehouse_showdown_paper10`, the six story figures, the environment
sweep, the follow-up campaigns, the HIL feedback study and the master table), what to run, where, with which
inputs, and where the output lands. The per-topic documents it points to hold the detail; nothing here
contradicts them. Method only.

Paths are relative to the repo root (`$REPO`, set by `scripts/env.sh`) unless written absolute. Absolute paths
under `results/` are records of what ran, not instructions.

---

## 0. Three places, three interpreters

`docs/Artifact/environment.md` §Environments is the one description of the two interpreters (`requirements-host.txt`,
`requirements-isaac.txt`) and of the machine-path variables. Every script here sources `scripts/env.sh`:

| where | what runs there | `env.sh` variable |
|---|---|---|
| **host** (the clone) | solver, calibration, trace reading, aggregation, figures, verifiers, master table | `REPO`; interpreter `HOST_PY` (default `$REPO/.venv/bin/python`: OR-Tools CP-SAT, numpy, matplotlib); results tree `RES=$REPO/results/codesign_feedback` |
| **simulator tree** | every Isaac flight (the IsaacLab source and assets live there; the campaign scripts `cd` into it) | `SIM_TREE` (the clone itself unless overridden; on the study machine `/scratch/agustin/projects/DIMA/XPU-RT`); interpreter `ISAAC_PY` (Isaac Sim 5.1 + IsaacLab + torch) |
| **board** SpaceMiT K1 (BananaPi M7, 8 RVV harts P0–3 / E4–7, `rdtime` 24 MHz) | XPU-RT tables and ROS 2 layouts, both on the same generated kernels | `ssh k1` (ssh-config entry; `MODELBLASTER_K1_HOST` overrides) |

Per-machine values go in `scripts/env.local.sh` (ignored; `scripts/env.local.sh.example` is the template) or in the
environment (`XPURT_REPO`, `XPURT_SIM_TREE`, `ISAAC_PY`, `HOST_PY`). The simulator sources the flights use
(`sims/scripts/sweep_rate_demo.py`, `sims/scripts/record_sensor_demo.py`,
`sims/training/{collect_fused_warehouse,train_fused}.py`, `sims/isaaclab_tasks/warehouse_nav/{mdp_gates,mdp_obstacles}.py`)
are byte-identical in the clone and the simulator tree when the two differ; edit in one, copy to the other.
Board access and toolchain: `docs/K1/k1_board.md` and `docs/K1/k1_modelblaster_xpurt_closed_loop.md` §0–2 (ssh, deploying the
board binaries). `eval "$(scripts/setup_spacemit_toolchain.sh)"` exports `CROSS` before anything that generates kernels.

Environment variables the host side reads on every solve (the scripts export them; listed so a manual solve
matches): `XPURT_CPSAT_PYTHON=$HOST_PY XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8`.

The GPU rule for the simulator: at most three Isaac processes on the 24 GB card. Every campaign script
gates each flight on `nvidia-smi` (free memory ≥ 8.5 GB and fewer than three compute processes,
`NEED_MB` / `MAX_SIMS` override). Each simulator process gets its own `TMPDIR` under its output directory.

---

## 1. Board, part A — the kernels and the XPU-RT tables

**Kernels.** Every network (`yolov8_nano_64x96`, `fused_full`, `mlp_control`, `ffn_block`, `dronet`) is
ModelBlaster-generated int8 C for `rvv_x60`, staged under `ModelBlaster/build/k1_xpurt/<net>/int8/`
(`graph.json` = the IR the scheduler and the trace reader both index, `weights.npz`, `io.npz`, `rvv_x60/`
sources, `ime_x60/` for the matrix-engine variant). Stage 1 of `ModelBlaster/scripts/run_xpurt_k1.sh`
generates them (extract graph → skeleton → kernels) and reuses them when present; `docs/K1/k1_modelblaster_xpurt_closed_loop.md`
§3–4 and §6 describe generation, verification against the golden output and the `CORE_KINDS` vs `--backends` distinction.

**Executing a table.** `scripts/run_xpurt_long.sh <schedule.json> <label> [replicates]`:

* resolves the networks the schedule names, passes their staged IRs (`--staged-ir <net>:ModelBlaster/build/k1_xpurt/<net>/int8`),
  starts the per-core sampler on the board (`/root/ros_mb/cpu_sampler 100 …`, ending any sampler left from an
  earlier run first), runs `run_xpurt_k1.sh --schedule … --backends ${BACKENDS:-rvv_x60,rvv_x60} --quant int8`
  under `SCHED_OTHER` (a FIFO run is a labelled variant, `MODELBLASTER_K1_RT_PRIORITY=80`; `HOGS=n` starts n
  unpinned CPU hogs first), retries when the board's sshd drops the copy (a complete trace = one row per
  scheduled dispatch);
* leaves in `results/codesign_feedback/xpurt_long/`: `trace_<label>_<policy>_run<k>.csv` (one row per
  executed dispatch: network, instance, hart, entry/exit ticks; `dispatch_id` is the kernel-call slot, joined
  to the IR through `k1_trace.ir_slot_map`), `cpu_<label>_…csv` (per-core busy %), `hart_acc_…csv`,
  `manifest_<label>_…json` (schedule, its sha256 prefix at run time, solver, policy, backends), `board_<label>_…log`.
* IME variant: `CORE_KINDS='rvv,ime,rvv_c1' BACKENDS='rvv_x60,ime_x60,rvv_x60'`.
* Two replicate axes, both in the file name: `_run<k>` is the slot of one `run_xpurt_long.sh` invocation (its
  `[replicates]` argument; the study's runs use one slot, so `_run1`), and `r<k>` at the end of the *label*
  (`acpsat_hardr2`) is the interleaved replicate `board_stage2.sh` launches — three separate invocations, so
  every replicate is its own trace/cpu/manifest set. `_fifo80_run1` marks the SCHED_FIFO variant of a label,
  `_hog2` a run with two CPU hogs.

**Trace-label families** (`xpurt_long/trace_<label>_…`), by how the table was made:

| labels | table | where described |
|---|---|---|
| `a`, `a30`, `a60`, `a90`, `a120`, `a150`, `a120h` + `{cpsat_hard,greedy}r<k>` | the chain specs solved by both solvers (§3, §8 step iii; `a` = `wh_chain45_solve`, the composite's rows) | §3, §8 |
| `ash{cpsat_hard,greedy}r<k>` | `wh_chain45_shard_solve`: YOLO free to shard, costed from `gen/mb_cal` | §3, §8 |
| `b`, `b5` + solver | `wh_chain90_rich_solve[_500]`, the heavier stack | §3, §7 |
| `fb<tag>r<round><solver>r<k>` | the HIL feedback study (rounds 0–2 per spec; `a120e` = execution-only calibration variant) | §7 |
| `best*`, `long*`, `tiled*`, `rich*`, `cam2*`, `coupled` | the earlier hand-built layouts (widths and pins chosen by hand, not solved); their numbers are `measured_timing.XPURT_POINTS`; drivers in `scripts/attic/` | `measurements_and_ablations.md` §1.5, `scripts/attic/README.md` |

**Reading a run.** `scripts/xpurt_trace_report.py --windows yolov8_nano_64x96=66.6667,fused_full=77.7778,mlp_control=10 --skip-ms 100 <trace.csv>`
prints per-frame YOLO and nav spans, camera→control (frame release → first control output after that
frame's nav) median/p95/max, control-output gaps, per-instance window misses (instance k released at
k·period; late if its last dispatch ends after release + window) and the per-hart accounting.

**Before any board run:** `ssh k1 'ps aux | grep -E "harness|ros_mb_chain" | grep -v grep'` must be empty
(`/proc/loadavg` has a permanent floor of 2.00 on this board and says nothing). Never compile on the board
while a run is in flight.

## 2. Board, part B — the ROS 2 layouts

`docs/Baselines/ros_baseline_reproduction.md` is the full recipe: §1 stages the same generated kernels on the board
(`/root/ros_mb/{yolo,nav,ctrl,pool,yolo4}`; the 4-hart YOLO is the same IR generated with `MB_SHARD_FACTOR=4`
and checked by `ModelBlaster/scripts/check_kernel_coverage.py`), §2 builds the traced node program
(`results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp`, plain / `-DMB_WITH_POOL` / rich
binaries), §3 defines every layout. The runs of this study:

```bash
scripts/board_vanilla.sh      # vanilla, rvanilla × 15–90 Hz × 3 replicates; vanilla + QoS 1 at 45 Hz
scripts/board_vanilla2.sh     # vanilla4, vanilla4t × 15–90 Hz × 3
scripts/board_rp3_90.sh       # rp3 at 90 Hz; rvanilla4 × 15–90 Hz × 3
scripts/board_vanilla4tm.sh   # vanilla4tm at 45 and 90 Hz × 3
RATES="45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4 1     # the QoS-depth-1 disclosure of the flights' baseline
RATES="45" scripts/ros_traced_matrix.sh vanilla4x2 1                     # two perception processes, pipelined by hand
scripts/board_campaign.sh; scripts/board_campaign2.sh                    # the tuned matrix (spin/smte/p3/p8/…), replicates
```

`scripts/ros_traced_matrix.sh <arm> [replicate]` (env `RATES`, `CTRL_HZ`, `QOS`, `HOGS`, `SUFFIX`, `SECS`; the
suffix in the run tag records the knob: `_q1` = QoS depth 1, `_c200` = 200 Hz control timer, `_hog2` = two CPU hogs,
`_smoke` = the 8 s smoke run — `docs/Baselines/ros_baseline_reproduction.md` §3)
runs one layout across camera rates with the sampler around every run and pulls
`results/codesign_feedback/ros_traced/<hz>_<arm><suffix>_r<k>/{trace,released,goals,consumed,ctrl_gaps,cpu}.csv + manifest.json`
(the manifest records executor, processes, affinity mask, policy, QoS, kernel and IR hashes).
`scripts/pull_ros_traced.py [tags…]` merges multi-process runs, drops the first 3 s, and writes
`chain.csv`, `summary.json` per run and `ros_traced/summary.csv` across runs (control-gap mean/p95/max,
camera→goal and camera→control medians, goals/s, frames dropped, per-core busy %).

## 3. Host — specs, costing, solves, and putting the tables on the board

**Specs** (`data/toplevel/`): the deployed chain camera → `yolov8_nano_64x96` → `fused_full` → `mlp_control`
(edges yolo→fused, fused→control; `chain_end_to_end_deadline_ms 80`), windows longer than periods so frames
may be in flight together, `machine_combination_mode shard`, profiled costs (`use_profiled`), `enable_impls false`.
Columns: horizon; per network camera rate / window ms / instances; networks free to shard.

| spec | horizon ms | networks | shard-free |
|---|---|---|---|
| `wh_chain45_solve` | 1000 | yolo 45 Hz / 66.7 / 45; fused 45 Hz / 77.8 / 45; control 100 Hz / 10 / 100 | none |
| `wh_chain45_solve_h200` | 200 | 9 / 9 / 20 (the hyperperiod certificate) | none |
| `wh_chain45_solve_soft` | 1000 | as `wh_chain45_solve`, solved on the soft path | none |
| `wh_chain45_shard_solve[_h200]` | 1000 / 200 | as above, YOLO free to shard (costed from `gen/mb_cal`, built by `scripts/calibrate_yolo_shard_profile.py`) | yolo |
| `wh_chain{30,60,90,120,150}_solve_500` | 500 | camera at that rate, same windows, half-second tables | none |
| `wh_chain{30,60,90,120,150}_solve_h200` | 200 | their certificates | none |
| `wh_chain{30,60,90}_solve` | 1000 | one-second tables | none |
| `wh_chain90_rich_solve_500` / `_h100` / `_soft` | 500 / 100 / 1000 | 90 Hz chain + `ffn_block` 10 Hz / 100 + `dronet` 30 Hz / 33.3 | ffn_block, dronet |
| `wh_chain90_solve_500_soft` | 500 | the private copy the soft path solves | none |

**Costing tables** (`--board-calibration`): `results/codesign_feedback/k1_board_calibration_yolo110.json`
(the stock per-op table with `per_dispatch_multiplier["yolov8_nano_64x96/<id>"] = 1.10`, YOLO measured
61/56 under load) is the study's default; `CAL=none` solves on the isolated profile alone (feedback round 0);
`hil_feedback/cal_<tag>_r<k>.json` are the service-time tables fitted from executed traces (§7).

**Solve one spec, both solvers** — `CAL=<table|none> scripts/solve_stage2_hard.sh <spec> <tag> [limit_s]`:
greedy (`--solver greedy_periodic --max-periodic-iters 1 --random-seed 42`), then hard-window CP-SAT
(`--solver cpsat --cpsat-time-limit <limit>`) with the soft-window CP-SAT (`--solver milp --scheduler cpsat`,
HEFT warm start, misses → lateness → makespan, on a private copy `<spec>_soft.json`) solved alongside
(`SOFT=0` skips it). The solver's own output files are removed before each solve, so a solve that returns
no table leaves no table. Outputs `schedules/fig_<tag>_{greedy,cpsat_hard,cpsat_soft}.json` (+`_metrics.json`),
logs `results/codesign_feedback/solver_v2/<spec>_{greedy,cpsat_hard,cpsat_soft}.log`.
Certificates on the hyperperiod specs: `scripts/solve_certificates.sh` (hard CP-SAT must be FEASIBLE/OPTIMAL;
greedy's predicted misses recorded first). Rate chains: `scripts/chain_rates2.sh` (30/60/90),
`scripts/chain_rates_high.sh` (120/150: certificate, tables, board), `scripts/chain_shard_solve.sh`.

**Execute on the board** — `scripts/board_stage2.sh <tag> <spec>`: waits for `STAGE2_DONE` in the solve log,
clamps widths to the codegen contract (`scripts/clamp_schedule_widths.py <table> <net>:<ir.json>… --out …_clamped.json`),
checks feasibility (`scripts/check_schedule_feasibility.py --schedule …` must exit 0: no double booking, no
dependency violation, every target on the board), waits for the board to be free, then runs greedy /
cpsat_hard / cpsat_soft (whichever tables exist) interleaved, three runs each, through `run_xpurt_long.sh`
with labels `<tag><solver>r<k>`. Read with `xpurt_trace_report.py --windows …` (§1).

**Pre-registered win criterion** (median of three runs, first 100 ms excluded): CP-SAT wins if fewer frames
end after release + window and no more control gaps above 15 ms than greedy; both reported with
camera→control p95 and predicted-vs-measured misses per arm.

## 4. Cadence traces — what the simulator replays

`scripts/ctrl_trace_from_board.py <xpurt_long/trace_<label>.csv | ros_traced/<tag>/ctrl_gaps.csv> --out results/codesign_feedback/ctrl_traces/<arm>.csv [--warmup-ms 100] [--max-s 6]`
writes the control-output times of one board run (XPU-RT: the end of each `mlp_control` instance's last
dispatch, looped over the table from warm-up to its end, silence included; ROS: every control fire) with the
source and gap statistics as `#` header lines. Every trace's header names its board run; the ones the
campaigns use:

| trace | board run | outputs/s |
|---|---|---|
| `xpu_a_cpsat_hard.csv`, `xpu_a_greedy.csv` | `xpurt_long/trace_acpsat_hardr1`, `trace_agreedyr1` (45 Hz chain, `fig_a_*_clamped`) | 100 (10.0 ms, p95 13.1) / bursty (2.2 ms mean, then silence) |
| `xpu_a90_cpsat.csv`, `xpu_a90_greedy.csv`, `xpu_a120h_cpsat.csv` | `trace_a90cpsat_hardr1`, `trace_a90greedyr1`, `trace_a120hcpsat_hardr1` | 100 / bursty / 100 |
| `ros_vanilla445.csv` | `ros_traced/45_vanilla4_r1` | 39 (25.7 ms) |
| `ros_vanilla4_q145.csv`, `ros_vanilla4t45.csv`, `ros_vanilla4x245.csv`, `ros_vanilla45.csv` | `45_vanilla4_q1_r1`, `45_vanilla4t_r1`, `45_vanilla4x2_r1`, `45_vanilla_r1` | 39 / 33 / 59 / 21 |
| `ros_vanilla4tm45.csv`, `ros_vanilla4tm90.csv`, `ros_p345.csv`, `ros_p390.csv`, `ros_p3_q145.csv` | `45_vanilla4tm_r1`, `90_vanilla4tm_r1`, `45_p3_r1`, `90_p3_r1`, `45_p3_q1_r1` | 100 each |

The camera→control latency an arm replays (`--percep_latency_ms`) and the goal rate it holds
(`--percep_hold_ms`) are the same run's medians from `ros_traced/summary.csv` / `xpurt_trace_report.py`
(e.g. vanilla4 242 ms, p3 55.8 ms, p3+QoS1 31.1 ms, CP-SAT 45 Hz 56.8 ms, 90 Hz 55.9 ms, 120 Hz 59.9 ms;
goal holds 30.3 ms for every ROS layout above 45 Hz, 11.1 / 8.3 ms for XPU-RT at 90 / 120 Hz).

## 5. Simulator — one flight, then the campaigns

**Scene.** Isaac warehouse aisle, gate course from `sims/isaaclab_tasks/warehouse_nav/mdp_gates.py`
(`WAREHOUSE_COURSE=a` the composite's course, `b` the unseen gates, `c` a third weave from a seeded generator,
`WAREHOUSE_COURSE_SEED` default 8); crates and pallets drawn per episode at `--prop_density`; walking people
2.4 m tall (`mdp_obstacles.PERSON_H`, `WAREHOUSE_PERSON_H` overrides; recorded per flight) on a north–south
patrol at 0.8 m/s (`--walk_speed` overrides, 0 = default); `--layout_seed` draws props and people from a fixed
seed independently of the episode seed (one scene for both arms). Sensors: one FPV camera + the ToF bank.
Guidance net `sims/models/warehouse/nav_fused_v12_cnn.pt` (§6), low-level controller
`rl_controller_velctrl_dr4.pt`, `--moment_scale` the control gain (0.0055 fixed policy; 0.5 / replayed rate
calibrated policy). A crash is the contact sensor (any contact above 1 N with anything); gates are credited
in order when the vehicle passes within the gate's pass radius; a flight succeeds after the last gate.

**One flight** (from the simulator tree; this is what every campaign cell runs, 12 seeds from 1000):

```bash
. scripts/env.sh; cd "$SIM_TREE"
$ISAAC_PY sims/scripts/sweep_rate_demo.py --headless \
  --controller rl --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sim_dt 0.01 --decimation 1 \
  --obstacle_level 8 --prop_density 0.30 --moment_scale 0.0055 --cruise_speed 1.0 --episodes 12 --seed 1000 --max_steps 1800 \
  --ctrl_trace <abs path>/ctrl_traces/xpu_a_cpsat_hard.csv [--percep_latency_ms 56.8] [--percep_hold_ms 0] \
  [--walk_speed 0] [--layout_seed 1005] --record_dir <out>/records/<cell> --sweep-csv <out>/campaign.csv
```

`--ctrl_trace` refreshes the held motor command only at control steps where the trace has an output
(restarting at every episode); `--percep_latency_ms` delays the navigation decision by the deployment's
measured camera→control latency; `--percep_hold_ms` refreshes the goal at the deployment's measured goal rate.
Each episode appends one row to the CSV (seed, speed, gain, trace, latency, hold, density, course, people
height, walk speed, outcome, gates, steps, crash type) and, with `--record_dir`, writes `ep<k>_s<seed>.npz`
(pose path, commanded wrench, body rates, navigation command, gates, outcome, every setting).
`sims/scripts/record_sensor_demo.py` (one recorded flight: chase + every model input + overhead path) shares the
flight flags (`--ctrl_trace`, `--layout_seed`, `--percep_latency_ms`, `--percep_hold_ms`, `--sched_latency_ms`,
`--save_video`, `--dump_figure_data`, `--gantt_schedule schedules/measured_gantt_<arm>.json` for the overlay,
`--post_success_steps`, `--moment_scale`, `--cruise_speed`, …) and adds `--keep_video` and `--keep_gates`; it has no
`--record_dir`, `--sweep-csv`, `--walk_speed`, `--walk_cross`, `--pipeline_zoh`, `--gust` or `--motor_tau` — those
are `sweep_rate_demo.py`'s. Both write `figure_data.npz` under `--dump_figure_data <dir>`.

**Campaigns** (each: one simulator per invocation, cells already in its CSV skipped, records kept):

| campaign | script | axes | output |
|---|---|---|---|
| replayed cadences vs speed | `ARMS="name:trace …" SPEEDS=… [GAIN=…] scripts/campaign_v2.sh` (`campaign_v2b.sh` keys cells by gain too) | arm × cruise | `campaign_v2/campaign_v2.csv` |
| unseen gates | `scripts/campaign_courseB_v2.sh` | course B | `campaign_v2_courseB/` |
| environment | `SPEEDS DENS COURSES GAIN scripts/env_sweep.sh <driver-id> "name:trace …"` | course a/b × density 0.20/0.30/0.40 × speed 1.0–1.8 | `campaign_env/env_sweep.csv`, `records/<arm>_<course>_d<dens>_c<cru>/` |
| latency + cadence + goal rate, follow-ups | `OUT=… ARMS="name:trace:latency_ms[:hold_ms] …" SPEEDS DENS COURSE WALK GAIN W scripts/campaign_percep.sh` | per arm | `<OUT>/campaign.csv`, `records/<tag>/` with tag `<name>_lat<l>_h<h>_<course>_d<dens>_w<walk>_g<gain>_c<cru>` |
| tuned ROS layouts, whole sweep | `scripts/sweep_tuned_ros.sh` (phases: course A 0.30; camera-rate arms; goal-rate holds; densities; courses b, c; faster people; calibrated gain) | as above | `campaign_percep/` |
| follow-ups queue | `scripts/queue_extra.sh` (energy v3 at 1.0 m/s + re-render; tuned phase 1; QoS-1 `campaign_qos1/`; calibrated gain in the tall scene `campaign_tallcal/`; course C `campaign_courseC/`; walking 1.5 m/s `campaign_walk/`) | | as named |
| heavier stack, extra seeds, crossing people, camera-rate arms | `scripts/queue_stronger.sh` (runner `campaign_percep2.sh`: `CROSS=1` = `--walk_cross`; `SEED0`) | rich table vs six-process vanilla; seeds 1012–1023; people crossing | `campaign_rich/`, `campaign_seeds24/`, `campaign_cross/` |
| calibrated gain | `scripts/campaign_v2_cal_now.sh` | 0.5 / replayed rate per arm | `campaign_v2/` |
| envelope grids | `scripts/hil_dense_grid.sh`, `scripts/gain_controlled_grid.sh`, `scripts/envelope_20hz.sh` | speed × rate (latency hold) × gain policy | `hil_ablation.csv`, `gain_controlled/gain_controlled.csv` |
| mechanism (power) | `CRUISE=1.0 ER=<dir> OUTCSV=<csv> scripts/run_energy_v2.sh` | 6 seeds per arm, wrench logged | `flight_energy_v{2,3}.csv`, `energy_runs_v{2,3}/` |

**Campaign CSVs** (one row per flight; the name says which driver wrote it):

| file | written by | rows |
|---|---|---|
| `campaign_v2/campaign_v2.csv`, `campaign_v2_courseB/campaign_v2.csv` | `campaign_v2.sh` / `campaign_v2b.sh` / `campaign_v2_cal_now.sh` (cadence-only replay, keyed by arm × cruise [× gain]) | the second-form flights (§1.7) |
| `campaign_env/env_sweep.csv` | `env_sweep.sh` (course × density × cruise) | the environment sweep (§1.8) |
| `campaign_<name>/campaign.csv` (`percep`, `qos1`, `tallcal`, `courseC`, `walk`, `tallnet`, `rich`, `seeds24`, `cross`, `break`, `break2`) | `campaign_percep.sh` / `campaign_percep2.sh` / `campaign_break*.sh` (latency + cadence + hold replay; `OUT=` names the directory) | the third-form flights |
| `campaign_scene/<cell>/campaign.csv` | `scene_runs.sh` (§8 step vi) | the 36 same-scene runs of a cell |
| `campaign/campaign.csv` | the first-form drivers (`scripts/attic/gpu_queue*.sh`), read by `campaign_select.py` and the `--xpu-arm/--ros-tag` verifier path | `warehouse_showdown_v2` |
| `hil_ablation.csv`, `gain_controlled/gain_controlled.csv` | the envelope grids | rate injected, no trace |
| `flight_energy_v{2,3}.csv` | `run_energy_v2.sh` | the mechanism flights |

`scripts/campaign_pipeline.py` (arms `xpu`/`ship`/`spin`/`cspin`/`cp3` with both the control cadence and the chain
delay from `measured_timing.derive()`, the RoSE-style free-running form) is an earlier driver of the same idea; it
wrote under `campaign_pipeline/` and is not on the path of any shipped figure — `campaign_percep.sh` is.

**The displayed pair.** `scripts/select_strong_cell.py` ranks every qualifying cell across all campaigns by XPU-RT's
completion count; `scripts/campaign_select_v2.py --xpu <trace> --ros <trace> [--policy fixed|calibrated|any]`
names the displayed cell by the rule written before the runs (fastest cruise where XPU-RT completes ≥ 3/12
while the baseline completes ≤ 1/12 and crashes after entering the course in ≥ half). One scene for both
flights: `CRUISE=1.0 scripts/display_same_env.sh` tries seeds in order (`--layout_seed = --seed`), keeps the
first seed where XPU-RT completes and the baseline crashes after one or two gates, logs every attempt, then
renders and verifies (§8). Output `campaign_v2/display_same/{xpu,ros}_s<seed>.mp4`, `…_figdata/`.
`scripts/display_v3.sh` is the earlier form (the campaign's seed sequence in one process).

**Three simulator slots, queued** (nothing stops; a chain hands its slot to the next):
slot 1 `scripts/slot1_percep.sh` (display pair → latency-replay cells → XPU-RT environment driver resumes);
slot 2 `scripts/env_sweep.sh 2 …` → `WAIT_PID=<driver pid> scripts/sweep_tuned_ros.sh`;
slot 3 `scripts/retrain_tall.sh` (§6) → `queue_extra.sh` → `campaign_v2_cal_now.sh` → envelope grids →
`env_sweep.sh 3` (pipelined / QoS-1 cells). `scripts/env_sweep_slots.sh` and `env_sweep_slot3.sh` are the
launchers. Waiting is on files or pids, never on a process-name pattern.

## 6. The guidance net

The shipped net `nav_fused_v12_cnn.pt` and its pipeline: `docs/Evaluation/warehouse_sensorfusion_reproduce.md` §3a
(expert demonstrations → `collect_fused_warehouse.py` → `train_fused.py --vision_encoder cnn`;
`train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt`). The tall-people variant
(`scripts/retrain_tall.sh`, weights `sims/models/warehouse/nav_fused_v20_tall_cnn.pt`), from the simulator tree:

```bash
. scripts/env.sh; cd "$SIM_TREE"; D=<train_out>/fused_bc_warehouse_tall; PY=$ISAAC_PY   # <train_out>: any writable directory outside the repo
# pilot (the planned expert with 2.4 m people): 8 episodes, seed 500
$PY sims/training/collect_fused_warehouse.py --headless --episodes 8  --max_steps 1500 --seed 500 --planned_expert --prop_density 0.30 --obstacle_level 8 --base_speed 1.4 --noise_std 0.06 --turn_slow --out $D/pilot_tall_d030.pt
# demonstrations: crowded aisle (60 episodes, seed 600) and the prop-free course (30 episodes, seed 700)
$PY sims/training/collect_fused_warehouse.py --headless --episodes 60 --max_steps 1500 --seed 600 --planned_expert --prop_density 0.30 --obstacle_level 8 --base_speed 1.4 --noise_std 0.06 --turn_slow --out $D/crowded_tall_d030.pt
$PY sims/training/collect_fused_warehouse.py --headless --episodes 30 --max_steps 1500 --seed 700 --planned_expert --prop_density 0.0  --obstacle_level 8 --base_speed 1.4 --noise_std 0.06 --turn_slow --out $D/gate_tall_d000.pt
# clean filter: crowded episodes reaching >= 3 gates (retrain_tall.sh, inline python) -> crowded_tall_d030_clean.pt
$PY sims/training/train_fused.py --data $D/crowded_tall_d030_clean.pt $D/gate_tall_d000.pt --vision_encoder cnn --epochs 60 --out_dir <train_out>/fused_bc_warehouse_v20_tall_cnn
# best.pt of the newest run -> sims/models/warehouse/nav_fused_v20_tall_cnn.pt (both trees); flown with W=<weights> campaign_percep.sh -> campaign_tallnet/
```

## 7. The HIL feedback study

`ROUNDS="0 1 2" LIMIT=3000 scripts/hil_feedback_study.sh` (board + host, `results/codesign_feedback/hil_feedback/study.log`),
per spec `a90:wh_chain90_solve_500`, `a120h:wh_chain120_solve_h200`, `b5:wh_chain90_rich_solve_500` (instance 1) and
`a90h:wh_chain90_solve_h200`, then `a90` at `LIMIT=9000` (instance 2, `study2.log`; the two serialise on `results/codesign_feedback/board.lock`):

1. round 0: `CAL=none solve_stage2_hard.sh <spec> fb<tag>r0` (isolated profile, no board knowledge) → `board_stage2.sh`;
2. round k ≥ 1: `scripts/calibration_from_executed.py --schedule schedules/fig_fb<tag>r<k-1>_<cpsat>_clamped.json --trace-glob 'xpurt_long/trace_fb<tag>r<k-1><cpsat>r*_other_run1.csv' --out hil_feedback/cal_<tag>_r<k>.json`
   (per-dispatch service time on its hart = execution + the unplanned idle that followed it, median of
   service/predicted over the warm instances of the three runs, keyed by IR dispatch id through
   `k1_trace.ir_slot_map`; an op-kind tier for dispatches under the 0.05 ms floor; aggregate = ratio of sums;
   `--execution-only` for execution alone) → `CAL=<that table> solve_stage2_hard.sh` → `board_stage2.sh`.
   The round's CP-SAT table is the hard-window one; when the hard certificate is not found within the limit
   the soft-window table solved alongside stands in, the log says which, and the next round calibrates from it.
   Calibration statistic variants: `scripts/hil_feedback_study_cal.sh` with `CALARGS="--execution-only --stat mean"` (tag `a120e`,
   round 0 linked to `a120h`'s tables and traces, `study3.log`).
3. figure: `scripts/hil_feedback_figure.py --tag <tag> --spec <spec>` → `refined/hil_feedback_<tag>.png` (+ `_metrics.json`);
4. per round and solver: the solver's prediction (`schedules/fig_fb<tag>r<k>_<solver>_metrics.json`), the
   executed camera→control, control gaps, frames late and worst lateness (`xpurt_trace_report.py --windows`),
   per-hart drift (`calibration_from_executed.py` output). The master table (§8) carries all of them.

## 8. Aggregation, figures, verification, the master table

In dependency order; every step reads only what the steps above it wrote. Host side (`$HOST_PY`) unless marked
**K1** or **GPU**. `. scripts/env.sh` first.

**(i) Specs — `data/toplevel/wh_chain*.json`.** Hand-written inputs, tracked (the table in §3). Each carries a
`hardware` block (`machines {cpu_p: 4, cpu_e: 4}`, `profile_hw rvv_x60`, `profile.gen_root` = the profile database
the costs come from: `gen/mb`, or `gen/mb_cal` for the shard specs), a `scheduler` block (`use_profiled`,
`machine_combination_mode shard`, `shard_only_networks`, `enable_impls false`, `random_seed`, time limits), the
`networks` (id, IR `dispatch_deps_path`, camera rate, window, instances), the `edges` yolo→fused→control, the
`chain` with `chain_end_to_end_deadline_ms 80` and its rationale, and `horizon_ms`. `wh_chain45_shard_solve*`
differs from `wh_chain45_solve*` in exactly two fields: `gen_root gen/mb_cal` and
`shard_only_networks [yolov8_nano_64x96]`. The `<spec>_soft.json` copies are written by `solve_stage2_hard.sh`
(a private copy so the soft-window solve's output name differs from the hard path's) and are tracked because the
soft tables cite them.

**(ii) Costing tables.**
* `gen/mb_cal/` (tracked): `scripts/calibrate_yolo_shard_profile.py --w1 <1-hart trace> --w2 <2-hart trace> --w4 <4-hart trace>`
  copies the profile database and sets YOLO's `topo_0` / `topo_0_1` / `topo_0_1_2_3` rows to the median wall time
  per dispatch over the warm instances of the named executed traces (the shard harness times each shard's work, the
  trace times the sharded dispatch); `results.orig.csv` and `results.provenance.json` sit beside the rows.
* `results/codesign_feedback/k1_board_calibration.json`:
  `scripts/emit_board_calibration.py --trace-glob 'results/k1_feedback_exact/board_runs*/[of]*_trace.csv' --out <table>`
  — per dispatch, actual/predicted from the trace's own columns (`rdtime` span against `predicted_duration_ms`),
  execution inflation only (queue delay excluded), mean at a 0.1 ms floor for the pooled op tier and the aggregate,
  every sample for the per-dispatch tier; `--validate-against <table>` diffs a rebuild against the committed one,
  `--stat median`, `--min-samples`, `--schedule <table>` (checks the dispatch numbering), `--workload`, `--tol`.
  Three tiers in `profile_loader._board_calibration_mult` order: `per_dispatch_multiplier["net/id"]`,
  `per_op_multiplier[op]`, `aggregate_multiplier`; `coverage.nets_exact` names the measured nets.
* `k1_board_calibration_yolo110.json` — the study's default (`CAL` in `solve_stage2_hard.sh`): the table above with
  `per_dispatch_multiplier["yolov8_nano_64x96/<0..97>"] = 1.10`, the detector's measured per-frame span under load
  against its profile (61 / 56 ms); the table's `note` field records the derivation. `_yolo100` and `_yolo120` are
  the sensitivity variants (`scripts/solve_certificates_sensitivity.sh`; `_yolo100` also costs the shard spec, whose
  YOLO rows are board-measured already).

**(iii) Solves — `schedules/fig_<tag>_{greedy,cpsat_hard,cpsat_soft}.json`.** `CAL=<table|none> scripts/solve_stage2_hard.sh <spec> <tag> [limit_s]`
(§3). Tag → spec:

| tag | spec | driver |
|---|---|---|
| `a` | `wh_chain45_solve` — the 45 Hz chain; `fig_a_cpsat_hard_clamped` / `fig_a_greedy_clamped` are the composite's XPU-RT rows and the tables `xpu_a_*.csv` replay | `solve_stage2_hard.sh wh_chain45_solve a 3000`, then **K1** `board_stage2.sh a wh_chain45_solve` (run by hand; logs `solver_v2/stage2_a.log`, `board_stage2_a.log`) |
| `a30`, `a60`, `a90` | `wh_chain{30,60,90}_solve_500` (certificate on `_h200` first) | `scripts/chain_rates2.sh` |
| `a120`, `a150` / `a120h` | `wh_chain{120,150}_solve_500` / `wh_chain120_solve_h200` | `scripts/chain_rates_high.sh` / `chain_rates_high_h200.sh` |
| `ash` | `wh_chain45_shard_solve` (`CAL=_yolo100`) | `scripts/chain_shard_solve.sh` |
| `b5` (`b`) | `wh_chain90_rich_solve_500` (`wh_chain90_rich_solve`) | `chain_rates2.sh` (`scripts/attic/stage2_b_chain.sh`) |
| `fb<tag>r<k>` | the feedback study's rounds (§7) | `scripts/hil_feedback_study.sh` |

**(iv) Board runs — K1.** `scripts/board_stage2.sh <tag> <spec>` clamps each table to the codegen contract
(`schedules/fig_<tag>_<solver>_clamped.json` — the file the board executes), checks feasibility and runs three
interleaved replicates per solver through `run_xpurt_long.sh`, labels `<tag><solver>r<k>` (§1, §3). ROS 2 layouts:
`ros_traced_matrix.sh` and the `board_*.sh` drivers, then `pull_ros_traced.py` (§2). Every XPU-RT manifest names the
executed table and its sha256 prefix at run time; `scripts/executed_tables.py --write` builds the ledger
`results/codesign_feedback/executed_tables.json` from the manifests and git history (per run: the table file, its
hash at run time, the sha256 of the canonical `dispatches` block, and — where a table's metadata was annotated after
its run, as the `fig_a_*`/`fig_b5_*` clamped tables were given `solver`, `solver_status` and `solved_from` in commit
`a75e9c21` — the annotation record with `dispatches_identical: true`). `scripts/executed_tables.py` checks every entry
against the files on disk; the verifier accepts a Gantt row's table when its dispatch hash equals the ledger's.

**(v) Cadence traces.** `scripts/ctrl_trace_from_board.py` for every arm (§4 table), including the two the third form
adds:
```bash
scripts/ctrl_trace_from_board.py results/codesign_feedback/ros_traced/90_vanilla4x2_r1/ctrl_gaps.csv --out results/codesign_feedback/ctrl_traces/ros_vanilla4x290.csv --warmup-ms 3000
scripts/ctrl_trace_from_board.py results/codesign_feedback/ros_traced/45_multi_r1/ctrl_gaps.csv --out results/codesign_feedback/ctrl_traces/ros_multi45.csv --warmup-ms 3000
```
The launch latency and hold each trace is flown with are the registry `scripts/figure_constants.py` (one `ReplayArm`
per trace: the values used as the campaign-CSV key and the pointer into `measured_timing` that re-derives them).

**(vi) Campaigns — GPU.** The §5 table (three simulator slots), then the same-scene runs and the queues:
```bash
CELL=<cell> LAYOUT_SEED=<seed> CRUISE=<m/s> [XGAIN=0.0055 RGAIN=0.0055 XLAT=56.8 RLAT=242 GLAT=748 PERSON_H=2.4 DENS=0.30 EPISODES=12] scripts/scene_runs.sh
#   -> campaign_scene/<cell>/{xpu_cpsat,ros_vanilla,xpu_greedy}/ep*.npz + campaign.csv: twelve flights per arm in ONE layout
#      (props and people from LAYOUT_SEED), each arm replaying its cadence and camera->control latency; panels K/H (v3, atlas), H/H' (final), S (paper10)
#      cells on disk: tall1005s (seed 1005, 1.0 m/s), tall1000s (seed 1000, 1.4 m/s), cal17
nohup bash scripts/queue_v3.sh > results/codesign_feedback/queue_v3.log 2>&1 &     # display pairs (ROS_GATES=2), scene runs, the eight-core ROS 2 flight arms, renders; prints QUEUE_V3_DONE
for q in v3c v3d v3b v3e v3f; do nohup bash scripts/queue_$q.sh > results/codesign_feedback/queue_$q.log 2>&1 & done   # each waits on the previous marker: pair search (baseline first) -> eight-core arms + renders -> replicates -> more pair searches
nohup bash scripts/campaign_break.sh > results/codesign_feedback/campaign_break.log 2>&1 &         # the breaking point: people 1.7 m, both arms replaying cadence AND latency, 0.8-2.0 m/s; prints CAMPAIGN_BREAK_DONE
nohup bash scripts/campaign_break_ros.sh > results/codesign_feedback/campaign_break_ros.log 2>&1 & # the ROS 2 half of the same campaign (shared CSV; flown cells skipped)
nohup bash scripts/campaign_break_seeds.sh > results/codesign_feedback/campaign_break_seeds.log 2>&1 & # after CAMPAIGN_BREAK_ROS_DONE: seeds 1012-1023 for every cell of both arms
nohup bash scripts/campaign_break2.sh > results/codesign_feedback/campaign_break2.log 2>&1 &       # after CAMPAIGN_BREAK_DONE: density 0.40, and the hand-pinned ROS 2 at 0.30
```
The display pair's videos: `display_same_env.sh` / `record_display_videos.sh` run `record_sensor_demo.py --save_video
<dir>/<arm>_s<seed>.mp4 --dump_figure_data <dir>/<arm>_s<seed>_figdata` (1800 × 1340, 50 fps), so the composite and the
video are the same flight. `refined/warehouse_showdown_final_pair_s1005.mp4` is `campaign_v2/display_same/xpu_s1005.mp4`
and `ros_s1005.mp4` side by side (`scripts/compose_pair_video.sh <xpu.mp4> <ros.mp4> <out.mp4> [left banner] [right banner]`, which also writes the `_half` copy; ffmpeg `hstack`, XPU-RT left, ROS 2 right; the baseline clip ends at its crash and
holds its last frame to the XPU-RT flight's 1596 frames, 3600 × 1340); `_half.mp4` is the same at 1800 × 670. The two
pair videos are tracked; every other mp4 is not (§9).

**(vii) The measured Gantt rows** — `scripts/make_measured_gantt_pair.py`, run first by `render_showdown_v3.sh` with the four arms
(`XL=results/codesign_feedback/xpurt_long`, `RT=…/ros_traced`):
```bash
--arm xpu:xpu:$XL/trace_acpsat_hardr1_other_run1.csv:$XL/cpu_acpsat_hardr1_other_run1.csv:$XL/manifest_acpsat_hardr1_other_run1.json:schedules/fig_a_cpsat_hard_clamped.json
--arm xpu2:xpu:$XL/trace_agreedyr1_other_run1.csv:$XL/cpu_agreedyr1_other_run1.csv:$XL/manifest_agreedyr1_other_run1.json:schedules/fig_a_greedy_clamped.json
--arm ros8:ros:$RT/45_vanilla4x2_r1/trace.csv:$RT/45_vanilla4x2_r1/cpu.csv:$RT/45_vanilla4x2_r1/manifest.json
--arm ros:ros:$RT/45_vanilla4_r1/trace.csv:$RT/45_vanilla4_r1/cpu.csv:$RT/45_vanilla4_r1/manifest.json
--window-ms 100 --skip-ms 400 --spec data/toplevel/wh_chain45_solve.json --out-prefix schedules/measured_gantt_v3
```
→ `schedules/measured_gantt_v3_{xpu,xpu2,ros8,ros}.json` (+ `_metrics.json`; tracked). `ros8` is `45_vanilla4x2`, the
two-perception-process graph on all eight cores; `ros` the vanilla 4-hart graph. Each row's busy % is the per-core
sampler (`cpu.csv`) aligned by `rdtime` to the window — the same instrument on the XPU-RT and ROS 2 rows; the
XPU-RT harness's own per-hart kernel fraction is recorded beside it in the sidecar (`kernel_frac_pct`), and
`busy_source` names which one the figure draws. Each sidecar names its trace, manifest and table, and the table's
dispatch hash is checked against the ledger (step iv).

**(viii) Renders** — in this order (each later figure imports panel functions from the earlier scripts and reads the
same inputs; `XPURT_FIG_ALLOW_FALLBACK` unset, so a missing measurement raises `MissingMeasurement` instead of being
substituted):
```bash
CELL=tall1005 TUNED=all MAIN=1 scripts/render_showdown_v3.sh   # refined/warehouse_showdown_v3_<cell>_<tuned>.{png,pdf,_metrics.json} + hil_envelope_story_v3_<cell>_board (companion) + _numbers.tex
scripts/showdown_atlas.py                                      # refined/warehouse_showdown_atlas.{png,pdf,_metrics.json}
scripts/story_figures.py [names]                               # refined/{crash_position,course_progress,rate_speed_map,ros_ladder,seed_pairs,latency_waterfall}.{png,pdf,_metrics.json}
scripts/showdown_final_figure.py [--dpi 60]                    # refined/warehouse_showdown_final.{png,pdf,_metrics.json}
scripts/showdown_paper10_figure.py --audit                     # refined/warehouse_showdown_paper10.{pdf,png,_metrics.json}; --audit = layout acceptance (exit 3 on findings)
scripts/showdown_paper_figure.py                                                       # host: refined/showdown_45hz_pinned_vs_rosdefault_s1003.{pdf,png} — the paper layout with every number measured (figure_runbook §3b item 11b)
```
`render_showdown_v3.sh` environment: `CELL` = which display pair (`tall1005`, default: seed 1005 at 1.0 m/s,
`campaign_v2/display_same/{xpu,ros}_s1005_figdata`; `tall1000`: seed 1000 at 1.4 m/s,
`campaign_v2/display_v3s_c1.4/{xpu,ros}_s1000_figdata`; `cal17`: the calibrated-gain cell, whose scene runs exist but
whose display dump is not on disk); `TUNED` = where the hand-pinned ROS 2 arm appears (`board` | `inb` | `none` |
`all` = the three renders); `PAPER=1` renders the paper-form skeleton (`--paper-form`, suffix `_paper`);
`MAIN=1` copies the cell's `board` render to the unsuffixed `warehouse_showdown_v3*` and `hil_envelope_story_v3*`
(`MAIN_CELL`, default `tall1005`, does so without the flag); `DPI` (300), `PY` (the host interpreter); `XPU_DIR`,
`ROS_DIR`, `SCENE_RECORDS` override the cell's dumps and same-scene records; `DISPLAY_CRUISE` the pair's cruise
(from the dump when unset); `ENERGY_CSV` the mechanism panel's table (`flight_energy_v2.csv`). Panel contents and
the story/atlas/final/paper10 panel lists: `docs/Evaluation/figure_runbook.md` §3b items 7–11; method and results:
`docs/Evaluation/measurements_and_ablations.md` §1.8c–h.

Other figures and tables from the same inputs:
```bash
scripts/env_sweep_summary.py            # campaign_env/env_sweep_summary.csv, refined/env_sweep.png
scripts/env_crash_map.py --course a --density 0.30   # refined/env_crash_map_a_d0.30.png
scripts/render_gantt_compare.py         # refined/gantt_compare_v2.png (measured rows CP-SAT / greedy / ROS 2 vanilla)
ENERGY_CSV=… scripts/hil_story_figure.py                # hil_envelope_story.{png,pdf}
scripts/plot_deployment_layers.py; scripts/plot_throughput_latency.py   # refined/deployment_layers, refined/throughput_latency
scripts/hil_feedback_figure.py --tag <tag> --spec <spec>                # refined/hil_feedback_<tag>.png (§7)
scripts/build_master_csv.py             # master_runs.csv + master_configs.csv (every board run, solve, flight, energy flight); scripts/refresh_master_csv.sh rebuilds it every 30 min while campaigns run
# the second form (figure_runbook §3b item 6): panel I rows from the named traces, verified by the --xpu-arm/--ros-tag path
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json \
  XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 \
  XPU_DIR=<display>/xpu_s1005_figdata ROS_DIR=<display>/ros_s1005_figdata ENERGY_CSV=$RES/flight_energy_v3.csv DPI=300 \
  scripts/render_showdown_measured.sh results/codesign_feedback/refined/warehouse_showdown_v2
```

**(ix) Verification** — all four before any figure is committed:
```bash
$HOST_PY scripts/verify_showdown_figure.py --all      # 0 FAIL
$HOST_PY scripts/measured_timing.py --verify          # 0 drift; figure_constants.check_registry() empty
$HOST_PY scripts/executed_tables.py                   # every ledger entry matches the table on disk
$HOST_PY -m pytest tests xpu-rt/tests -q              # tests/test_showdown_figures.py pins the counting rules
```
`verify_showdown_figure.py --all` re-derives every `refined/*_metrics.json` sidecar — the v3 composite in every cell
and placement, the atlas, the final figure, paper10 and the six story figures — from the campaign CSVs, records,
traces and manifests under the rules the figure scripts use (`flight_cells` cell selection, timeout censoring and
paired seeds `censor_and_pair`, the equal-replicate rule `equalise`, one flight per seed; the 2.4 m panels pool the
2.4 m scene only, `person_h`), the chain medians of the waterfall from the traces (sidecar `chain_median_ms` beside
`sum_of_part_medians_ms`: the label is the chain's median, the bars are component medians), the forest rows and
summary, and the Gantt rows' provenance; it requires `fallbacks_used` to be empty, each sidecar's `inputs` sha256 to
match the files on disk, every display literal ("57 ms", "39 Hz", …) to come from the registry, and every png/pdf in
`refined/` to have a sidecar or be on its allowlist (`REFINED_ALLOWLIST`: `env_crash_map_*`, `env_sweep`,
`hil_envelope_story_v3*`, `deployment_layers`, `throughput_latency`, `gantt_compare_v2`, `cores_yolo_service`,
`warehouse_gatecourse`, `warehouse_showdown_envelope`, `warehouse_showdown_v2`, `hil_feedback_*`,
`schedule_evolution_*` — figures with named producers above and in `figure_runbook.md`, except
`cores_yolo_service`, whose producer `fig_fair_v6.py` lives in a second checkout under
`XPU-RT/results/codesign_feedback/refined_src/` and is tracked by neither repo; see
`figure_verification_inventory.md` §4 for which of its rows re-derive). `--metrics <sidecar>` checks
one figure (`--v3-metrics` is kept as an alias); `--xpu-arm <label> --ros-tag <tag> --xpu-dir … --ros-dir …` is the
second form's path. `measured_timing.py --verify` recomputes every board constant from its trace and runs
`figure_constants.check_registry()` (each replay arm's launch latency and hold against its re-derivation).

## 9. Where everything lands

```
results/codesign_feedback/
  xpurt_long/                 XPU-RT board runs: trace_/cpu_/hart_acc_/manifest_/board_<label>_<policy>_run<k>          tracked
  ros_traced/<hz>_<arm>[_suffix]_r<k>/  ROS 2 board runs (+ summary.csv across runs; yolo_standalone/)                 tracked
  solver_v2/                  solve and board logs per spec and tag (stage2_<tag>.log, board_stage2_<tag>.log)            tracked
  executed_tables.json        the ledger of executed tables (file, hash at run time, dispatch hash, annotation)           tracked
  ctrl_traces/                cadence traces (header = source run + gap statistics)                                        tracked
  k1_board_calibration*.json  costing tables (base, _yolo100/110/120)                                                    tracked
  campaign_v2/                cadence-vs-speed campaign (campaign_v2.csv), display pairs (display_same/, display_v3s_c*/)  CSV tracked; figure_data.npz, frames/, mp4 not
  campaign_v2_courseB/        the unseen course (campaign_v2.csv)                                                          CSV tracked
  campaign_env/               environment sweep (env_sweep.csv, env_sweep_summary.csv, records/)                          CSVs tracked; records not
  campaign_percep/            latency + cadence + goal-rate cells and the tuned ROS sweep (campaign.csv, records/)         CSV tracked; records not
  campaign_{qos1,tallcal,courseC,walk,tallnet,rich,seeds24,cross,break,break2}/   the follow-ups (same layout)             CSV tracked; records not
  campaign_scene/<cell>/      the same-scene runs: campaign.csv + <arm>/ep*.npz (panels K/H, H/H', S)                     CSV tracked; records archived
  campaign/, campaign_pipeline/   the first-form campaign and the earlier pipeline driver (drivers in scripts/attic/)      campaign.csv tracked
  energy_runs_v2/ energy_runs_v3/ flight_energy_v{2,3}.csv     mechanism flights and rotor-model totals                    CSVs tracked
  hil_ablation.csv gain_controlled/    the flight envelopes                                                                 tracked
  hil_feedback/               calibration tables per round, study.log                                                       tracked
  master_runs.csv master_configs.csv   the master table                                                                     tracked
  refined/                    the figures: every png/pdf with its _metrics.json sidecar (or on the verifier's allowlist),
                              *_numbers.tex, the two pair videos warehouse_showdown_final_pair_s1005[_half].mp4              tracked
  archive_v3/MANIFEST.sha256  sha256 + size of the three archives below                                                     tracked
  archive_v3/*.tar*           display_dumps_v3.tar (the two display pairs' figure_data + frames: panels A, a-d, S, H, H'),
                              scene_records_v3.tar (campaign_scene records: panels K, H and the scene maps),
                              logs_2026-09.tar.zst (the campaign and queue driver logs)                                    NOT tracked (archive)
schedules/                    fig_<tag>_<solver>[_clamped|_metrics].json, measured_gantt_v3_<row>.json (+_metrics)          tracked
data/toplevel/wh_chain*.json  the specs (+ the _soft copies)                                                                tracked
gen/mb_cal/                   the board-measured YOLO shard profile (results.orig.csv, results.provenance.json)              tracked
sims/models/warehouse/        the weights the flights use (README.md = provenance)                                          tracked
requirements-host.txt requirements-isaac.txt   the two interpreters (docs/Artifact/environment.md)                                  tracked
scripts/env.sh, scripts/env.local.sh.example   machine paths (env.local.sh itself is ignored)                              tracked
scripts/attic/, sims/scripts/attic/            one-shot drivers and retired tools, with README indexes; the documented
                                               path is everything outside attic/                                           tracked
```

Tracked: board traces, manifests and stdout, campaign CSVs, schedules, sidecars, figures, the pair videos, the
archive manifest. Not tracked (archive or regenerate): flight dumps (`figure_data.npz`, `frames/`), per-episode
`records/**/*.npz`, every other `.mp4`, driver logs, `tmp/` and IsaacLab scratch, lock and pid files, `scripts/env.local.sh`.
Everything a figure needs that is not tracked is in `archive_v3/` (hosting is decided at publication; the clean-clone
test uses the local archive — untar, check against `MANIFEST.sha256`): panels A / a–d / S / H / H′ need the display
dumps, panels K and the scene maps need the scene records; everything else renders from committed inputs. The
recorder is non-deterministic, so a re-flight gives an equivalent but not identical flight. The archived copy of a
displayed pair is the one the figure was drawn from, and that is checkable without a re-flight: the
figure's sidecar hashes each dump it read, so

```bash
tar -xf results/codesign_feedback/archive_v3/display_dumps_v3.tar -C <tmp> \
    display_same/xpu_s1005_figdata/figure_data.npz display_same/ros_s1005_figdata/figure_data.npz
sha256sum <tmp>/display_same/*/figure_data.npz           # against `inputs` in the figure's _metrics.json
```

matches byte for byte for the pair the paper figure draws.

## 10. From scratch, in order

1. Host venv, simulator env, board access and toolchain (§0, `docs/Artifact/environment.md`); deploy board binaries; stage
   and verify the kernels (§1, §2 step 1); build the traced ROS node (§2).
2. Board: the ROS layouts (§2 scripts) → `pull_ros_traced.py`; certificates and tables for the specs
   (§3) → `board_stage2.sh` per tag; the rate chains.
3. `ctrl_trace_from_board.py` for every arm (§4); latencies and goal rates from `ros_traced/summary.csv`
   and `xpurt_trace_report.py`.
4. Simulator: the campaigns (§5) in three slots; the guidance-net retraining if wanted (§6); the display
   pair; the energy flights.
5. The feedback study (§7), board and host, whenever the board is free of the ROS sweeps.
6. Aggregate, render, verify, rebuild the master table (§8 in its order: specs → costing → solves → board →
   cadence traces → campaigns and scene runs → Gantt rows → renders); `verify_showdown_figure.py --all` 0 FAIL,
   `measured_timing.py --verify` 0 drift, `executed_tables.py` clean, tests green.
