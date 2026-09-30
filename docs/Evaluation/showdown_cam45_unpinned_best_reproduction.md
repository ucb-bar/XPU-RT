# The warehouse showdown at a 45 Hz camera, on the solver-placed schedule

This page reproduces
`results/codesign_feedback/refined/warehouse_showdown_cam45_unpinned_best_v2.{png,pdf}`.

It is the same layout and the same measured ROS 2 baseline as
[`showdown_cam45_ros_unpinned_reproduction.md`](showdown_cam45_ros_unpinned_reproduction.md), with
three quantities changed: the XPU-RT arm is `p45free` rather than `a_cpsat_hard`, both flights are
drawn at **one episode seed**, and panel D normalises to the arm the figure actually draws as 1×.
The earlier figure and its dumps are left where they are; this is a separate stem.

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md),
[`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) for building and deploying the
baseline, [`showdown_cam30_solver_placed_reproduction.md`](showdown_cam30_solver_placed_reproduction.md)
for the 30 Hz form.

---

## 1. The two arms

Both run the **same generated ModelBlaster kernels** for the same three networks on the same eight
harts of the same SpacemiT K1; the verifier checks that both board runs carry the same staged YOLO
IR, `ir=0c783539626c`. Only the orchestrator differs.

| | XPU-RT `p45free` | ROS 2 `vanilla445` |
|---|---|---|
| arrangement | CP-SAT schedule, **placement chosen by the solver**, control in its own slot | 4 processes, single executor, **unpinned**, control chained to the goal |
| board run | `xpurt_long/trace_p45freer1_other_run1.csv` | `ros_traced/45_vanilla4_r1/` |
| camera→control | **27.45 ms** (window) · 28.30 ms (run) | **242.10 ms** |
| control cadence | 9.89 ms (≈100 Hz) | 25.96 ms (39 Hz) |
| frames late | 0 of 21 | **760 of 766** |
| cadence trace replayed | `ctrl_traces/xpu_p45free.csv` | `ctrl_traces/ros_vanilla445.csv` |

**Read the baseline for what it is.** It is ROS 2 as normally written — one process per node, default
executor, nothing pinned — and 99.2 % of its frames are late. It is the representative default, not a
crippled build; the pinned arms (`p3`, `cp3`) and the arrangement that uses all eight harts
(`vanilla4x2`) are measured separately and are the subject of their own figures. The caption must say
"default, unpinned" so the row is not read as the best ROS 2 can do.

**Board-measured timing, simulated flight consequence**: the drone never leaves Isaac Lab. Each
flight replays its arm's measured control-output series (`--ctrl_trace`) and its measured
camera→control latency (`--percep_latency_ms`). This form uses no goal hold (`percep_hold_ms = 0`),
because the 45 Hz baselines were all flown without one.

---

## 2. Where the XPU-RT placement comes from

`data/toplevel/wh_chain45_free.json` carries **no placement instruction at all**: no
`allowed_machines`, no `machine_width`, no `preferred_hw`, and no pin, affinity, core or hart key.
Each network declares only `dispatch_deps_path, id, identifier, num_instances, period,
window_duration` (`stateful` additionally for nav). The solver is given the hardware's capability
statement — `smt.vmadot` legal on cluster 0 only — and the measured per-width cost tables, and
derives the placement itself: all three networks across all eight harts of both clusters, YOLO mostly
four-wide, nav mostly one-wide, control always one-wide, and 1012 dispatches on the IME.

```bash
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain45_free.json \
    --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 3600 --use-profiled --board-calibration $CAL
REPS=3 scripts/board_partitioned30.sh \
    schedules/scheduled_wh_chain45_free_cpsat_profiled.json p45free gen/mb_shard_nav
.venv/bin/python scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/xpurt_long/trace_p45freer1_other_run1.csv \
    --out results/codesign_feedback/ctrl_traces/xpu_p45free.csv
```

The warm-up is the default 100 ms here, not the 3 s the ROS arms use. A `ros_traced` run is about
20 s and its executor needs that long to settle; an `xpurt_long` run is under a second end to end, so
a 3 s warm-up leaves nothing — `ctrl_trace_from_board.py` exits 1 with `only 0 gaps after warm-up` and
writes no file. The committed `xpu_p45free.csv` reproduces byte for byte at the default.

### 2.1 How much of the 27.45 ms is reproducible

`xpu-rt/cpsat_scheduler.py` states that CP-SAT is reproducible only with `workers=1`: with several
search workers the result depends on thread interleaving, measured spread about ±1.5 ms on a 46 ms
schedule. `XPURT_CPSAT_WORKERS` defaults to 8, and the table the board ran came from an 8-worker
solve. `scripts/solve45_determinism.sh` asks what one worker gives, on two copies of the spec that
differ only in `random_seed`:

```bash
scripts/solve45_determinism.sh        # specs data/toplevel/wh_chain45_free_det{1,2}.json, seeds 42 and 7
scripts/board_det45.sh                # the deterministic table through codegen and onto the K1 as p45det
```

| | makespan | critical path | deadline misses | IME | oracle gap | solve |
|---|---|---|---|---|---|---|
| 8 workers, seed 42 | 490.083 ms | 105.056 ms | 0 | 1012 | 1.229 % | 265 s |
| 1 worker, seed 42 | 490.083 ms | 105.061 ms | 0 | 1012 | 1.229 % | 184 s |
| 1 worker, seed 7 | 490.083 ms | 105.061 ms | 0 | 1012 | 1.229 % | 186 s |

The two single-worker tables are **identical dispatch for dispatch** — machine *and* start time,
2836 of 2836 — so with one worker the answer does not depend on the seed either. Single-worker was
also faster here.

They differ from the 8-worker table only in *width mix*: YOLO two-wide on 500 dispatches against
389, nav two-wide on 44 against 25, reaching the same makespan by a different route. Width is what
codegen builds, and on the board that difference is visible:

| board run (3 reps) | rep 1 | rep 2 | rep 3 | spread |
|---|---|---|---|---|
| `p45free` (8-worker table) | 27.45 | 27.46 | 27.46 | 0.01 ms |
| `p45det` (1-worker table) | 29.92 | 30.20 | 30.59 | 0.67 ms |

The board is repeatable to 0.01 ms within an arm, so the **+2.74 ms is a property of the table, not
measurement noise**. The solver optimises makespan, not camera→control latency, so both tables are
equally good by the objective and the chain difference is incidental to it. State this plainly: the
reproducible recipe yields **30.2 ms**; 27.45 ms is what one non-deterministic solve found. Either
number carries the comparison against 242 ms, so a figure that wants a number a reader can
reproduce exactly should draw the deterministic one.

---

## 3. The displayed pair (needs a GPU)

```bash
scripts/display_pairs_unpinned45.sh        # env SEEDS MAX_SIMS
```

Both arms at cruise 1.4 m/s, gain 0.0055, scene (layout seed) **1000**, and **the same episode
seed** — which is the point of re-flying, since the earlier figure drew episode seed 1100 for the
scheduled arm against 1000 for the baseline.

The baseline's own draw, episode seed 1000, **does not separate the arms**: `p45free` crashes there
at gate 3 against the baseline's gate 2, and `verify_showdown_figure.py` rejects that render with
"panel A: displayed XPU-RT flight completes the course (3 gates)". It is kept as its own stem,
`warehouse_showdown_cam45_unpinned_seed1000`, because it is a result.

The scene census (§4) names the seeds that do separate them, and all of those are flown as pairs
rather than one being chosen. Accepted pair for this figure:

| arm | episode seed | outcome | gates | steps | control rate replayed |
|---|---|---|---|---|---|
| XPU-RT `p45free` | **1003** | success | 4/4 | 1149 | 103.5 Hz |
| ROS 2 `vanilla445` | **1003** | crash, hits a crate after G1 | **1/4** | 506 | 39.1 Hz |

**A display flight is a separate run from its census cell.** The census flies twelve episodes in one
`sweep_rate_demo.py` process; a display flight flies one episode under `record_sensor_demo.py`,
which is required because only the latter writes a `seed` key into `figure_data.npz` and panel A
reads one. An episode's outcome in the census therefore need not reproduce in a one-episode run:
seed 1006 completed in the census and crashed at gate 3 on its own. Flying every separating seed
rather than one is how that is handled — the figure draws a seed whose pair actually came out, and
panel A's legend carries the census tally it was drawn from.

---

## 4. Panel A's census

```bash
CELL=tall1000s LAYOUT_SEED=1000 CRUISE=1.4 GAIN=0.0055 \
  ARMS="xpu_p45free:results/codesign_feedback/ctrl_traces/xpu_p45free.csv:28.3:0.0" \
  OUT=results/codesign_feedback/campaign_scene/tall1000s scripts/scene_runs_pair.sh
```

Twelve episode seeds, 1000–1011 (`scene_runs_pair.sh` hard-codes `--seed 1000`), read only through
`scripts/flight_quarantine.py:flight_rows`:

| arm | 1000 | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1010 | 1011 | completed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `xpu_p45free` | ✗3 | ✗2 | ✗2 | **OK** | ✗2 | ✗2 | **OK** | **OK** | ✗2 | **OK** | ✗3 | ✗1 | **4/12** |
| `ros_vanilla445` | ✗2 | ✗2 | ✗0 | ✗1 | ✗1 | ✗1 | ✗2 | ✗1 | ✗0 | ✗1 | ✗1 | ✗1 | **0/12** |

Four seeds separate them — 1003, 1006, 1007, 1009 — and the baseline crashes at every one.

---

## 5. Panel D's energy ladder

Four conditions at one cruise speed, six seeds each. The three arms kept from `energy_runs_v2` were
flown at cruise **1.8**, and `energy_runs_free45` at **1.4**; mixing them would compare arms at
different speeds, so `p45free` is flown at 1.8 to join the v2 set rather than pasted across
campaigns (`scripts/followon_unpinned45.sh` stage 3). The condition directories of the three kept
arms are symlinked into a fresh run dir so the CSV carries exactly four conditions.

```bash
CONDS="xpu_p45free:<trace>" CRUISE=1.8 ER=results/codesign_feedback/energy_runs_unpinned45 \
  OUTCSV=results/codesign_feedback/flight_energy_unpinned45.csv scripts/run_energy_pair.sh
```

Drawn, relative to `xpu_p45free`: moment 1× / 4× / 20× / 46×, power 1× / 3× / 12× / 103× for
`xpu_p45free`, `xpu_greedy`, `ros_vanilla`, `ros_shipped`.

**The 1× reference must be the arm the panel draws as 1×.** `draw_mechanism`
(`scripts/hil_story_figure.py`) orders conditions by `ARM_NAMES` and takes the first `xpu` one;
`energy_ratios` (`scripts/showdown_paper_figure.py`) took the alphabetically first. For any CSV
holding both `xpu_p45free` and `xpu_greedy` those disagree — sorted gives greedy — so the sidecar
recorded ratios against an arm the figure does not draw as 1×. `energy_ratios` now chooses the
reference the way `draw_mechanism` does.

---

## 6. Panel I, and the spec the Gantt pair is built against

```bash
R=results/codesign_feedback; X=$R/xpurt_long; V=$R/ros_traced/45_vanilla4_r1
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$X/trace_p45freer1_other_run1.csv:$X/cpu_p45freer1_other_run1.csv:$X/manifest_p45freer1_other_run1.json:schedules/scheduled_wh_chain45_free_cpsat_profiled_clamped.json" \
  --arm "ros:ros:$V/trace.csv:$V/cpu.csv:$V/manifest.json" \
  --window-ms 100 --spec data/toplevel/wh_chain45_free.json \
  --out-prefix $R/refined/unpinned45b/measured_gantt_unpinned45b
```

**`--spec` is not optional at any rate but 25 Hz.** It defaults to
`data/toplevel/wh_coupled_chain_long25.json`, whose perception period is 40 ms, and the spec supplies
both the period the window bands are labelled with and the window a frame must finish inside to count
as on time. Without it, this pair labels the 45 Hz camera's period 40 ms instead of 22.22 and judges
lateness against 40 ms instead of the spec's 66.67, reading the baseline as 766 of 766 frames late;
the 45 Hz spec gives **760 of 766**. The chain latencies, 27.454 ms and 242.097 ms, are medians over
the traces and do not depend on the spec, nor does the XPU-RT row's 0 of 21.

`verify_showdown_figure.py` compares the figure against its sidecar, and both come from the same
build, so it does not see a spec mismatch. A ROS manifest records the rate its run used, so the
mismatch is detectable from inside the tool, and `make_measured_gantt_pair.py` exits on it:

```
spec/rate mismatch: ros's manifest ran at 45 Hz (22.22 ms period) but --spec
data/toplevel/wh_coupled_chain_long25.json declares a 40.00 ms perception period. Pass the spec for
this camera rate (e.g. data/toplevel/wh_chain45_free.json), or --yolo-window-ms explicitly.
```

---

## 7. Render and verify

```bash
R=results/codesign_feedback; P=$R/campaign_free45/display/pairs45
ENERGY_CSV=$R/flight_energy_unpinned45.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $P/xpu_s1003_figdata --ros-dir $P/ros_s1003_figdata \
  --scene-records $R/campaign_scene/tall1000s --display-cruise 1.4 \
  --xpu-trace xpu_p45free.csv --ros-trace ros_vanilla445.csv \
  --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
  --ros-label "ROS 2 · default deployment, unpinned, control chained to the goal" \
  --camera-hz 45 --gantt-prefix $R/refined/unpinned45b/measured_gantt_unpinned45b --gantt-rows xpu,ros \
  --out $R/refined/warehouse_showdown_cam45_unpinned_best_v2

.venv/bin/python scripts/verify_showdown_figure.py \
  --metrics $R/refined/warehouse_showdown_cam45_unpinned_best_v2_metrics.json   # 48 PASS, 0 FAIL
.venv/bin/python scripts/measured_timing.py --verify                            # 0 DRIFT
.venv/bin/python scripts/figure_constants.py                                    # 0 problems
```

`ENERGY_CSV` is an environment variable, not a flag. `--xpu-label` must name the solver the board
manifest records (`cpsat`), or the verifier's label check fails.

What the verifier establishes, beyond arithmetic: each Gantt row's trace, sampler and manifest carry
the sha256 recorded in the sidecar; the XPU row's executed table is on disk with the sha256 the
manifest states; both rows run the same staged YOLO IR; the lanes each network is *drawn* across
equal the lanes the schedule places it on **and** the lanes the trace shows it executing on; every
millisecond and hertz in drawn text is a registry value; and the clearance drawn in panel A was
measured against an obstacle.
