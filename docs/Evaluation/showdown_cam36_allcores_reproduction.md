# The warehouse showdown at a 36 Hz camera, against a baseline that uses every hart

This page reproduces the 36 Hz family in `results/codesign_feedback/refined/`:

| stem | baseline drawn | displayed seed |
|---|---|---|
| `showdown_36hz_solver_vs_rosallhart_s1006` | `vanilla4x2ns4` — two YOLO pools **and** a nav pool | 1006 |
| `warehouse_showdown_cam36_allcores_s1003` | `vanilla4x2` — two YOLO pools, nav on one hart | 1003 |
| `warehouse_showdown_cam36_allcores_s1007` | same | 1007 |
| `warehouse_showdown_cam36_allcores_s1009` | same | 1009 |

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md),
[`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md) for every measured ROS 2 arrangement,
[`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) for building and deploying one,
[`nav_sharding.md`](../K1/nav_sharding.md) for the nav worker pool,
[`showdown_cam30_solver_placed_reproduction.md`](showdown_cam30_solver_placed_reproduction.md) and
[`showdown_cam45_unpinned_best_reproduction.md`](showdown_cam45_unpinned_best_reproduction.md) for
the other two camera rates.

---

## 1. Why this figure exists, and why the camera runs at 36 Hz

The mechanism the figure isolates is the **control-rate floor**: ROS 2 chains control to perception,
so the command rate equals the camera rate no matter how fast the perception chain is. XPU-RT gives
control its own slot, so it commands at 100 Hz off a 36 Hz camera.

Every earlier form of this figure drew a baseline that was in some way less than the machine could
do — pinned to fewer harts, or running one YOLO pool while four harts sat idle. **This form removes
that objection.** The baseline here uses all eight harts, meets the deadline it was given, and leaves
no hart idle; the only thing it cannot do is command faster than its camera.

The camera rate is chosen so that the contrast is *drawable*, and this is a real constraint rather
than a presentational one. `verify_showdown_figure.py:506` requires panel A's baseline flight to end
with **one or two gates passed** — the drone must enter the course and lose it before the third gate.
Outside that window the panel has nothing to show:

| camera | what the all-hart baseline does on this scene | drawable? |
|---|---|---|
| 30 Hz | crashes **before the first gate** (0 gates) — hits a crate on the approach | no: the course never appears |
| **36 Hz** | enters the course, clears one or two gates, loses it | **yes** |
| 45 Hz | the arms stop separating — both meet the deadline and the rate gap alone is not decisive | no |

The 30 Hz renders are on disk (`warehouse_showdown_cam30_allcores_*`) and are *stronger* results for
XPU-RT than the one we draw. They are not the headline precisely because a baseline that never
reaches the course makes a worse figure, not a better claim. §7 records their outcomes.

---

## 2. The three arms

All three run the **same generated ModelBlaster kernels** for the same networks on the same eight
harts of the same SpacemiT K1 — the verifier checks that every board run carries the same staged
YOLO IR, `ir=0c783539626c`. Only the orchestrator differs.

| | XPU-RT `p36free` | ROS 2 `vanilla4x2` | ROS 2 `vanilla4x2ns4` |
|---|---|---|---|
| arrangement | CP-SAT schedule, **placement chosen by the solver**, control in its own slot | 5 processes, single executor, unpinned; **two** YOLO pools (harts 0–3 and 4–7) | as `vanilla4x2`, plus a **four-way worker pool under the nav network** |
| board run | `xpurt_long/trace_p36freer1_other_run1.csv` | `ros_traced/36_vanilla4x2d2_r1/` | `ros_traced/36_vanilla4x2ns4c_r1/` |
| camera→control, median | **25.82 ms** | 32.56 ms | 37.35 ms |
| camera→control, p95 | 30.45 ms | 47.11 ms | 53.91 ms |
| control cadence | **9.86 ms (≈100 Hz)** | 27.74 ms (36 Hz) | 27.76 ms (36 Hz) |
| harts carrying work | 8 of 8 | 8 of 8 | 8 of 8 |
| affinity | solver-assigned | `0xff` on all five nodes | `0xff` on all five nodes |
| cadence trace replayed | `ctrl_traces/xpu_p36free.csv` | `ctrl_traces/ros_vanilla4x236.csv` | `ctrl_traces/ros_vanilla4x236ns4.csv` |
| `--percep_latency_ms` | 25.8 | 32.3 | 37.4 |

**Nothing in XPU-RT's column is hand-placed.** The schedule is solved against
`data/toplevel/wh_chain36_free.json` with no `preferred_hw` and no hart constraints; which network
lands on which hart is the solver's output, not an input. See
[`partitioned_schedule.md`](../K1/partitioned_schedule.md).

**Board-measured timing, simulated flight consequence.** The drone never leaves Isaac Lab. Each
flight replays its arm's measured control-output series (`--ctrl_trace`) and its measured
camera→control latency (`--percep_latency_ms`), with the goal held for one camera period
(`--percep_hold_ms 27.8`). No timing in any flight is modelled.

---

## 3. The navigation worker pool, and why the baseline is not being held back

A fair objection to `vanilla4x2` is that its navigation network runs on a single hart while YOLO gets
eight. `vanilla4x2ns4` answers it: the nav network is built with `MB_SHARD_FACTOR=4`, which splits all
seven of its convolutions on the output-channel axis, and the nav node is given a four-way worker
pool (`--nav-pool 4`). Three placements of that pool were measured:

| nav arrangement | board run | nav callback | camera→control | control cadence |
|---|---|---|---|---|
| one hart (no pool) | `36_vanilla4x2d2_r1` | 4.15 ms | **32.56 ms** | 27.74 ms |
| 4-way pool, **unpinned** | `36_vanilla4x2ns4c_r1` | 5.76 ms | **37.35 ms** | 27.76 ms |
| 4-way pool, pinned 0–3 | `36_vanilla4x2ns4a_r1` | 5.77 ms | 38.71 ms | 27.79 ms |
| 4-way pool, pinned 4–7 | `36_vanilla4x2ns4b_r1` | 5.95 ms | 39.17 ms | 27.78 ms |

**Giving nav more harts makes the chain slower, and the reason is not the kernel.** Run standalone on
an idle board, the same sharded nav is **1.66× faster** on four harts than on one (226 k → 136 k
cycles). In the chain it is 1.39× *slower*, and YOLO's own callback degrades from 25.89 ms to
29.93 ms alongside it. The board has no spare harts: `vanilla4x2` already keeps all eight between
36 % and 59 % busy, so a nav pool does not find idle capacity, it takes YOLO's.

That is why the nav-pool arm is the one worth drawing. It is the **most capable ROS 2 arrangement we
have measured** — nothing in the perception→navigation chain is left on a single core — and it is
still bound by the same control-rate floor. The unpinned variant (`ns4c`) is the fastest of the three
and is the one the figure uses: the baseline is drawn at its best.

The nav pool is reached generically, not by a hand-written rule for this network:
`shard_conv_weights` splits any convolution whose output-channel count divides by the factor
(nav's are 16, 32, 64, 64, 16, 16 — all divisible by 4; a factor of 3 shards none of them), and
`scripts/ros_traced_matrix.sh` exposes it as `NAVPOOL`/`NAVHARTS` rather than as a nav-specific path.

---

## 4. Building it

### 4.1 The board runs

```bash
# XPU-RT, solver-placed, against the 36 Hz workload spec. Same two steps as the 45 Hz sibling in
# scripts/board_free45.sh: solve with the placement free, then carry the table through the codegen
# contract and run it. manifest_p36freer1_other_run1.json records what actually ran --
# schedules/scheduled_wh_chain36_free_cpsat_profiled_clamped.json, solver cpsat, camera_hz 36.
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain36_free.json \
    --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 3600 --use-profiled --board-calibration $CAL
REPS=3 bash scripts/board_partitioned30.sh \
    schedules/scheduled_wh_chain36_free_cpsat_profiled.json p36free gen/mb_shard_nav

# ROS 2, two YOLO pools over all eight harts
HZ=36 scripts/ros_traced_matrix.sh vanilla4x2

# ROS 2, the same plus a four-way nav pool (unpinned; omit NAVHARTS to let the OS place it)
HZ=36 NAVPOOL=4 scripts/ros_traced_matrix.sh vanilla4x2
```

The nav pool requires the sharded nav kernels and an `rv64gcv_zvfh` build (the f16 vector types the
sharded convolutions emit need `zvfh`, not plain `rv64gcv`); see
[`nav_sharding.md`](../K1/nav_sharding.md) §*Building the sharded nav for the board*.

### 4.1b The cadence traces the flights replay

A flight does not run either runtime; it replays the control cadence each produced on the board.
`scripts/ctrl_trace_from_board.py` cuts that from a board artifact and writes the source, the warm-up
and the resulting cadence into the file's own header, so a re-cut can be checked against the
committed file line for line.

```bash
R=results/codesign_feedback
# XPU-RT: from the dispatch trace. Default 100 ms warm-up -- an xpurt_long run is under a second end
# to end, so the ROS arms' 3 s would discard all of it and the tool exits 1 writing nothing.
.venv/bin/python scripts/ctrl_trace_from_board.py \
    $R/xpurt_long/trace_p36freer1_other_run1.csv --out $R/ctrl_traces/xpu_p36free.csv

# ROS 2, two YOLO pools: from the per-callback gap log
.venv/bin/python scripts/ctrl_trace_from_board.py \
    $R/ros_traced/36_vanilla4x2_r1/ctrl_gaps.csv --out $R/ctrl_traces/ros_vanilla4x236.csv

# ROS 2, the same plus the nav pool (the arm this figure draws)
.venv/bin/python scripts/ctrl_trace_from_board.py \
    $R/ros_traced/36_vanilla4x2ns4c_r1/ctrl_gaps.csv --out $R/ctrl_traces/ros_vanilla4x236ns4.csv
```

The committed files head with `eff_hz=99.90`, `eff_hz=36.14` and `eff_hz=36.02` respectively, each at
`warmup_ms=100.0`. Note these two ROS cuts take the default warm-up rather than the 3 s the 45 Hz ROS arms
use: their runs settle inside the first 100 ms at this rate, and the header is what says which was
used for a given file.

### 4.2 The Gantt rows

Both rows are drawn from the board traces, not from the schedule the solver emitted:

```bash
X=results/codesign_feedback/xpurt_long
V=results/codesign_feedback/ros_traced/36_vanilla4x2ns4c_r1
S=schedules/scheduled_wh_chain36_free_cpsat_profiled_clamped.json
scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$X/trace_p36freer1_other_run1.csv:$X/cpu_p36freer1_other_run1.csv:$X/manifest_p36freer1_other_run1.json:$S" \
  --arm "ros:ros:$V/trace.csv:$V/cpu.csv:$V/manifest.json" \
  --window-ms 100 \
  --out-prefix results/codesign_feedback/refined/navshard36/measured_gantt_navshard36
```

Use `36_vanilla4x2d2_r1` and the prefix `refined/allcores36/measured_gantt_allcores36` for the
nav-on-one-hart form.

**How a pooled dispatch is drawn.** The board records two things for each YOLO frame: the real
per-worker slices stamped by `modelblaster_pool_trace_arm`, and a per-op reconstruction that
apportions the callback span by profile cycles. The reconstruction is always the wider of the two,
because it charges every layer a share of the pool's dispatch and barrier overhead, so drawing both
would paint the same frame twice and put more than a hart-second of work on a hart-second of lane.
`make_measured_gantt_pair.py` therefore keeps a per-op row only when the measured slices on that
hart account for less than `COVER_FRAC` (0.5) of its span — that is, only when the op really did run
sequentially — and draws everything else from the slices themselves. The slices of one dispatch are
then coalesced by `coalesce_shards()` into a single bar spanning the lanes that ran it, so a frame
reads as one four-lane block rather than as a few hundred sub-millisecond slivers. A lane drawn from
measured slices carries `lanes_measured`, which suppresses the "reconstructed" hatch that
`showdown_gatecourse.py` applies to inferred placements.

Both YOLO pools record their slices, so the drawn baseline occupies all eight lanes:
`CPU_E#1+CPU_P#1+CPU_P#2+CPU_P#3` for pool 1 and `CPU_E#1+CPU_E#2+CPU_E#3+CPU_P#0` for pool 2.

**Which replicate the Gantt comes from, and which one the flights replay.** Recording the second
pool's per-shard detail costs the baseline a little tail latency: mean cadence is unchanged to
0.1 ms (27.77 ms either way), but p95 gap rises from 31.4 ms to 36.2 ms, outside the r1–r3 spread of
the un-instrumented replicates (31.4 / 31.2 / 30.3 ms). **The Gantt is drawn from the instrumented
replicate `d2`, and the flights replay the un-instrumented `r1` cadence** — the arm is flown as
deployed, and only the picture comes from the instrumented run. Mean cadence, which is what a flight
actually follows, is the same in both.

### 4.3 Panel D's energy ladder

Panel D isolates what the control cadence alone costs, so its runs use no latency and no goal hold:

```bash
R=$PWD/results/codesign_feedback; T=$R/ctrl_traces
CONDS="xpu_p36free:$T/xpu_p36free.csv ros_vanilla4x2ns4_36:$T/ros_vanilla4x236ns4.csv" \
  CRUISE=1.4 GAIN=0.0055 ER=$R/energy_runs_navpool36 \
  OUTCSV=$R/flight_energy_navpool36.csv MAX_SIMS=3 bash scripts/run_energy_pair.sh
```

The ladder must hold **the arm the figure draws**. `energy_ratios()` normalises to the arm the
drawing treats as 1×, so a ladder built for one baseline under a flight flown by another would record
a normalisation the panel does not show; the nav-pool figure gets its own CSV for that reason.

---

## 5. The census

Twelve episodes per arm on the display scene, at the deployed gain, through `sweep_rate_demo.py`:

```bash
bash scripts/campaign_allcores36.sh    # xpu_p36free and ros_vanilla4x236
bash scripts/campaign_navpool36.sh     # ros_vanilla4x236ns4, into the same cell
```

Cell `results/codesign_feedback/campaign_scene/tall1000_ac36`, layout seed 1000, cruise 1.4, gain
0.0055, goal hold 27.8 ms. Read it only through `scripts/flight_quarantine.py:flight_rows`, which
drops GPU-faulted batches.

| arm | 1000 | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1010 | 1011 | completed | mean gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ros_vanilla4x236` | x1 | x1 | x3 | x2 | x1 | x1 | x1 | x0 | x1 | x1 | x1 | x1 | **0/12** | 1.17 |
| `ros_vanilla4x236ns4` | x3 | x1 | x1 | x1 | x1 | x3 | x3 | x2 | x2 | x0 | x3 | x2 | **0/12** | 1.83 |
| `xpu_p36free` | x3 | OK4 | x2 | OK4 | x1 | x2 | x3 | OK4 | x1 | OK4 | x3 | x1 | **4/12** | 2.67 |

`OK4` = completed the four-gate course; `xN` = crashed after N gates. This tally is what panel A's
legend carries, and it is the population the claim is about.

---

## 6. The displayed flight, and how its seed was chosen

A census cell and a one-episode display run are **different runs and disagree often**. The census
flies twelve episodes in one process and writes no `seed` key into `figure_data.npz`; panel A reads
one, so the displayed flight must come from `record_sensor_demo.py --episodes 1`. An episode's
outcome in the census does not reliably reproduce in a one-episode run.

The way that is handled here is to fly **the entire population as display pairs** rather than to draw
seeds until one looks right:

```bash
bash scripts/display_all12_ac36.sh     # 12 seeds x 3 arms = 36 one-episode flights
```

Log: `results/codesign_feedback/display_all12_ac36.log`. Every outcome:

| seed | `xpu_p36free` | `ros_vanilla4x236` | `ros_vanilla4x236ns4` |
|---|---|---|---|
| 1000 | crash 3 | crash 1 | crash 2 |
| 1001 | **success 4/4** | crash 3 | crash 3 |
| 1002 | crash 2 | crash 2 | crash 1 |
| 1003 | **success 4/4** | crash 1 | *success 4/4* |
| 1004 | crash 2 | crash 1 | crash 1 |
| 1005 | crash 2 | crash 1 | crash 2 |
| **1006** | **success 4/4** | crash 1 | **crash 2** |
| 1007 | **success 4/4** | crash 1 | crash 1 |
| 1008 | crash 2 | crash 1 | crash 1 |
| 1009 | **success 4/4** | crash 1 | crash 1 |
| 1010 | crash 2 | crash 1 | crash 1 |
| 1011 | crash 2 | crash 1 | crash 1 |

Seed **1006** is the only seed in this population where the scheduled arm completes the course and a
baseline ends inside panel A's one-or-two-gate window, so it is the seed the headline figure draws.
It was not selected by the baseline's outcome: the whole population was flown first and is recorded
above, and the figure's own legend carries the census tally rather than the displayed flight.

**The baseline is unreliable at this rate, not incapable.** At seed 1003 the nav-pool baseline flew
the whole course, where its census cell has it crashing after one gate. Captions must state the
census fraction — 0 of 12 — and must not say the baseline cannot fly the course. Panels B and C
already put it correctly, as a success fraction against control rate.

Flights that fall outside what panel A can draw are still rendered, under a name that says so
(`..._xpu_2of4`, `..._ros_3of4`), rather than dropped.

---

## 7. Rendering and verifying

```bash
bash scripts/render_navpool36.sh                  # SEEDS="1006 1000 1005"
bash scripts/render_allcores36.sh                 # the nav-on-one-hart form
```

Each render is followed by `scripts/verify_showdown_figure.py --metrics <stem>_metrics.json`.
Expected results across the family:

| stem | verifier | why |
|---|---|---|
| `..._cam36_navpool_s1006` | **0 FAIL** | baseline at 2 gates, scheduled arm completes |
| `..._cam36_allcores_s1003` / `_s1007` / `_s1009` | **0 FAIL** | baseline at 1 gate |
| `..._cam36_allcores_s1001` | 1 FAIL | baseline reaches 3 gates — outside panel A's window |
| `..._cam30_allcores_*` | 1 FAIL | baseline crashes before the first gate (0 gates) |

Only the 0-FAIL stems are listed in `FIGURES` in `artifact/verify_no_hardware.sh`. The others are
kept on disk with their sidecars: a render the panel-A rule rejects is still a recorded flight, and
naming it distinctly is how that is said.

---

## 8. Caveats to carry into a caption

* **Solver time limit.** The 36 Hz schedule was solved under a 3600 s bound and returned with an
  oracle gap of 0.2247 % — a near-optimal schedule, not a proven-optimal one. See
  [`scheduler_oracle_gap.md`](../Feature/scheduler_oracle_gap.md).
* **The baseline meets its deadline.** At 36 Hz `vanilla4x2` is late on 1 frame of 353. The figure is
  not about a baseline that misses deadlines; it is about one that meets them and still cannot
  command faster than its camera. Do not describe the ROS 2 row as overloaded.
* **`vanilla4x2` is unpinned and default-executor** — ROS 2 as normally written, with the one change
  that YOLO is given two worker pools so that all eight harts carry work. The pinned arrangements
  (`p3`, `cp3`, `part8`) are measured separately in [`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md).
* **Four gates is the whole course.** "success 4/4" means the drone finished, not that it scored.
