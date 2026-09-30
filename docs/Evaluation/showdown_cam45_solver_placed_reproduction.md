# The warehouse showdown at a 45 Hz camera, with the placement left to the solver

This page reproduces two stems in `results/codesign_feedback/refined/`:

| stem | Gantt pair | panel I, XPU-RT | panel I, ROS 2 |
|---|---|---|---|
| `showdown_45hz_solver_vs_rospinned_s1007` | `refined/free45b/` | 27.454 ms, 0/21 late | 56.757 ms, 0/769 late |
| `warehouse_showdown_cam45_solver_placed` | `refined/free45/` | 27.454 ms, 0/21 late | 56.757 ms, 765/769 late |

The two draw the **same flights and the same board runs**; they differ only in the workload spec the
Gantt pair was built against, which decides the window a frame must finish inside to count as on time.
§3.3 states which spec belongs to this camera rate and why the choice is not free.

Companion pages: [`showdown_cam45_static6_reproduction.md`](showdown_cam45_static6_reproduction.md) —
the same baseline against a *hand-pinned* XPU-RT arm, which is the comparison this figure replaces;
[`showdown_cam45_unpinned_best_reproduction.md`](showdown_cam45_unpinned_best_reproduction.md) for the
solver-placed arm's own board provenance in more depth;
[`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md), [`artifact_checklist.md`](../Artifact/artifact_checklist.md).

---

## 1. Why this figure exists

Its sibling at `showdown_cam45_static6_reproduction.md` draws XPU-RT's **hand-pinned** CP-SAT arm,
which runs 56.86 ms camera→control — the same latency as the pinned ROS 2 baseline's 56.757 ms. With
no latency difference the comparison rested on command rate alone, and on that scene the census came
out a tie. That is an honest bound, but it also means the 45 Hz comparison was measuring the placement
we chose by hand rather than what the runtime can do.

This figure removes the hand-placement. The spec `data/toplevel/wh_chain45_free.json` carries **no
`allowed_machines` and no `machine_width`**: where each kernel lands is the solver's answer, not an
instruction. At a 22.2 ms camera period it finds **27.45 ms** — less than half the hand-pinned arm, and
less than half the baseline.

Panel A's census, twelve seeds on the scene the pair flies:

| arm | completions | mean gates |
|---|---|---|
| XPU-RT · solver-placed CP-SAT, conv on the IME | 3 / 12 | 2.50 |
| ROS 2 · static 6-core partition | 0 / 12 | 1.92 |

This is the result the tie at `cam45_static6` bounds: give the solver the placement and the gap at
45 Hz reopens, against the same well-deployed baseline.

**What the Gantt pair shows, scored against the 45 Hz spec.** Both arms meet their perception window on every frame — the
baseline is not missing deadlines at this rate; the "765 of 769 late" of a pair scored against the 40 ms default is the
window, not the run (§3.3). So the figure is not a deadline-miss result. It is two things at once: a
**2.07× shorter chain** (27.45 against 56.76 ms) and a **2.7× higher command rate** (103.5 against
38.5 Hz), and the flights separate on the second.

## 2. The two arms

| | XPU-RT | ROS 2 |
|---|---|---|
| arm | `p45free` — CP-SAT, hard windows, placement unconstrained, conv on the IME | `cp3` — 3 processes, single executor, control chained to the goal |
| placement | 8 harts, chosen by the solver | percep (camera + perception + a 4-hart YOLO pool) on harts 0–3; nav pinned to hart 4; control pinned to hart 5. Two harts idle. |
| camera→control | **27.46 ms** (pooled median of three replicates: 27.454 / 27.464 / 27.455) | 56.757 ms |
| control cadence | 9.84 ms → **103.5 Hz** | 25.99 ms → **38.5 Hz** |
| board run | `xpurt_long/trace_p45freer{1,2,3}_other_run1.csv` | `ros_traced/45_cp3_r1/` |
| executed table | `schedules/scheduled_wh_chain45_free_cpsat_profiled_clamped.json` | — |
| cadence trace | `ctrl_traces/xpu_p45free.csv` | `ctrl_traces/ros_cp345.csv` |

The flights were flown with `--percep_latency_ms 28.3`, the value recorded for this arm at the time;
the pooled median is 27.46. The difference is 0.85 ms **against** the scheduled arm — the flight was
given slightly more latency than the board measures — and both numbers are in the dumps and the
registry.

## 3. Building it

### 3.1 The board runs

```bash
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json

# XPU-RT: solve the 45 Hz chain with the placement free, then run the table three times.
# scripts/board_free45.sh does all of this, including the cadence cut in §3.2.
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain45_free.json \
    --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 3600 --use-profiled --board-calibration $CAL
REPS=3 scripts/board_partitioned30.sh \
    schedules/scheduled_wh_chain45_free_cpsat_profiled.json p45free gen/mb_shard_nav

# ROS 2, three pinned processes, control chained to the goal topic
HZ=45 scripts/ros_traced_matrix.sh cp3                     # -> ros_traced/45_cp3_r{1,2,3}/
```

`board_partitioned30.sh` carries the table through the codegen contract (clamping dispatch widths to
what the kernels can actually be built at) before running it, so the executed table is
`…_clamped.json` and is what the Gantt and the verifier read.

### 3.2 The cadence traces the flights replay

A flight does not run either runtime; it replays the control cadence each produced on the board.

```bash
# XPU-RT: from the dispatch trace, default 100 ms warm-up
.venv/bin/python scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/xpurt_long/trace_p45freer1_other_run1.csv \
    --out results/codesign_feedback/ctrl_traces/xpu_p45free.csv

# ROS 2: from the per-callback gap log, 3 s warm-up so the executor has settled
.venv/bin/python scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/ros_traced/45_cp3_r1/ctrl_gaps.csv \
    --out results/codesign_feedback/ctrl_traces/ros_cp345.csv --warmup-ms 3000
```

**The two warm-ups differ because the two runs do.** A `ros_traced` run is about 20 s and its executor
needs seconds to settle; an `xpurt_long` run is under a second end to end, so a 3 s warm-up discards
all of it and `ctrl_trace_from_board.py` exits 1 with `only 0 gaps after warm-up`, writing no file.
The committed files carry `warmup_ms=100.0 … eff_hz=101.63` and `warmup_ms=3000.0 … eff_hz=38.47`; a
re-cut that does not reproduce those header values did not use these arguments.

### 3.3 The Gantt rows

**`--spec` is required, and the spec must be the one for this camera rate.** It supplies both the
period the window bands are labelled with and the window a frame must finish inside to count as on
time. `wh_chain45_free.json` declares `yolov8_nano_64x96.period = 22.22 ms` and
`window_duration = 66.67 ms`. Built against a spec for another rate, every latency stays right while
the lateness count belongs to a different workload — `make_measured_gantt_pair.py` now refuses to run
without a spec for exactly this reason.

```bash
R=results/codesign_feedback; X=$R/xpurt_long; V=$R/ros_traced/45_cp3_r1
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --spec data/toplevel/wh_chain45_free.json --window-ms 160 \
  --arm "xpu:xpu:$X/trace_p45freer1_other_run1.csv:$X/cpu_p45freer1_other_run1.csv:$X/manifest_p45freer1_other_run1.json:schedules/scheduled_wh_chain45_free_cpsat_profiled_clamped.json" \
  --arm "p3:ros:$V/trace.csv:$V/cpu.csv:$V/manifest.json" \
  --out-prefix $R/refined/free45b/measured_gantt_free45b
```

Each row picks the 160 ms window whose own frames carry the run's median latency, so the bars and the
caption describe the same thing; `window_pick: representative` in the sidecar records that. The
XPU-RT row's lane list shows the solver's answer directly — single harts, pairs and full-cluster
groups side by side, because it picks a width per dispatch rather than per network.

### 3.4 Panel D's energy ladder

Four rungs, cadence only — the flights differ in nothing but the replayed control trace:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
CONDS="xpu_p45free:$T/xpu_p45free.csv xpu_greedy:$T/xpu_a_greedy.csv ros_static6:$T/ros_cp345.csv ros_shipped:$T/ros_vanilla_c5045.csv" \
  CRUISE=1.4 GAIN=0.0055 ER=$PWD/results/codesign_feedback/energy_runs_free45 \
  OUTCSV=$PWD/results/codesign_feedback/flight_energy_free45.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh
```

Panel D normalises to the first XPU-RT condition in `hil_story_figure.ARM_NAMES` order, resolved
through `canon_arm()` — `xpu_p45free` canonicalises to the solver arm, so the 1× reference is this
figure's own arm and not `xpu_greedy`.

## 4. The census and the displayed pair (GPU)

`scripts/followon_free45.sh` is the whole GPU chain in order, and is what produced everything below.

```bash
bash scripts/followon_free45.sh          # env: CRUISES, MAX_SIMS
```

Its stages:

0. `scripts/campaign_free45.sh` — the census. **Only the scheduled arm is flown here**; `cp3`'s rows
   carry over from `campaign_static6_45`, which flew the identical baseline at the identical cruise
   and gain. This is the fair-count rule: both arms have twelve seeds on the scene, from the same
   simulator configuration.
1. `scripts/display_search_free45.sh` → `display_same_env.sh` → `record_sensor_demo.py`, searching
   cruise speeds for a pair where the arms separate. It accepted **cruise 1.4, seed 1007**, writing
   `campaign_free45/display/search_c1.4/{xpu,ros}_s1007_figdata`.
2. `scene_runs_pair.sh` with `CELL=free45_l1007`, twelve seeds per arm on the scene the pair flies —
   the census panel A's legend reports.
3. `run_energy_pair.sh` as in §3.4.

Both display dumps are `.gitignore`d (78–314 MB each) and archived in
`archive_v3/display_dumps_cam45_forms.tar`, whose sha256 is in the tracked
`archive_v3/MANIFEST.sha256`. The figure's sidecar records each dump's own sha256. Unpack into
`results/codesign_feedback/campaign_free45/display/search_c1.4/`.

## 5. Rendering and verifying

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_free45.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_free45/display/search_c1.4/xpu_s1007_figdata \
  --ros-dir $R/campaign_free45/display/search_c1.4/ros_s1007_figdata \
  --scene-records $R/campaign_scene/free45_l1007 --display-cruise 1.4 \
  --xpu-trace xpu_p45free.csv --ros-trace ros_cp345.csv \
  --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
  --ros-label "ROS 2 · static 6-core partition, two cores idle" \
  --camera-hz 45 --gantt-prefix $R/refined/free45b/measured_gantt_free45b --gantt-rows xpu,p3 \
  --out $R/refined/showdown_45hz_solver_vs_rospinned_s1007

.venv/bin/python scripts/verify_showdown_figure.py \
  --metrics $R/refined/showdown_45hz_solver_vs_rospinned_s1007_metrics.json   # 0 FAIL
```

`ENERGY_CSV` is an environment variable, not a flag. `--xpu-label` must name the solver recorded in the
board manifest behind `xpu_p45free.csv` (`cpsat`), which the verifier checks.

## 6. Caveats to carry into a caption

* **Neither arm completes the course on every seed.** The census is 3/12 against 0/12, mean 2.50
  against 1.92 gates. State the population, not only the displayed flight.
* **Both arms meet their perception window on every frame.** This is not a deadline-miss result; the
  separation is command rate, 103.5 Hz against 38.5 Hz, on top of a 2.07× shorter chain.
* **The baseline leaves two harts idle** by construction — "static 6-core partition". It is a good
  deployment, not the maximal one; the arm that uses every hart is at 36 Hz, in
  [`showdown_cam36_allcores_reproduction.md`](showdown_cam36_allcores_reproduction.md).
* **The solve is `feasible`, not proven optimal.** CP-SAT returns within the 3600 s bound without
  closing the gap; `docs/Feature/scheduler_oracle_gap.md` records how far the table sits from the oracle
  floor.
* **The flights were flown at 28.3 ms** injected latency while the pooled board median is 27.46 ms —
  0.85 ms in the baseline's favour, recorded in both the dumps and the registry.
