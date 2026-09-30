# The warehouse showdown at a 45 Hz camera, against a statically partitioned ROS 2

This page reproduces two stems in `results/codesign_feedback/refined/`:

| stem | Gantt pair | panel I, XPU-RT | panel I, ROS 2 |
|---|---|---|---|
| `showdown_45hz_pinned_vs_rospinned_s1011` | `refined/static6_45b/` | 56.86 ms, 0/44 late | 56.757 ms, 0/769 late |
| `warehouse_showdown_cam45_static6` | `refined/static6_45/` | 56.86 ms, 44/44 late | 56.757 ms, 765/769 late |

The two draw the **same flights and the same board runs**; they differ only in the workload spec the
Gantt pair was built against, which decides the window a frame must finish inside to count as on time.
§3.3 states which spec belongs to this camera rate and why the choice is not free.

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md),
[`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md) for every measured ROS 2 arrangement,
[`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) for building and deploying one,
[`showdown_cam45_solver_placed_reproduction.md`](showdown_cam45_solver_placed_reproduction.md) for the
companion figure that draws the solver-placed `p45free` arm in place of `a_cpsat_hard` (whose figure label reads "hand-pinned placement"; its spec carries no placement keys), and
[`showdown_cam36_allcores_reproduction.md`](showdown_cam36_allcores_reproduction.md) for the 36 Hz
family.

---

## 1. What this figure is for

**It is the falsification test, not a win.** Every other form of the showdown draws a ROS 2 baseline
that is in some way less than the machine could do. This one gives the baseline a *good* deployment —
three pinned processes, YOLO on its own four-hart pool, nav and control each on their own hart — and
asks whether the gap survives.

It does not, on this scene. Panel A's own legend says so:

| arm | completions over 12 seeds | mean gates |
|---|---|---|
| XPU-RT · CP-SAT (`a_cpsat_hard`) | 4 / 12 | 2.75 |
| ROS 2 · static 6-core partition | 4 / 12 | **2.83** |

The baseline edges us. The displayed flight (seed 1011) shows it clearing two gates and hitting a
crate, but one flight is not the population and the header states the 4-against-4 tally on the
figure's face. **No caption may read this figure as a win.** Its value is that it bounds the claim:
the control-rate gap is a property of how ROS 2 is *usually* deployed, and a well-partitioned ROS 2 at
45 Hz closes it.

The Gantt pair, scored against the 45 Hz spec, sharpens that. Both arms have essentially the same chain — 56.86 ms against
56.757 ms — and both meet their perception window on every frame. What is left between them is the
command rate: **96 Hz against 38 Hz**. The figure isolates that and nothing else.

## 2. The two arms

| | XPU-RT | ROS 2 |
|---|---|---|
| arm | `a_cpsat_hard` — CP-SAT, hard windows, solved from `wh_chain45_solve.json` (no placement keys; the figure stem calls it "pinned", and `xpurt_arm_ranking.md` lists its placement as the solver's choice) | `cp3` — 3 processes, single executor, control chained to the goal |
| placement | 8 harts, scheduled | percep (camera + perception + a 4-hart YOLO pool) on harts 0–3; nav pinned to hart 4; control pinned to hart 5. **Two harts idle** — the "static 6-core partition" the label names. |
| camera→control | 56.86 ms | 56.757 ms |
| control cadence | 10.0 ms → **96.2 Hz** | 25.99 ms → **38.5 Hz** |
| board run | `xpurt_long/trace_acpsat_hardr1_other_run1.csv` (+ `r2`, `r3`) | `ros_traced/45_cp3_r1/` |
| executed table | `schedules/fig_a_cpsat_hard_clamped.json` | — |
| cadence trace | `ctrl_traces/xpu_a_cpsat_hard.csv` | `ctrl_traces/ros_cp345.csv` |

The chain latencies are within 0.1 ms of each other. That is deliberate: a comparison in which the two
arms differ in *both* latency and command rate cannot attribute the outcome to either.

## 3. Building it

### 3.1 The board runs

Both take the spec as a **bare name** under `data/toplevel/`, not a path, and `board_stage2.sh` blocks
until the solve log it is paired with reports `STAGE2_DONE` — so the solve must be redirected there.
This is the `run_pair` idiom of `scripts/chain_rates2.sh`:

```bash
L=results/codesign_feedback/solver_v2
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 \
       XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8

# XPU-RT, CP-SAT with hard windows, solved from wh_chain45_solve.json, against the 45 Hz workload spec.
# tag `a` -> schedules/fig_a_{greedy,cpsat_hard,cpsat_soft}.json, clamped by board_stage2.sh
bash scripts/solve_stage2_hard.sh wh_chain45_solve a 3000 > $L/stage2_a.log 2>&1
bash scripts/board_stage2.sh a wh_chain45_solve            # -> trace_acpsat_hardr{1,2,3}_other_run1.csv

# ROS 2, three pinned processes, control chained to the goal topic
HZ=45 scripts/ros_traced_matrix.sh cp3                     # -> ros_traced/45_cp3_r{1,2,3}/
```

`scripts/pull_ros_traced.py` pulls the ROS runs into `ros_traced/<hz>_<arm>_r<k>/` and appends to
`ros_traced/summary.csv`, which is what `measured_timing.py --verify` re-derives the ROS constants
from.

### 3.2 The cadence traces the flights replay

A flight does not run either runtime; it replays the control cadence each one produced on the board.
`scripts/ctrl_trace_from_board.py` cuts that from a board artifact, and writes the source, warm-up and
the resulting cadence into the file's own header.

```bash
# XPU-RT: from the dispatch trace, default 100 ms warm-up
.venv/bin/python scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/xpurt_long/trace_acpsat_hardr1_other_run1.csv \
    --out results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv

# ROS 2: from the per-callback gap log, 3 s warm-up so the executor has settled
.venv/bin/python scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/ros_traced/45_cp3_r1/ctrl_gaps.csv \
    --out results/codesign_feedback/ctrl_traces/ros_cp345.csv --warmup-ms 3000
```

The committed files carry `warmup_ms=100.0 … eff_hz=100.04` and `warmup_ms=3000.0 … eff_hz=38.47`
respectively; a re-cut that does not reproduce those header values did not use these arguments.

### 3.3 The Gantt rows

**`--spec` is required, and the spec must be the one for this camera rate.** It supplies both the
period the window bands are labelled with and the window a frame must finish inside to count as on
time. `wh_chain45_free.json` declares `yolov8_nano_64x96.period = 22.22 ms` and
`window_duration = 66.67 ms`. Build the pair against a spec for a different rate and the lateness
count belongs to a different workload while every other number stays right —
`make_measured_gantt_pair.py` now refuses to run without a spec for exactly this reason.

```bash
R=results/codesign_feedback; X=$R/xpurt_long; V=$R/ros_traced/45_cp3_r1
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --spec data/toplevel/wh_chain45_free.json --window-ms 160 \
  --arm "xpu:xpu:$X/trace_acpsat_hardr1_other_run1.csv:$X/cpu_acpsat_hardr1_other_run1.csv:$X/manifest_acpsat_hardr1_other_run1.json:schedules/fig_a_cpsat_hard_clamped.json" \
  --arm "p3:ros:$V/trace.csv:$V/cpu.csv:$V/manifest.json" \
  --out-prefix $R/refined/static6_45b/measured_gantt_static6_45b
```

Each row picks the 160 ms window whose own frames carry the run's median latency, so the bars and the
caption describe the same thing; `window_pick: representative` in the sidecar records that.

### 3.4 Panel D's energy ladder

Four rungs, cadence only — the flights differ in nothing but the replayed control trace:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
CONDS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv xpu_greedy:$T/xpu_a_greedy.csv ros_static6:$T/ros_cp345.csv ros_shipped:$T/ros_vanilla_c5045.csv" \
  CRUISE=1.4 GAIN=0.0055 ER=$PWD/results/codesign_feedback/energy_runs_static6_45 \
  OUTCSV=$PWD/results/codesign_feedback/flight_energy_static6_45.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh
```

Two of the four rungs are ours, so the panel reads as a spectrum over scheduling quality rather than a
two-horse race. Moment is measured from the logged wrench; power is modelled (mixer + momentum
theory), and the panel keeps the two visually distinct.

## 4. The census and the displayed pair (GPU)

`scripts/followon_static6_45.sh` is the whole GPU chain in order, and is what produced everything
below. It waits on the 30 Hz chain first, because they share the simulator budget.

```bash
bash scripts/followon_static6_45.sh          # env: CRUISES, MAX_SIMS
```

Its four stages:

0. `scripts/campaign_static6_45.sh` — the 45 Hz census, baseline only; the scheduled arm's rows carry
   over from the earlier campaign.
1. `scripts/display_search_static6_45.sh` → `display_same_env.sh` → `record_sensor_demo.py`, searching
   cruise speeds for a pair where the arms separate. It accepted **cruise 1.4, seed 1011**, writing
   `campaign_static6_45/display/search_c1.4/{xpu,ros}_s1011_figdata`.
2. `scene_runs_pair.sh` with `CELL=static6_45_l1011`, twelve seeds per arm on the scene the pair flies
   — this is the census panel A's legend reports, and it is where the 4-against-4 tie comes from.
3. `run_energy_pair.sh` as in §3.4.

Both display dumps are `.gitignore`d (78–314 MB each) and archived in
`archive_v3/display_dumps_cam45_forms.tar`, whose sha256 is in the tracked `archive_v3/MANIFEST.sha256`.
The figure's sidecar records each dump's own sha256, so a render can be tied to the exact bytes it
read. Unpack into `results/codesign_feedback/campaign_static6_45/display/search_c1.4/`.

## 5. Rendering and verifying

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_static6_45.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_static6_45/display/search_c1.4/xpu_s1011_figdata \
  --ros-dir $R/campaign_static6_45/display/search_c1.4/ros_s1011_figdata \
  --scene-records $R/campaign_scene/static6_45_l1011 --display-cruise 1.4 \
  --xpu-trace xpu_a_cpsat_hard.csv --ros-trace ros_cp345.csv \
  --xpu-label "XPU-RT · CP-SAT, hand-pinned placement" \
  --ros-label "ROS 2 · static 6-core partition, two cores idle" \
  --camera-hz 45 --gantt-prefix $R/refined/static6_45b/measured_gantt_static6_45b --gantt-rows xpu,p3 \
  --out $R/refined/showdown_45hz_pinned_vs_rospinned_s1011

.venv/bin/python scripts/verify_showdown_figure.py \
  --metrics $R/refined/showdown_45hz_pinned_vs_rospinned_s1011_metrics.json    # 0 FAIL
```

`ENERGY_CSV` is an environment variable, not a flag. `--gantt-rows xpu,p3` must name the two arms
exactly as the `--arm` specs in §3.3 named them.

## 6. Caveats to carry into a caption

* **The census is a tie**, 4/12 against 4/12, and the baseline's mean gate count is higher (2.83
  against 2.75). Say so. The displayed seed is one draw, not the population.
* **Both arms meet their perception window on every frame.** Nothing here is a deadline-miss result.
  The whole difference is the command rate, 96 Hz against 38 Hz.
* **Two harts are idle in the baseline** by construction — that is what "static 6-core partition"
  means. It is a good deployment, not the best possible one; the arm that uses every hart is
  `vanilla4x2` at 36 Hz, in
  [`showdown_cam36_allcores_reproduction.md`](showdown_cam36_allcores_reproduction.md).
* **The 45 Hz tie is a bound on the claim, not a counterexample to it.** The control-rate floor is
  stated over the deployment ROS 2 users actually write; this figure shows what a careful partition
  buys, and it buys parity at this rate.
