# Reproducing the measurements and the ablations

This is the operational companion to [`feedback_loop_reference.md`](../Feature/feedback_loop_reference.md),
which explains what the co-design loop *is* and what the study *claims*. This document is
about how to **re-run it and check it**: the K1 measurements, the ablation of the
experiments (which rungs, and why those), and the ablation of the feedback (which half of
the loop is doing the work).

Related, and not duplicated here:

| for | read |
|---|---|
| what the loop is, and the argument | [`feedback_loop_reference.md`](../Feature/feedback_loop_reference.md) |
| board access, toolchain, one model at a time | [`k1_board.md`](../K1/k1_board.md) |
| the loop stage by stage, and the graph-rewrite arm | [`codesign_loop_reproduction.md`](../Feature/codesign_loop_reproduction.md) |
| porting to a target that is not the K1 | [`REPRODUCE.md`](../Artifact/REPRODUCE.md) |
| the ROS 2 baseline, end to end | [`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) |
| the w4/w5 rungs specifically | [`w4_w5_inner_outer.md`](../Feature/w4_w5_inner_outer.md) |

Everything below runs from a checkout at any path. Two interpreters are involved:
`$REPO/.venv/bin/python` for the scheduler and the analysis (override with `XPURT_PY`),
and a separate Isaac interpreter for the flight sims (`ISAAC_PY`), which is only needed
for the HIL figures and not for anything in this document.

---

## 0. The short version

```bash
# the inner loop alone -- no board, deterministic, ~2 minutes
.venv/bin/python scripts/run_codesign_loop.py \
    --workload data/toplevel/scaling/s5_solvable_reveal.json \
    --solver greedy --max-rounds 4 --replay

# the whole outer loop on real silicon -- needs the K1, ~40 minutes
scripts/run_board_arc.sh --workload data/toplevel/scaling/s5_solvable_reveal.json \
    --stem s5_arc --models mlp_control,fused_full,ffn_block,dronet,yolov8_nano_64x96

# the 2x2 feedback ablation, both solvers
.venv/bin/python scripts/ablate_feedback_loops.py \
    --workloads data/toplevel/scaling/{s5_solvable_reveal,w4_ffn_dronet_sensor}.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --solvers cpsat,greedy --cpsat-time-limit 150 --out-dir results/loop_ablation_postfix
```

`--replay` is the mode someone without our hardware can check: it pins
`XPURT_CPSAT_WORKERS=1` and refuses anything that would touch the board, so two runs of
the same inputs produce byte-identical schedules.

---

## 1. Reproducing the K1 measurements

### 1.1 The chain, and why it is a script

`scripts/run_board_arc.sh` runs five steps, each needing the previous one's output path.
It is a script rather than a runbook because doing it by hand is where the `--backends`
arity and the `--staged-ir` flags kept getting fumbled.

```
  0. inner loop            run_codesign_loop.py         -> a converged spec
  1. solve                 run_xpurt_schedule.py        -> a schedule
  2. execute on the K1     ModelBlaster/scripts/run_xpurt_k1.sh -> a trace CSV
  3. calibrate             emit_board_calibration.py    -> measured multipliers
  4. attribute             attribute_board_misses.py    -> execution vs queueing
  5. re-cost and re-solve  run_codesign_loop.py         -> the four-beat arc
```

Steps 2–4 are the outer loop proper. Steps 0–1 are the inner loop, and can be run alone
on any machine.

### 1.2 Four things the chain has to get right

Each of these is enforced in code rather than described in prose, because each is a
property of the run that a reader of the output cannot see.

**The converged spec is the last ACCEPTED round, not the newest file.** The loop writes a
candidate spec for every lever it *tries*, so the final round — the one that accepts
nothing and thereby ends the search — leaves the most recent files on disk, and those are
rejects. Picking by mtime once sent a *rejected* round-3 candidate (`shard:dronet`, which
the accept rule had just refused) to the board. The driver reads `loop_report.json`.

**`--backends` takes one entry per core kind, not per model.** `rvv_x60,rvv_x60` is
`cpu_p,cpu_e`, and it stays two entries whether you run one network or five.

**Calibrate from the run's own trace, every time.** A calibration measured on a different
rung is an extrapolation. Reusing `b5z`'s table for `s5` mispredicted it by 1.66× — which
is exactly the class of error the outer loop exists to catch, so importing it at the
calibration step defeats the purpose. There are nine calibrations in
`results/codesign_feedback/` for this reason, one per board run.

**Pass `--schedule` to the calibrator.** The runner records zero-cost ops with
`dispatch_id -1` and numbers the rest from zero, while the schedule numbers all of them,
so on a network with zero-cost ops the two numberings diverge. Ratios are computed within
a row and are always right; only the *key* can be wrong, and that is invisible in the
emitted table. `--schedule` turns on the alignment check, and a network that fails it
loses its per-dispatch keys rather than keeping plausible-looking wrong ones.

### 1.3 What was measured

Nine board runs, each calibrated from its own trace, all alignment-verified:

| rung | dispatches | mean inflation | median | nets exact | trace |
|---|---:|---:|---:|---:|---|
| `c2` | 109 | 1.132× | 1.069× | 2 | `c2_aot_sched_trace.csv` |
| `c3` | 289 | 1.323× | 1.134× | 3 | `c3_aot_sched_trace.csv` |
| `b4` | 394 | 1.312× | 1.149× | 4 | `b4_aot_sched_trace.csv` |
| `b5x` | 333 | 1.429× | 1.202× | 4 | `b5x_aot_sched_trace.csv` |
| `b5y` | 333 | 1.434× | 1.204× | 4 | `b5y_aot_sched_trace.csv` |
| `b5z` | 333 | 1.448× | 1.183× | 4 | `b5z_aot_sched_trace.csv` |
| `s5` | 207 | 1.728× | 1.329× | 4 | `s5_arc_sched_trace.csv` |
| `w4` | 394 | 1.304× | 1.135× | 4 | `w4_aot_sched_trace.csv` |
| `w5` | 484 | 1.348× | 1.221× | 4 | `w5_shard_sched_trace.csv` |

Read the **median** as the typical dispatch and the **mean** as the one the scheduler has
to survive; they differ because inflation has a tail. The mean is what the calibration
applies, deliberately.

Two properties of these numbers worth stating precisely, because they bound what the
calibration means:

* **Execution only.** Queue delay is carried separately by the trace and excluded. A
  dispatch that *waited* is not a dispatch that *ran slowly*, and folding queueing in
  would charge the scheduler twice for its own placement.
* **Three tiers, degrading honestly.** `network/dispatch_id` where it was measured, op
  kind where it was not, aggregate as the floor. A net absent from `coverage.nets_exact`
  is costed by its op kinds, which is a *prediction* about that net rather than a
  measurement of it — and the JSON says so in its own `coverage.note`.

Timing is `rdtime` at 24 MHz. The pooled op tier and the aggregate are floored at
predicted ≥ 0.1 ms (2400 ticks) so timer granularity cannot inflate a pooled multiplier.

### 1.4 The four-beat arcs

The shape the study is about: **baseline misses → AOT clears → the board reveals →
the board re-solve clears again.**

| rung | arc | levers | ends at |
|---|---|---|---|
| `sensor_evo_auto` | 2 → 0 → **6** → **0** | shard + IME | **all met** |
| `sensor_evo_ime` | 2 → 0 → 1 → 0 | IME | all met |
| `s5_solvable_reveal` | 1 → 0 → 6 → **2** | shard | 2 residual |
| `b4_board_sized` | 10 → 3 → 4 → 4 | shard | 4 residual |
| `w4_ffn_dronet_sensor` | 10 → 5 → 4 → 4 | shard | 4 residual |
| `w5_ffn_dronet_yolo` | 11 → 7 → 16 → 14 | shard | 14 residual |

`sensor_evo_auto` is the one the figure uses: it is the only arc that applies **both**
levers (12 dispatches at width 2, 61 at width 4, plus IME) *and* returns to all-met.

The rows that do not return to zero are kept, not hidden, and step 4 of the chain says
why each one does not. `s5`'s residual is one execution-bound yolo instance — no
placement recovers a dispatch that cannot fit its window at any width — and `w5`'s is
larger because at 492 dispatches CP-SAT never proves phase 1 (see §2.3), so the re-solve
is optimising against an unconverged bound.

```bash
.venv/bin/python scripts/attribute_board_misses.py \
    --trace <trace.csv> --spec data/toplevel/scaling/s5_solvable_reveal.json \
    --json-out results/codesign_feedback/s5_miss_attribution.json
```

`execution_bound_instances: 0` means every remaining miss is queueing, and the loop is
accountable for it. Anything above zero is a compiler gap wearing a scheduler's clothes.
Measured: `b5y` 0, `s5` 1, `b4` 2, `b5x` 6, `w5` 6.

---

### 1.5 The pipeline over one second: periodic schedules on the board

The single-frame coupled run answers what one camera frame costs. The flight figures need
what the runtime delivers *continuously*: how often the control loop fires, whether frames
start on time, how old a goal is when it is acted on. That takes a schedule that spans a
second, and the walker executes whatever table it is given, so the length of a run is the
number of periodic instances in its schedule.

Two ways to produce one, both kept:

* **Tile the solver's frame.** `scripts/tile_schedule.py` repeats the CP-SAT single-frame
  placement at the camera period (frame *k* offset by *k*·period, the camera→YOLO→nav edges
  kept inside each frame) and places `mlp_control` as its own periodic task — one instance per
  control period, carrying only its internal edges — because in the deployed pipeline control
  acts on the newest goal available at its own release rather than waiting for a particular
  frame. A spec that puts an edge from nav into control makes every control instance wait for
  its frame's nav result; that is the topic-chained shape, and it is measured as such on the
  ROS side (`cship`/`cspin`/`cp3`), not as the XPU-RT arm.
* **Place explicitly.** `scripts/build_best_schedule.py` writes the same kind of table for the
  layouts the runtime makes possible: YOLO sharded 4-way on the P cluster with nav and control
  on their own E harts (`p4`); frames alternating between the two clusters so two are in flight,
  at width 2 so the packed weights are laid out once, with control alone on `CPU_E#3` (`alt2`).
  Every dispatch of a frame carries the frame's release as its start, so within a frame the
  walker runs dependency-driven and the hardware sets the pace.

Two rules the codegen imposes on any periodic schedule, checked by `schedule_shards` before a
build and enforced by `scripts/clamp_schedule_widths.py` on solver output: a packed-weight
convolution takes one shard width across all its instances, and that width divides its output
channels.

`scripts/run_xpurt_long.sh <schedule> <label> [n]` runs a schedule *n* times with the per-core
sampler around it and keeps, per run, the trace, the harness's per-hart accounting
(`hart_acc_*.csv`; its `_us` columns are 24 MHz ticks), the sampler series and a manifest
(solver, schedule hash, IR hash, scheduling policy, `run_t0_rdtime`). Primaries run under
`SCHED_OTHER`, the policy the ROS arms run under; a FIFO run is a labelled variant.
`scripts/xpurt_trace_report.py` reads a trace back: per-frame YOLO span, camera→control from
the frame's release, the control-output gap distribution, frame start lag, per-hart kernel
fraction.

What the one-second traces show, at the 45 Hz camera the workload is specified for
(`results/codesign_feedback/xpurt_long/`):

| schedule | frames | camera→control | control output gap |
|---|---|---|---|
| CP-SAT frame tiled at 25 Hz, control decoupled | on time | 30 ms | 10.0 ms mean, p95 12 |
| `p4` at 25 Hz | on time | 29.8 ms | 10.00 ms, 9.93–10.06 |
| `p4` at 45 Hz | start 62 ms late (YOLO 24 ms > 22 ms period) | 90 ms | 10.00 ms, 9.98–10.03 |
| **`alt2` at 45 Hz** | **on time** | **40 ms** | **10.00 ms, 9.98–10.10** |
| `alt2` at 60 Hz | on time | 40 ms | 10.00 ms |
| `alt2` at 75 Hz | start 90 ms late — the width-2 layout's ceiling is ~64 Hz | 120 ms | 10.00 ms |
| `alt1` at 90 Hz (six frames in flight, width 1) | on time | 57 ms | 10.00 ms |
| `alt2` at 45 Hz + `ffn_block` 10 Hz + `dronet` 30 Hz | on time | 41 ms | 9.98 ms, max 12.1 |
| `alt2` at 45 Hz, control asked for 200 Hz | on time | 38 ms | 5.00 ms, max 5.3 |
| `alt2` at 45 Hz, two unpinned busy-loop processes alongside (`HOGS=2`) | 8 of 88 frames late, up to 30 ms | 46 ms (p95 63) | 9.97 ms, max 15.1 |
| `alt2` at 45 Hz, five-second table (996 control gaps) | on time | 40 ms | 10.00 ms, 9.98–10.03 |
| `alt1` with **two** 45 Hz cameras (90 frames/s offered) | on time (lag ≤ 5 ms) | 72 ms | 10.00 ms, 9.98–10.03 |
| `alt2` at 45 Hz + `ffn_block` + `dronet`, `ffn_block` on the matrix engine | on time | 41 ms | 9.97 ms, max 12.2 |
| `alt1`, two 45 Hz cameras **and** the heavier stack (`ffn_block` on the IME) | 16 of 178 frames start late, by ≤ 10 ms | 74 ms | 9.96 ms, max 12.2 |

**Solved, not placed.** The tables above are explicit placements. The figure's XPU-RT rows are
the solvers' own output on one periodic spec, `data/toplevel/wh_chain45_solve.json`: the
deployed chain at a 45 Hz camera, YOLO's window three periods (66.7 ms, so frames may be in
flight together), nav's 77.8 ms, control a 100 Hz task with a 10 ms window paired to the newest
closed goal. A window is a deadline. It enters the CP-SAT model as a lateness variable and a boolean miss
flag per operator — `lat = max(0, end - deadline)`, `miss = (lat > 0)` — which lead a
lexicographic objective weighted 10^12 for a miss and 10^8 for lateness, above makespan. There
is no `end <= deadline` bound, so the solver does not report infeasible when a window cannot be
met; it returns the least-missing schedule and records the count. On every spec solved here that
count is 0 (`schedules/fig_*_cpsat_hard_metrics.json`, `deadline_miss_count`), while greedy
(`greedy_periodic`), which does not carry the objective, records 2766 to 7258. What does fail
hard is finding no solution at all: the run raises and writes no table.
On the 200 ms hyperperiod CP-SAT is optimal with 0 window misses and greedy has 336; the
one-second tables (`schedules/fig_a_{cpsat_hard,greedy}_clamped.json`, widths clamped to the
codegen contract, feasibility-checked) were each run three times on the board, interleaved:

| solver, same spec, same kernels | YOLO frames late | camera→control | control gap |
|---|---|---|---|
| CP-SAT | **0 of 120** | **57 ms** (p95 72) | 10.00 ms mean, max 19.4 (12/294 > 15 ms) |
| greedy | **120 of 120** (frames pile up, 713 ms each) | **748 ms** (p95 1065) | 13.25 ms mean: bursts (p50 0.03) then a 1.04 s silence |

Both tables place the same 5785 operations on the same eight machines from the same cost
database (`pdb_hash` a7f1f637…), so the difference is the objective, and the schedules show where
it lands. Greedy's makespan is only 1.29× CP-SAT's (1322.3 ms against 1027.6), because packing
machines is what a list scheduler is good at. What it does not hold together is a frame: taking
each network instance's span from its first operator's start to its last operator's end,

| solver | YOLO instance span (median) | nav | control |
|---|---|---|---|
| CP-SAT | 45.3 ms | 4.6 ms | 0.08 ms |
| greedy | 722.7 ms | 4.6 ms | 0.08 ms |

Nav and control are identical; the whole difference is that greedy interleaves one frame's
operators with fifteen other frames'. A schedule can therefore be near-optimal on makespan and
still miss every window, which is why the board shows a 13× chain gap under a 1.29× makespan gap.

### The mechanism panel, checked against a second campaign

The energy campaign (`scripts/run_energy_v2.sh` -> `flight_energy_v2.csv`) flies six seeds per arm
at cruise 1.8 and replays each arm's control cadence. It does not inject the camera-to-control
latency, and the flight campaign says that does not matter for this quantity: the same trace flown
with and without its latency gives the same mean commanded moment to within a few percent
(`ros_vanilla445` 0.8594 against 0.8608 at 242 ms; `xpu_a_greedy` 0.0959 against 0.0954 at 748 ms).
Commanded moment follows the control cadence, not the perception delay.

The ratios the panel draws reproduce from `campaign_percep`'s own records at the same cruise, an
independent set of 36 flights per arm:

| arm | energy campaign (n=6) | flight campaign (n=36) | panel draws |
|---|---|---|---|
| XPU-RT · CP-SAT | 0.0169 | 0.0437 | 1x |
| XPU-RT · greedy | 0.0689 (4.1x) | 0.1351 (3.1x) | 4x |
| ROS 2 vanilla | 0.3216 (19.0x) | 0.8226 (18.8x) | 19x |

The absolute moments differ by a constant between the two record sets; the ratios, which is what
the panel shows, agree. Note the panel's flights are at cruise 1.8 while the top-down pair is at
1.4, so the two panels describe the same arms at different speeds.

### The dark camera frame in the crash strip

The baseline's moment strip shows a nearly black forward camera. That is the measurement, not a
missing frame: the strip is taken at the last recorded step, and the baseline ends against an
obstacle. At that step the cross-ToF reads 0.025-0.103 m on all four sensors and the chase view
shows the drone at the crate, so the three views in the strip agree. Brightness falls only on the
final frame — 50 of 51 frames in the displayed flight peak above 0.5, and the last peaks at 0.23.

It is characteristic of a crash rather than of one seed: the alternative pair at seed 1007 (also
XPU-RT completing, the baseline crashing after one gate) ends the same way, 47 of 48 frames legible
and a final peak of 0.34. Both pairs are on disk under `campaign_v2/display_lat_c1.4/`.

### What a displayed pair carries

A replayed flight carries whichever parts of a deployment's timing were injected. `--ctrl_trace`
replays the measured control cadence; `--percep_latency_ms` injects the measured camera-to-control
latency. They are independent, and a pair flown with the trace alone differs from its arm's
deployment in exactly the second one.

This matters because the scene census printed beside a displayed pair is flown with both. At
`tall1005s` seed 1005 the solved arm completes the course on a cadence-only replay and crashes at
the first gate once its 56.8 ms is injected, so a pair and a census that disagree on this are two
experiments on one panel. The dumps under `campaign_v2/display_same/` predate the recorder storing
`percep_latency_ms` and cannot say what they carried; `display_v3/` and `display_v3s_c1.4/` record
56.8 ms and 242.0 ms.

A displayed pair is usable when all three hold: the two arms share a seed, both record the latency
their arm measures, and a census exists at the same cruise. Check with

```bash
python - <<'EOF'
import numpy as np
for a in ("xpu", "ros"):
    z = np.load(f"<dir>/{a}_s<seed>_figdata/figure_data.npz", allow_pickle=True)
    print(a, int(z["seed"]), float(z["percep_latency_ms"]), float(z["cruise_speed"]))
EOF
```

`verify_showdown_figure.py` fails a figure whose drawn flights cannot show they carried their
arm's latency.

### Camera rate

The camera rate is the axis on which the deployments separate, and the board was swept across it.
Each row is the best replicate of that arm at that rate; XPU-RT rows are the CP-SAT table solved
for that rate from the same spec family.

| camera | XPU-RT CP-SAT | XPU-RT greedy | ROS 2 `vanilla4` | ROS 2 `p3` |
|---|---|---|---|---|
| 30 Hz | 55.2 ms | 70.1 ms | 31.1 ms | 30.4 ms |
| 45 Hz | 56.8 ms | 748.0 ms | 242.2 ms | 56.4 ms |
| 60 Hz | 58.5 ms | 583.6 ms | 189.4 ms | 55.9 ms |
| 90 Hz | 55.9 ms | 947.5 ms | 136.5 ms | 56.2 ms |
| 120 Hz | 59.9 ms | 566.3 ms | 110.5 ms | 56.4 ms |

The solved table is flat over the whole sweep, 55.2 to 59.9 ms from 30 to 120 Hz, because the
window is a constraint on every instance at every rate. The untuned deployment tracks the camera
while a frame fits between two of them and breaks once it does not: 31.1 ms at 30 Hz, 242.2 ms at
45. Hand-pinning moves that break point but does not remove it — `p3` holds about 56 ms from 45 Hz
up. Greedy is above the window at every rate past 30 Hz.

Three cautions on reading this table. The ROS rows below 45 Hz are not a like-for-like win: a
30 Hz camera leaves a whole frame period of slack, so every arrangement fits and the 31 ms is the
chain with no queueing in it.

The second is sharper, because the table reads backwards without it. `vanilla4` appears to recover
at 120 Hz — 110.5 ms against 242.2 ms at 45 — and it has not. The chain median is taken over frames
that produced a goal, and at 120 Hz that arm drops most of them: 2171 frames in, 643 goals out, 30%
surviving, against 81% at 45 Hz. Goal delivery is the same at both rates (37.8/s and 38.0/s), so
the arm is saturated and the extra camera buys nothing; the median improves because the frames that
queued no longer appear in it. The hand-pinned arm keeps 93-95% at both rates and its chain is
unchanged, which is what a rate-independent deployment looks like. Read goals per second and the
surviving fraction alongside the median, never the median alone.

The third: every ROS row is the best replicate of that arm, and the arms were not swept at every
rate, so a blank is an unmeasured cell rather than a failure.

### Ceilings, at equal workload

Hand-placement helps both systems, so the honest contrast is drawn at matched effort and matched
workload. All rows below deploy the same three networks at a 45 Hz camera on the same board, and
all but one deploy a single model instance:

Two latencies are reported per arm and they are not interchangeable. Camera-to-goal ends when the
navigation result is published; camera-to-control ends when the controller acts on it. They
coincide when control is chained to the goal and separate when control runs on its own timer, which
is exactly what `p3` does. The schedule panel draws camera-to-control, so the table below uses that
for every row, with camera-to-goal beside it where the two differ.

| deployment | placed by | model instances | camera→control | camera→goal | control |
|---|---|---|---|---|---|
| XPU-RT `best45alt2c200` | hand | 1 | 37.9 ms | — | every 5.00 ms |
| XPU-RT `best45alt2long` | hand | 1 | 40.1 ms | — | every 10.00 ms |
| XPU-RT `acpsat_hardr` | CP-SAT, from the spec | 1 | 56.8 ms | — | every 10.00 ms |
| ROS 2 `p3` | hand | 1 | 61.6 ms | 56.4 ms | every 10.00 ms |
| ROS 2 `vanilla4x2` | hand | 2 | 37.1 ms | 37.0 ms | every 22.2 ms |
| ROS 2 `vanilla4` | none | 1 | 242.4 ms | 242.4 ms | every 25.9 ms |

Two readings follow. At matched effort and one model instance, the hand-placed schedule reaches
37.9 ms with control every 5.00 ms where the hand-pinned deployment reaches 61.6 ms with control
every 10.00 ms. And the solver, given only the spec, lands at 56.8 ms — ahead of the hand-pinned
deployment's 61.6 ms on the same quantity, without anyone choosing a core. The 4.8 ms between them
is the phase wait `p3` pays because its control runs on a free timer rather than on the goal: its
navigation result is ready at 56.4 ms and waits for the next control tick.

The one row that reaches 37.0 ms without a solver does it by running the network twice: two model
instances, two copies of the weights, alternate frames at 22.5 fps each, and control every
22.2 ms rather than every 10.

The gap between the solved table and the hand placement is the objective, not the machine. The
shipped table is the hard-window solve: every window is a constraint and the solver stops when it
can meet them, which it does with 0 misses. It is not asked to minimise the chain. Re-solving the
same spec with the lateness objective (`SOFT=1 scripts/solve_stage2_hard.sh wh_chain45_solve …`)
does not currently produce a table: the hard path returns `UNKNOWN` at a 900 s limit under load,
and the soft path returns no assignment at all. So the 56.8 ms row is what this solver produces
for this spec today, and the 40.1 ms hand placement is the standing evidence that the spec admits
a lower chain than the shipped table reaches.


The same on the second spec — one camera at **90 Hz** plus `ffn_block` 10 Hz and `dronet`
30 Hz, a half-second table (`wh_chain90_rich_solve_500.json`), three runs each: CP-SAT 0 of 108
frames late, 58 ms camera→control, control 9.95 ms; greedy 108 of 108 late, 150 ms, and every
`ffn_block` and control window missed. Costing the detector at ×1.00 or ×1.20 instead of ×1.10
moves greedy's predicted misses (255 / 507) and leaves CP-SAT at 0.

Across camera rates (the same chain spec at 30, 60 and 90 Hz, half-second tables, three runs
each): CP-SAT keeps every frame on time at every rate — 55.2 / 58.5 / 55.9 ms camera→control
at 30 / 60 / 90 Hz, control at 10.0 ms — while greedy holds only at 30 Hz (70.1 ms, no frame
late) and loses every frame above it (583.6 ms at 60 Hz, 947.5 ms at 90 Hz). At 120 Hz the
half-second table is beyond what the hard-window solve returns in 3000 s, so both solvers run the
200 ms hyperperiod table itself (`wh_chain120_solve_h200.json`, `scripts/chain_rates_high_h200.sh`,
three runs each): CP-SAT 59.9 ms camera→control, control at 10.0 ms, 0 of 36 frames late; greedy
566 ms, 36 of 36 late, its 200 ms table taking 700 ms to execute. Letting the
solver shard YOLO on a profile calibrated from the executed shard traces (`gen/mb_cal`,
`scripts/calibrate_yolo_shard_profile.py`) did not carry to the board — both solvers' tables
ran late there (291.0 / 490.6 ms) — so the figure's tables keep YOLO one hart wide.

YOLO's multi-hart profile is flat, so neither solver can value sharding and both run it one
hart wide; that is why the solved table's chain is 57 ms where the hand-placed `alt2` reaches
40 ms. `scripts/solve_certificates.sh`, `solve_stage2_hard.sh`, `board_stage2.sh`, and
`xpurt_trace_report.py --windows` produce and read every number in the table.

The control loop is what the schedule guarantees: its window is its own hart's, and it holds
whether or not perception keeps up — at 200 Hz as at 100, over five seconds as over one, and
under two background hogs the control hart keeps its 10 ms while the frames absorb the
disturbance (the hogs are unpinned and land wherever CFS puts them, the scheduled harts
included). Throughput at the design rate needs both clusters carrying
frames; `alt2` is the placement that does it, and frames-in-flight is the knob that trades
latency for headroom beyond it (`alt1`). `scripts/plot_throughput_latency.py` draws every run of
both runtimes on one throughput–latency map. Putting `ffn_block`'s two linears on the IME
(`--ime-nets ffn_block`, backends `rvv_x60,ime_x60,rvv_x60`) halves them — fc1 13.7 → 7.2 ms,
fc2 10.5 → 5.9 ms per instance — without touching the chain, which never runs on that hart.
With a second camera the same table shape carries 90 frames a second on the eight harts at a
72 ms chain; the ROS 2 pool arm given two cameras fires control every 60 ms with a 420 ms
camera→goal, and one process per camera holds 10.00 ms at 210–230 ms camera→goal with every
core 70–96 % busy. With both at once — two cameras and the heavier stack — the hand-partitioned
ROS 2 arm (one process per camera, nav, control, extras) holds its 10.00 ms control but the
camera→goal chain is 227–239 ms with 8–19 % of frames dropped and every core 62–83 % busy;
the same load on XPU-RT is a 74 ms chain with 16 of 178 frames starting up to 10 ms late and control at 9.96 ms
(max 12.2). For scale, ViNT — the transformer policy the
richer specs are modelled on — was put through the same pipeline all-int8 and timed on the
board at 2.3 s per frame (`scripts/vint_k1_build.sh`); it is not a workload this class of SoC
carries at a control-relevant rate under any runtime. At the other end, the VitFly
convolutional front end (`scripts/vitfly_k1_build.sh`, nine ops) costs 0.3–0.4 ms a frame on one
hart — lighter than `fused_full`, so it is not the heavier navigation stage; `dronet` is.

### 1.6 The ROS 2 baseline, traced

**As one would write it.** The baseline of the flight figure is the graph a ROS 2 user writes
first: one node per stage, each `rclcpp::spin(node)` in its own process, launched unpinned,
default executor and QoS, the generated `run_model()` as-is, control computed in the goal
callback (`vanilla`); the same with the model's 4-hart build (`vanilla4`), with control on a
100 Hz timer instead (`vanilla4t`), and with `ffn_block` and `dronet` added as their own
processes (`rvanilla`, `rvanilla4`). Measured on the K1 across camera rates 15–90 Hz, three
replicates each:

| deployment, 45 Hz camera | control-output gap | camera→goal | goals reaching control | frames dropped |
|---|---|---|---|---|
| `vanilla` — serial YOLO, chained control | 48.2 ms (20 Hz) | 265 ms | 20.4/s | 57 % |
| `vanilla4` — 4-hart YOLO, chained control | 25.7 ms (39 Hz) | 242 ms | 38.4/s | 19 % |
| `vanilla4t` — 4-hart YOLO, timer control | 30.4 ms (33 Hz) | 121 ms | 32.6/s | 1 % |
| `rvanilla` — serial YOLO + heavier stack | 83.8 ms (12 Hz) | 301 ms | 11.7/s | 75 % |

Two settings a ROS 2 user might add without touching code or cores were measured as well:
keep-last-1 QoS on the topics (`vanilla4_q1`) empties the stale-frame queue — camera→goal
242 → 42 ms — but leaves the cadence at 25.6 ms, because the goal callback still runs behind one
YOLO; and pipelining by hand (`vanilla4x2`: the camera alternates frames between two perception
processes, each with its own model instance and 4-hart pool, all eight cores busy) lifts
control to 16.6 ms (60 Hz, jittering to 47) at a 455 ms camera→goal at 45 Hz (254 ms at 90 Hz,
46 % dropped) — the two pools and the unpinned nav and control contend for the same harts.

The control cadence is set by the perception node, because the goal callback is where control
runs (or, on the timer variant, because the executor thread that fires the timer is the one
running YOLO); the camera rate only changes how many frames the queue throws away. The tuned
arms below are what it takes to change that by hand.

**Tuned by hand.**

The baseline is real ROS 2 Jazzy running the same kernels on the same board, written as raw
files in the same trace contract; [`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md)
is the recipe. The one result to carry: the control timer is starved exactly when a YOLO
callback shares its thread. As the graph ships (one process, default executor, serial kernels)
the 100 Hz control timer fires every 22 ms at a 15 Hz camera and every 53 ms from 25 Hz; with
YOLO on a 4-hart pool it holds ~13 ms up to 30 Hz and falls to 30 ms at 45 Hz; with control on
its own thread or process, or under the multi-threaded executor, it holds 10.00 ms at every
rate. Chained control (run per goal) fires at the camera rate up to ~33 Hz. Each of those is a
deployment someone would write, and the flight figure names the one it uses.

The pool arm at 45 Hz was also run under the same three perturbations as the XPU-RT table
above (`ros_traced/45_*_{q1,c200,hog2}_r1`): asking its timer for 200 Hz leaves the control
gap at 30 ms (the thread, not the timer, is the limit; the three-process arm goes to 5.00 ms);
two background hogs change nothing for either (they land on idle E cores); QoS depth 1 raises
the pool arm to 21 ms — the shallow queue drops frames, so fewer YOLO callbacks share the
thread — at a 60 ms camera→goal, and the as-shipped arm to 36 ms.

Read as three deployments (`scripts/plot_deployment_layers.py` →
`refined/deployment_layers.png`, every bar from a trace), same kernels, same board:

| ROS 2 deployment, 45 Hz camera | control gap | camera→control | frames to control |
|---|---|---|---|
| as deployed — one process, default executor, YOLO on a 4-hart pool | 30 ms | 120 ms | 33/s |
| the multi-threaded executor | 10.00 | 264 ms | 23/s, 48 % dropped |
| hand-partitioned — a process per node, cores pinned | 10.00 | 53 ms | 39/s, 6 % dropped |
| XPU-RT, `alt2` | 10.00 | 40 ms | 41/s of 45, none dropped |

The first is where the flight figure's crash comes from. The second is the fix a reader would
try first: the control loop survives because half the perception is thrown away. The third is
the developer doing the scheduler's job by hand; it competes at one camera and falls furthest
behind under the combined load (two cameras and the heavier stack: 237 ms and 13 % dropped
against 74 ms with nothing dropped; the as-deployed arm is at 624 ms with half its frames gone,
the multi-threaded executor at 254 ms with two thirds gone).
ROS 2 can use the cores; the schedule decides when.

### 1.7 The flights, second form: replayed cadences

Every flight of `results/codesign_feedback/campaign_v2/` replays a control-output series measured on
the K1 (`ctrl_traces/`): XPU-RT's CP-SAT and greedy tables of the 45 Hz chain spec, and the
out-of-the-box ROS 2 graph with the model's 4-hart YOLO (`ros_vanilla4`, 39 outputs/s), with a
100 Hz timer (`ros_vanilla4t`, 33/s) and with the serial YOLO (`ros_vanilla`, 21/s). Ten cruise
speeds, 12 seeds a cell, the same gain on every arm (0.0055); a second policy sets each arm's
gain to 0.5 / its replayed rate. `scripts/campaign_select_v2.py` names the displayed cell — the
fastest speed at which XPU-RT completes at least 3/12 while the baseline completes at most 1/12
and crashes after entering the course in at least half its flights — and the recorder
(`scripts/display_v3.sh`) flies that cell the way the campaign flew it and keeps the first
episode meeting each arm's rule, video and figure data from the one flight.

At the fixed gain the rule picks **1.8 m/s**: XPU-RT CP-SAT 4/12 complete, the 39 Hz baseline
1/12 with nine flights crashing right after the second gate. At the speeds below that the
39 Hz baseline is close to the 100 Hz loop under this gain (the controller was tuned at 50 Hz,
so a 100 Hz loop is over-authoritative and a 39 Hz loop near-authoritative), which is what the
calibrated policy exists to separate; the serial-YOLO graph (21 outputs/s) cannot hold altitude
at any speed, and neither can greedy's table (its control outputs come in a burst followed by a
second of silence): 0/12 with no gate cleared at every speed flown.

Under the calibrated policy — each arm's gain set to 0.5 / its replayed rate, so the 39 Hz
baseline flies at the authority a 39 Hz loop calls for and the 96 Hz loop at its own — the
separation is the envelope's: the baseline is 0/12 at 1.8 and 1.6 m/s (most flights never reach
the first gate) where XPU-RT completes 3/12 and 6/12, and XPU-RT completes 9/12 at 1.2 and
7/12 at 1.0. The fixed-gain cells and the calibrated cells are both in `campaign_v2.csv`
(`moment_scale` column) and `campaign_select_v2.py --policy` reports either.

On the unseen-gate course (`WAREHOUSE_COURSE=b`, `campaign_v2_courseB/`), replaying the same
cadences at the fixed gain: XPU-RT CP-SAT 7/12, 3/12 and 7/12 at 1.0, 1.2 and 1.4 m/s; the
timer-driven vanilla graph (33 Hz) 2/12, 2/12 and 0/12 — eleven of its twelve flights at 1.4
crash right after the first gate. `campaign_v2.csv` carries every flight; the composite's legend carries the cell's
counts.

### 1.8 The flights, third form: the environment as an axis

`scripts/env_sweep.sh` flies the replayed cadences across the environment rather than only across
speed: both gate courses (`WAREHOUSE_COURSE=a`, the course of the composite, and `b`, the unseen
gates), three obstacle densities (`--prop_density 0.20 / 0.30 / 0.40`, crates and pallets drawn
per episode), five cruise speeds (1.0–1.8 m/s), 12 seeds a cell, the fixed gain on every arm.
The walking people are 2.4 m tall in every flight (`mdp_obstacles.PERSON_H`, recorded in each
flight's data), so the vehicle has to go around them rather than over. Every flight is
recorded (`--record_dir`): the pose path, the commanded wrench, the measured body rates, the
navigation command, the gates cleared, the outcome and the settings, one `.npz` per episode under
`campaign_env/records/<arm>_<course>_d<density>_c<cruise>/`. Outcomes, here and in every campaign: a **crash** is
any contact above 1 N on the body (`illegal_contact`, threshold 1.0 in the environment config
`warehouse_nav_env_cfg.py`) or the body below 0.1 m (`root_height_below_minimum`); a **success** is the last gate
passed; a flight still airborne at the horizon is a **timeout** and is neither. `scripts/env_sweep_summary.py` turns
the wrench of each flight into rotor thrusts through the rotor model of `flight_energy_model.py`
and reports, per cell, the success fraction, the gates-cleared histogram, the mean propulsive
power and the mean commanded moment (`campaign_env/env_sweep_summary.csv`,
`refined/env_sweep.png`: success against speed, one panel per environment).

At the fixed gain the environment and speed axes alone do not separate the two arms
(`refined/env_sweep.png`): the 0.0055 gain suits the 39 Hz loop, so vanilla ROS matches or exceeds
XPU-RT in several cells (course A density 0.20, 0.58 vs 0.24 at 1.2 m/s), which is why the
separation is carried by the latency, frequency and added-load axes rather than by raw speed. The
mechanism holds even where ROS clears gates, though: through the rotor model, vanilla ROS commands
a mean moment near 0.30 and about 10 propulsive-power units against XPU-RT's 0.013-0.021 and about
1.0, roughly twenty times the commanded moment and ten times the power - the starved loop thrashes
whether or not it happens to finish. `scripts/env_crash_map.py` draws every recorded flight of one environment on the aisle, one panel
per speed, both arms overlaid, with each crash point marked, and reports next to the success count
the fraction of the course completed (the furthest gate line reached over the span from the start
line to the last gate) — a graded measure for the speeds at which both arms mostly crash
(`refined/env_crash_map_<course>_d<density>.png`).

The displayed pair of flights shares one scene: `--layout_seed` fixes the seed from which the
props and the people's positions are drawn, independently of the episode seed, and
`scripts/display_same_env.sh` tries seeds in order until, in the same scene, XPU-RT completes the
course and the baseline crashes after one or two gates; every attempt is logged next to the pair.

The follow-ups run through the same recorder (`scripts/campaign_percep.sh`, cells skipped once in
the CSV): the QoS-depth-1 ROS 2 deployment at its measured latency and cadence-only
(`campaign_qos1/`); the calibrated gain, 0.5 / replayed rate, for both main arms in the tall-people
scene (`campaign_tallcal/`); gate course C (`WAREHOUSE_COURSE=c`: a third weave drawn from a seeded
generator, `WAREHOUSE_COURSE_SEED`, default 8 = the first seed spanning at least 0.5 m laterally
with no gate-to-gate step above course A's largest and one direction reversal; `campaign_courseC/`);
and people walking at 1.5 m/s instead of the 0.8 m/s patrol (`campaign_walk/`). A third form of
replay, `--percep_latency_ms`, applies each deployment's measured camera-to-control latency as a
transport delay on the navigation decision next to its control cadence (`campaign_percep/`). With
both measurements replayed in the tall-people scene (course A, density 0.30, fixed gain), XPU-RT
CP-SAT (56.8 ms, 100 Hz) completes 2/12 at 1.0 m/s and 3/12 at 1.2 m/s; the vanilla graph
(242 ms, 39 Hz) completes 0/12 at both, ten of its twelve flights at 1.2 m/s crashing right after
the first gate, and 1–2/12 at 1.4–1.8 m/s where XPU-RT is at 0–1/12: the latency the baseline's
queues add costs it the course at the speeds the cadence alone still allowed it to enter.

A guidance net re-trained for the tall people (`scripts/retrain_tall.sh`, `nav_fused_v20_tall_cnn.pt`:
demonstrations collected with 2.4 m people, the same expert and training recipe as the shipped net,
offline yaw-sign agreement 0.954) was flown against both replayed cadences in the same scene
(`campaign_tallnet/`): 1/12 at 1.0 m/s and 0/12 elsewhere for XPU-RT, 0/12 everywhere for the baseline,
against 6/12 and 1/12 for the shipped net at 1.0 m/s. The shipped net stays the study's net; the
retrained one is recorded and not used.

Four more cells sharpen the same comparison (`scripts/queue_stronger.sh`, runner
`scripts/campaign_percep2.sh`, which also waits while a simulator launched in the last 90 s has not yet
claimed its memory). The heavier stack (`campaign_rich/`): the 90 Hz rich table — `ffn_block` at 10 Hz
and `dronet` at 30 Hz added to the chain — as CP-SAT and greedy executed it on the K1 (`xpu_b5_cpsat`,
`xpu_b5_greedy`: 57.6 / 149 ms camera→control, every frame processed) against the six-process vanilla
graph at 45 and 90 Hz cameras (`ros_rvanilla4`, `ros_rvanilla4_90`: 243 / 137 ms, 37 goals/s, the rest
of the frames dropped), cadence, latency and goal rate replayed. Twelve more seeds (1012–1023) on the
displayed cell (`campaign_seeds24/`). People crossing the aisle east–west instead of patrolling along it
(`--walk_cross`; `campaign_cross/`, tag suffix `x`): a crossing has to be seen and answered inside the
crossing time. And `scripts/select_strong_cell.py` applies the display rule across every campaign on
disk and ranks the qualifying cells by XPU-RT's own completion count, so the displayed cell is chosen
where XPU-RT is strong rather than only where the baseline is weakest; `scripts/display_same_env.sh`
takes the cell's scene and replay settings (`DENS`, `XLAT`/`RLAT`, `XHOLD`/`RHOLD`, `XGAIN`/`RGAIN`,
`XT`/`RT`, `OUTDIR`, `RENDER_OUT`).

The tuned ROS 2 layouts go through the same sweep (`scripts/sweep_tuned_ros.sh`, cells in
`campaign_percep/`): the vanilla graph with control on its own 100 Hz timer in its own process
(`vanilla4tm`: 10.0 ms control, 242 ms camera→goal, 33 goals/s at 45 Hz), the three-process pinned
layout `p3` (10.0 ms, 56 ms, 33/s) and `p3` with QoS depth 1 (31 ms), each replaying its cadence
and its latency; and the camera-rate arms — XPU-RT CP-SAT at 90 and 120 Hz against the ROS
layouts at 90 Hz — where the navigation goal is also refreshed at the deployment's measured goal
rate (`--percep_hold_ms`: 30.3 ms for every ROS layout, which stays at 33 goals/s above 45 Hz;
11.1 and 8.3 ms for XPU-RT, which processes every frame). The 120 Hz XPU-RT arm replays the
45 Hz cadence trace, its control cadence being the same measured 10.0 ms at both rates.

### 1.8b Three ways XPU-RT separates from ROS 2, all on the same eight cores

Three cells sharpen the environment story, each flown with both arms' board-measured cadence and,
where noted, their measured camera-to-control latency and goal-refresh rate (`scripts/queue_stronger.sh`,
runner `scripts/campaign_percep2.sh`; every flight recorded).

*The heavier stack* (`campaign_rich/`): `ffn_block` at 10 Hz and `dronet` at 30 Hz are added to the
chain, so five networks share the eight harts. XPU-RT's CP-SAT table (`xpu_b5_cpsat`, 57.6 ms
camera-to-control, every frame processed) clears the course at 3, 3, 4, 1 and 2 of 12 across
1.0-1.8 m/s; its greedy table (`xpu_b5_greedy`) at 0, 1, 2, 2, 2; the out-of-the-box six-process
ROS graph, whose control loop the OS spreads over all eight cores but cannot overlap frames
(37 goals/s, 243 ms camera-to-control), at 0 of 12 at every speed, and the same at a 90 Hz camera
(`ros_rvanilla4_90`, 137 ms) - a faster camera does not lift it off the 37 goals/s plateau. Adding two
networks leaves XPU-RT's completion rate intact and drops ROS to zero: the ordering CP-SAT > greedy >
ROS that the board shows holds in flight.

*People crossing the aisle* (`--walk_cross`, `campaign_cross/`): the 2.4 m people walk east-west across
the lane instead of patrolling along it, so a crossing has to be seen and answered inside the crossing
time. With each arm's measured latency, XPU-RT (57 ms) clears 3 of 12 at 1.2 and 2 of 12 at 1.4 m/s
while the vanilla graph (243 ms) is 0 of 12 at 1.0 and 1.2 and 1 of 12 at 1.4, every failure a
collision. Cadence-only (latency removed) the vanilla graph recovers to 2 of 12 at 1.0 and XPU-RT to
5 of 12: the 243 ms decision delay, not the control rate, is what the crossing scene punishes.

*More seeds on the display cell* (`campaign_seeds24/`): seeds 1012-1023 add to the twelve the cell was
chosen on, so the displayed pair's rate rests on 24 flights rather than 12.

### 1.8c The composite, third form

`scripts/showdown_v3_figure.py` (wrapper `scripts/render_showdown_v3.sh`) draws the flight pair and, beneath it,
one panel per sweep, every number read at render time and written to `<figure>_metrics.json`
(`scripts/verify_showdown_figure.py --v3-metrics` re-derives each from its source). What each panel is made of:

* **A, a–d** — the same-scene pair of §1.7/1.8 (`display_same_env.sh`, one layout seed for both arms). The
  top-down plate is the per-pixel median of the dump's overhead sequence (`ov_seq`, frames at t ≥ 2 s): static
  props survive the median, the drone and the walkers do not. The baseline's crash object is named from the
  nearest static prop to its last pose (`obst_kind`).
* **B, C** — the latency-replayed cells of course A at density 0.30 and fixed gain, `campaign_percep` and
  `campaign_seeds24` pooled: success against cruise speed with Wilson 95 % bands, and at the display speed the
  distribution of how far each arm gets (crash before G1 / after G1 / after G2 / after G3 / completed). ROS 2's
  successes are drawn where they occur (2 of 12 at 1.4, 1 of 12 at 1.6 and 1.8 m/s). Counts are fair by
  construction: the simulator is non-deterministic and arms carry replicates, so for every (cruise, seed) the
  panel keeps the first k flights of each arm, k being the fewest any drawn arm flew that seed — the same seeds,
  the same number of times, for every arm (the sidecar records how many flights this sets aside; the master table
  keeps them all). Panels F (flights) and H apply the same rule.
* **D′** — one cell per environment (course × density × people crossing), success pooled over 1.0–1.4 m/s for
  the arms flown there with their latency; a cell with fewer than twelve flights is left grey.
* **F** — the camera-rate envelope on the K1: for XPU-RT the runs of `wh_chain{30,45,60,90}` and the 120 Hz
  hyperperiod table (`trace_a…{cpsat_hard,greedy}r{1,2,3}`), camera→control = median of the warm per-frame chain,
  frames delivered = warm frames per run second, a frame late when its last dispatch ends after release + the spec
  window; for ROS 2 the roll-up `ros_traced/summary.csv` (`e2e_goal_med_ms`; delivered = `n_consumed` per second
  past warm-up — the same convention as the layers figure, 38–39/s for the 4-hart layouts; the full-run value,
  33/s, is also written to the sidecar).
* **G** — where the work lands: the share of executed inference time per hart from the traces (`worker_hart`),
  three runs pooled; a ROS 2 YOLO callback is credited to its node's pool harts (`pool_harts` in the manifest)
  together with the callback's own hart. The sampler CSVs are not used for this panel.
* **H** — the added-load bars of the layers figure (90 Hz camera + `ffn_block` + `dronet`) with the same arms'
  flights from `campaign_rich` pooled over 1.0–1.8 m/s.
* **I** — the measured Gantt rows over a 100 ms window (`make_measured_gantt_pair.py`): CP-SAT, greedy, the ROS 2
  graph with two YOLO nodes (`45_vanilla4x2`: two 4-hart pools, every core 58–88 % busy, camera→control 447 ms,
  control every 16.6 ms) and the vanilla graph. Given all eight cores the callback graph still cannot overlap
  frames.
* **J** — the drift curves of §1.9 (isolated profile vs board-calibrated, `hil_feedback_closeup.load`).
* **K** — every recorded run of the display scene (`scripts/scene_runs.sh`: twelve flights per arm with the scene's
  layout seed, cadence and latency replayed; `sweep_rate_demo.py --record_dir` stores the static props, their kinds,
  the walkers' tracks and the replay settings with each flight) drawn over the crates, racks, people and gate frames.

Variants are rendered as a matrix: the display cell × where the hand-tuned ROS 2 layout appears (`board`: panel F
and the caption; `inb`: also in B; `none`). Display cells, each a pair flown in one layout seed with both arms'
board cadence and camera-to-control latency replayed (`scripts/display_same_env.sh`, `scripts/display_pair_search.sh`,
the baseline flown first on every seed and XPU-RT retried where the baseline clears the wanted gates):

* `tall1005` — 2.4 m people, 1.0 m/s, seed 1005: XPU-RT completes, the baseline clears G1 and hits a crate. This
  is the main figure: at 1.0–1.2 m/s the cell's statistics are XPU-RT's (4/24 vs 0/24 and 3/12 vs 0/12 per seed,
  paired seeds 49 : 15 : 20), and clearing exactly one gate is the baseline's characteristic outcome there — at
  1.2 m/s it crashes after exactly one gate in 20 of 24 scenes (seeds 1000–1023), after two in 2, before G1 in 2.
* `tall1000_c1.4` — 2.4 m people, 1.4 m/s, seed 1000: the baseline clears G1 and G2 and hits the gate frame;
  XPU-RT completes; in that scene 36 recorded runs give CP-SAT 4/12, the baseline 0/12, greedy 0/12. Kept as a
  variant: across seeds at 1.4 m/s the arms are even (per seed CP-SAT 0/12, baseline 2/12), so the cell's own
  success panel does not carry the top-down.
* On the two 1.2 m/s scenes where the baseline reaches G2 (seeds 1001, 1018), XPU-RT completed none of 24
  attempts (reaching G2–G3 in most): the scenes that let the baseline through a second gate are the ones that
  are hard for everyone, so a two-gate baseline crash and an XPU-RT completion in the same scene exist at 1.4 m/s
  but not at 1.2 m/s.
* `cal17` — 1.7 m people, 1.2 m/s, gain 0.5 / control rate per arm, seed 1010: XPU-RT completes, the baseline
  clears G1; the scene's 36 runs give CP-SAT 10/12, the baseline 1/12, greedy 0/12.

### 1.8d Six more readings of the same data

`scripts/story_figures.py` draws, from the campaign records, the campaign CSVs, the ROS 2 roll-up and the board
traces (one flight per seed throughout):

* **Where the flights end** (`refined/crash_position.png`): the along-aisle position of every flight's last pose
  per arm and cruise, gate lines marked. At 0.8–1.2 m/s XPU-RT's median end lies at the third gate (≈12 m) while
  the vanilla graph's lies between the first and second (4–6 m); above 1.4 m/s the arms converge; the greedy
  table ends within the first metre at every speed.
* **Course fraction covered** (`refined/course_progress.png`): the graded version of success — XPU-RT covers
  0.57–0.69 of the course at 0.8–1.2 m/s against 0.25–0.38 for the vanilla graph (95 % bootstrap bands
  disjoint), converging above 1.4 m/s; the hand-pinned layout tracks XPU-RT.
* **The same seed under both runtimes** (`refined/seed_pairs.png`): of the seed–speed pairs both arms flew,
  XPU-RT gets further on 49, ROS 2 vanilla on 15, 20 tie.
* **Camera rate × cruise** (`refined/rate_speed_map.png`): courses completed for every camera-rate arm flown
  with its cadence, latency and goal rate.
* **The hand-tuning ladder** (`refined/ros_ladder.png`): serial YOLO → 4-hart pool → control timer → QoS 1 → two
  YOLO nodes → hand-pinned → hand-pinned + QoS 1, against the solved table, on the board at 45 and 90 Hz and in
  flight. Hand-pinning brings the chain to 56 ms and the flights to XPU-RT's level at 45 Hz; at 90 Hz every ROS 2
  layout still delivers 38–51 goals/s while the solved table delivers 72.
* **Where the latency goes** (`refined/latency_waterfall.png`): the median camera→control chain split into queue
  wait, YOLO span, YOLO→nav and nav→control from the traces — 211 ms of the vanilla graph's 241 ms is queueing
  before YOLO starts; the solved table's 55 ms is compute.

### 1.8e The atlas

`refined/warehouse_showdown_atlas.png` (`scripts/showdown_atlas.py`) puts the whole programme on one page: the
census (5,124 recorded flights in 16 campaigns, 319 ROS 2 and 249 XPU-RT runs on the K1, 64 solved tables
executed, 6 feedback studies), the same-scene pair, the camera-rate envelope, the hand-tuning ladder at 45 and
90 Hz, the latency waterfall, paired seeds, the graded course fraction, CP-SAT against greedy per camera rate,
the added load, and the ablation forest: for every condition flown by both arms (courses A/B/C, densities
0.20–0.40, crossing and walking people, the heavier stack at 45 and 90 Hz, the 90 and 120 Hz cameras, the
goal-rate hold, QoS depth 1, the two gain policies, both people heights, the retrained net, cadence-only replay,
and the hand-pinned baseline) the difference in courses completed, XPU-RT CP-SAT minus the row's baseline,
pooled over 1.0–1.4 m/s where both arms flew them, one flight per seed, with a Newcombe 95 % interval. XPU-RT
is ahead in 16 of 19 conditions (three intervals clear of zero, none behind). The summary row of the forest is the
mean of the per-condition Δ with a 95 % percentile bootstrap over conditions (seeded), and its right-hand counts are
distinct flights — each flight counted once, attributed to the first condition containing it — while the summed
per-row counts stay in the sidecar for reference; the per-condition rows keep their Newcombe intervals. The
sidecar `refined/warehouse_showdown_atlas_metrics.json` carries every number drawn, and
`scripts/verify_showdown_figure.py --all` re-derives them from the CSVs and traces.

### 1.8f The final figure

`refined/warehouse_showdown_final.png` (`scripts/showdown_final_figure.py`) orders the evidence from the specific
to the general. S: the two runtimes on one board as one flow — how each is written, where it places work, what the
board measures (ROS 2 vanilla: control every 25.6 ms, camera→control 242 ms, 38 goals/s; XPU-RT CP-SAT: 10.0 ms,
57 ms, on time to 120 Hz) and how those measurements are replayed into the flights. A–C: one flight pair in one
scene; the control-rate envelope of the paper figure (the sim-injected rate sweep, 120 flights, colour = cruise)
with the two measured control rates drawn on it — the baseline's 39 Hz sits on the floor, XPU-RT's 96 Hz above it;
the same floor on the unseen course. a–d: the moments. D–G: the pair inside — body rate, nav goal heading, forward
speed, XPU-RT's velocity through the gates. H / H′: the same scene flown twelve times per arm and a second scene
likewise (the display scene at 1.0 m/s: CP-SAT 1/12, ROS 2 0/12, greedy 0/12; the two-gate scene at 1.4 m/s: 4/12,
0/12, 0/12 — every success and collision drawn over the obstacles). I–L: the breaking point (§1.8g) and the 2.4 m scene's success against cruise, then paired seeds — same seed, same
speed, who clears more gates: in the 1.7 m scene XPU-RT further on 120 pairs, ROS 2 on 21, 27 tied (2.4 m scene
49 : 15 : 20) — where every flight ends in the 1.7 m scene, one flight per seed (pooled 1.0–1.4 m/s medians 13.5 m vs 5.7 m; 2.4 m scene 9.3 vs 5.0), the mechanism. M–Q: across conditions (the forest; its summary row is in the sidecar) and why on the board (camera rate, latency waterfall, added load, where the work lands). R: the
measured schedules. Panel M's summary row is the mean of the condition Δs with a seeded percentile bootstrap over
conditions and counts distinct flights (§1.8e); panel O's label is the chain's median camera→control latency from
the traces while its bars are the component medians (sidecar `chain_median_ms` and `sum_of_part_medians_ms`);
every count and latency is in `refined/warehouse_showdown_final_metrics.json` and re-derived by
`verify_showdown_figure.py --all`.
What the figure supports, stated as such: a decision-latency advantage (57 vs 242 ms), a frequency advantage (on
time to 120 Hz where every ROS 2 layout plateaus) and an added-load advantage; in flight the advantage shows
wherever the latency binds (up to 1.2 m/s, crossing people, added load, higher camera rates) and vanishes where
perception binds (from 1.4 m/s in the 2.4 m-people scene both arms collide with the same early crates); a
hand-pinned ROS 2 reaches the same latency at 45 Hz and cannot follow above it. The breaking-point campaign
(`scripts/campaign_break.sh`, §1.8g when it lands) asks the same question in the 1.7 m-people scene, where the
crates rather than the people set the difficulty.

### 1.8g The breaking point: the 1.7 m-people scene with the latency replayed

Question: in a scene where the crates rather than the people set the difficulty, at what cruise speed does each
runtime's camera→control latency stop fitting? `scripts/campaign_break.sh` flies course A, density 0.30, people
1.7 m (overflown), fixed gain 0.0055, with each arm replaying its measured K1 control cadence **and** its
camera→control latency (CP-SAT 56.8 ms, ROS 2 vanilla 242 ms, greedy 748 ms), 12 seeds a cell at 0.8, 1.0, 1.2,
1.4, 1.6, 1.8 and 2.0 m/s (`scripts/campaign_break_ros.sh` flies the baseline's cells on a second simulator into
the same CSV; `scripts/campaign_break_seeds.sh` adds seeds 1012–1023 to every cell of both arms; `scripts/campaign_break2.sh` repeats
at density 0.40 and adds the hand-pinned ROS 2 at 0.30).
Panel I′ of the final figure and the forest row "people 1.7 m, with latency" read `campaign_break/campaign.csv`
with the equal-replicate rule; the panel shades, per speed, where both arms complete at least half their
courses (at least a quarter each), where only XPU-RT does (the baseline below one in six), and where neither reaches one in six.
A flight still airborne at the simulator's 18 s horizon (the 24 m course needs about 22 s at 0.8 m/s; five of twelve
XPU-RT flights at 0.8 m/s ended that way with three gates cleared) is neither a success nor a collision and is left
out of the count, and at each speed the two runtimes are compared on the seeds both decided; the seed extension flies with a 30 s horizon.

Landed (courses completed / decided flights, the two runtimes compared on the seeds both decided; seeds 1000–1011 plus
1012–1023 where the extension has flown): XPU-RT CP-SAT 11/17 · 9/22 · 13/24 · 7/23 · 9/24 · 8/24 · 6/24 at 0.8 · 1.0 ·
1.2 · 1.4 · 1.6 · 1.8 · 2.0 m/s — a quarter to two thirds at every speed, no speed at which it stops; ROS 2 vanilla 0/17 ·
0/22 · 0/24 · 3/23 · 7/24 · 4/24 · 0/24 (at 1.2 m/s eight of the first twelve flights clear the first gate, one clears
three, none the course); XPU-RT greedy 0/12 at 1.0, 1.2 and 1.6 m/s — with 748 ms between camera and control every
flight hits the ground within two seconds. The 1.4 m/s baseline cell was flown twice on the first seeds (the two
simulators reached it within the same hour; the second replicate completed 2 of 12); the panel keeps the first
replicate per seed under the equal-replicate rule, and the helper was stopped once the main campaign had caught up
with it. Read as a whole: with 242 ms between camera and control the baseline completes no course at 0.8–1.2 and
2.0 m/s and a fraction at 1.4–1.8 m/s, where its stale decision happens to clear the crate after G1; XPU-RT's
completion does not depend on speed in this range. The forest row "people 1.7 m, with latency" pools 1.0–1.4 m/s:
30/72 vs 3/72, +0.38 [+0.25, +0.49]. The denser scene (density 0.40, `campaign_break2.sh`) is drawn dashed in the same panel: XPU-RT CP-SAT 3/8 · 2/10 ·
7/12 · 6/12 · 3/12 · 3/12 · 3/12, ROS 2 vanilla 0/7 · 0/10 · 0/12 · 2/11 · 1/12 · 0/12 · 0/12 at 0.8–2.0 m/s — 27/78 vs
3/76, and the 1.6 m/s tie of the 0.30 scene does not recur. The hand-pinned ROS 2 (`ros_p345.csv`, 55.8 ms, control at
100 Hz — the baseline's best configuration) flown in the 0.30 scene completes 0/6 · 4/12 · 2/10 · 5/12 · 4/12 · 6/12 ·
2/12 (23/76): between vanilla (14/158) and XPU-RT (63/158), below XPU-RT at 0.8–1.2 m/s and within its band from
1.4 m/s — the same latency reached by hand placement buys most, not all, of the difference, and the camera-rate panel
shows it cannot be carried above 45 Hz. The figure's sidecar `refined/warehouse_showdown_final_metrics.json` (`I2`)
holds the counts drawn.

The two runtimes report work at different granularities and the Gantt rows say so. XPU-RT's trace carries one row
per executed dispatch, about ninety per camera frame, each on the hart it ran on. A ROS 2 node's trace carries one row
per callback, one per frame, on the callback thread's hart; when that node runs YOLO on a worker pool the kernel is
parallelised across the pool's harts over the same interval, which the callback's duration and the per-core sampler
both show, but the trace does not record it. The bars are drawn accordingly: the traced hart solid, the pool's other
lanes hatched, and the per-lane busy percentages beside each row come from the sampler, not from the bars. Which
lanes carry the hatch is decided by the sampler too, not by the manifest's declared `pool_harts`: the vanilla nodes
run unpinned, and on that arm the operating system placed the pool's threads on harts 1, 2, 3 and 5 while the declared
hart 0 stayed idle at 3 % over the run, so only the harts the sampler saw busy are credited.

The kernel's own scaling is what makes a four-wide pool the baseline's best configuration rather than a handicap.
Alone on the board, bit-exact against golden on every run, YOLOv8n int8 takes 47.85 ms on one hart, 24.58 ms on a
four-hart pool and 25.03 ms on eight: past four harts it does not improve. `measured_timing.py --verify` re-derives
all four of those from their run logs.

### 1.8g2 What the scheduler was allowed to know about that scaling

The profile tree the chain spec costs against records one directory per hart topology --
`topo_0`, `topo_0_1`, `topo_0_1_2_3` -- and the scheduler reads the table whose tag matches the
size of a machine combination. Summed over the 90 YOLO dispatches those three tables read 55.75,
55.60 and 55.65 ms: a four-hart combination is priced like one hart. Nothing in that cost model
makes sharding worth anything, so both solvers place every dispatch on one hart, and asking for
perception inside two camera periods instead of three comes back INFEASIBLE in 1.8 s.

`scripts/emit_shard_profile.py` fills the width dimension from the board. The standalone harness
runs the same build at each width with `MODELBLASTER_CPU` set to the harts it may use, printing
one profile block per iteration; the first iteration of each dispatch is dropped and the rest
taken at the median, which is the statistic the tables already hold. Bit-exact against golden at
every width (`max_abs_err=0`, 30 iterations each), the 90 dispatches take 44.54 ms on two harts,
24.66 on four and 24.60 on eight. The single-hart table is left as it was: the pool build measures
82.85 ms on one hart because it is a different, slower kernel, and a width-1 placement does not
run it. The tables go to `gen/mb_shard/`, so `gen/mb/` and every schedule's `pdb_hash` are
untouched, and `provenance.json` next to them names the run each number came from.

Costed that way, at the same three-period window, CP-SAT returns FEASIBLE with 0 window misses and
shards -- 2869 dispatches at width 4, 1010 at width 2 -- and three board runs read 53.4, 53.0 and
53.2 ms against 56.9 for the table solved on the flat profile. Holding perception to two camera
periods as well, the list scheduler reads 36.0, 36.1 and 35.9 ms with YOLO at 23.6 ms, no queue
wait and a control gap of 9.99 ms mean and 17.02 max. Both changes are load-bearing and neither
is sufficient: with the measured costs and the original window the chain is 58.3 ms, of which
21.6 ms is queue wait, because the objective is makespan and a frame may be spread over any span
the window permits.

Two defects had to be fixed to see this, and both are the kind that leave a measurement honest
about what ran and wrong about what was asked for.

`clamp_schedule_widths.oc_of` read a dispatch's output-channel count from the top level of its IR
record, and a fused `conv2d_batchnorm2d_silu_s8` keeps the convolution's shape one level down in
`sub_ops`: 57 of this network's 63 packed convolutions returned nothing and were narrowed to one
hart by the rule that a width must divide OC. On the chain that is the difference between 36 ms
and 467 ms. Reading `sub_ops`, with the `OC16` in the module name as a second source, takes the
same schedule from 2565 targets narrowed to 0. Every greedy and soft table in the repo had been
executed with its widths removed this way, so each was re-run with them restored: the 45 Hz chain
reads 748.0 ms on all three runs against the shipped 748.0, 90 Hz 945.5/946.9/945.4 against 947.5,
120 Hz 1370.7 against 1370.7, 30 Hz 70.1 against 70.1. A list scheduler's chain here is set by how
far it spreads one frame, which sharding does not shorten, so no published number moved.

The second defect is what the implementation axis does. With `scheduler.enable_impls` on, the
solver put 268 YOLO dispatches on the IME and the board stopped at the first one -- `FATAL entry 21
of yolov8_nano_64x96 asks for impl 'ime', which this binary was not built with` -- after the solve,
the clamp, the cross-build and the deploy. The profile tree names its directories after the
implementation that was asked for rather than the one the kernel picker returned, so
`ime_x60/.../results.csv` carries a row for every dispatch of a network with no IME kernel, and all
90 of this one report `curated[rvv]/...`. The solver was choosing between two measurements of the
same kernel and reading their run-to-run noise as NPU speedup. Real `curated[ime]` kernels exist
only for `linear_s8`: two in `ffn_block`, one of them 9.4x slower than RVV, seven in `attn_block`
all slower, one in `dronet` 8x slower. `profile_loader` now requires an `ime` cost cell to come
from a row whose `implementation` names the engine, falling through to the INFEASIBLE_COST path an
op with no such kernel already belonged in, and `check_schedule_feasibility.find_missing_kernels`
fails the same schedule host-side, naming the dispatch and the row that disagrees.

Offering the axis costs something even when nothing is placed on it. Every core-group combination
is emitted once per legal implementation, which changes which combination finishes earliest in a
tie, and the list scheduler takes a worse one: 2655 dispatches at width 4 instead of 3015, 990 at
width 1 instead of 720, and a chain of 55.5-57.2 ms instead of 36.0. The deployed spec therefore
leaves `enable_impls` off, and the reason is that no implementation on this board has a kernel for
this workload's ops -- there is no conv-on-IME kernel, and the chain is 57 fused convolutions plus
a 0.08 ms MLP.

### 1.8g2b The feedback rounds re-executed, and where that changes a conclusion

The width clamp had removed shard widths from every greedy and soft table in the repo, so the
feedback-loop study's arms were re-executed with the widths their solvers chose (90 board runs,
bit-exact verify on each). No published number got worse. The soft CP-SAT arms are unchanged --
36 runs, 7 of them moving by more than run-to-run variation and all of those downward by 2 to
10 ms. The greedy arms improve, most of them slightly and three of them a great deal.

That last group changes a conclusion and it is stated here rather than left in a table. On spec B
(90 Hz camera plus the heavier stack, the `fbb5` rounds) greedy with its widths restored reads
115.2, 179.6 and 113.5 ms against CP-SAT's 265.5, 277.2 and 314.4: **the list scheduler beats the
solver on that workload**. The two schedules were clamped by the same pass and CP-SAT's is
unaffected by it -- its tables are all width 1 before and after -- so the comparison is like for
like. The mechanism is the cost model again. With a profile that prices every width the same,
sharding cannot reduce a makespan and CP-SAT, minimising makespan, correctly declines it; greedy
picks the combination that finishes its own op earliest and stumbles into widths that the flat
model says are free and the hardware says are twice as fast. On the other four specs CP-SAT still
wins by 69.5 against 859.5 ms and similar margins.

The prediction this makes -- that CP-SAT re-solved against the measured per-width tables beats
greedy again on spec B -- is not yet tested, and it is the honest next step rather than a claim.

### 1.8g3 What the flight outcome is actually sensitive to, and the effort ladder that follows

Two saturations decide every comparison in this scene, and neither is visible from the timing
table alone.

**Perception latency does nothing below about 60 ms.** The same XPU-RT arm flown over the same
cells with its measured 56.8 ms and with none at all: 18/139 against 17/140, a difference of
+0.8 points [-7.1, +8.7] over 144 paired cells. Deleting the entire perception latency is not
measurable here. It bites at 242 ms, which is why the queue-bound ROS arms lose, and it does not
bite between 28 and 57 ms, which is why no schedule improvement in that range can separate two
arms that both sit there.

**Control rate has a cliff and then a plateau.** Over the speed-by-rate grid the envelope reads
0/60 at 20 Hz, 1/60 at 25, 13/60 at 33, 23/60 at 50 and 21/60 at 100, and the cliff sits between
25 and 50 Hz at every cruise speed from 1.0 to 1.8 m/s. Above 50 Hz more rate buys nothing.

Together they place every deployment on a two-axis map, and the flight results follow it exactly.
Paired over the cells and seeds both arms flew, against the XPU-RT arm the composite draws:

| ROS 2, cumulative effort | effort | pairs | ROS | XPU-RT | separation |
|---|---|---|---|---|---|
| default, control chained in the goal callback | none | 84 | 1/79 | 9/80 | +10.0 [+2.4, +18.8] |
| control on a wall timer, its own process | one design choice | 480 | 42/479 | 73/472 | +6.7 [+2.6, +10.9] |
| the pool asked for all eight harts as well | none beyond the above | -- | 241.9 ms camera-to-goal against 242.5: the same flight | | |
| QoS depth 1 | one constructor argument | 60 | 6/60 | 6/60 | +0.0 [-11.5, +11.5] |
| taskset pinning of each node | expert | 468 | 77/462 | 72/460 | -1.0 [-5.8, +3.8] |
| pinning and QoS depth 1 | expert | 468 | 76/461 | 70/460 | -1.3 [-6.0, +3.5] |
| two model instances across the clusters | restructuring | 156 | 31/152 | 21/153 | -6.7 [-15.1, +1.8] |

The graded outcome says the same thing as the binary one, which is worth checking separately
because a threshold can manufacture a difference: mean gates reached, paired and bootstrapped over
the per-seed differences, gives +1.10 [+0.80, +1.40] against the default arm, +0.33 [+0.19, +0.46]
against the timer-driven one, and intervals spanning zero against all three tuned arms.

Every arm fails the same way -- `clip`, 99 to 100 % of crashes in all of them, ours included. What
differs is how far they get before it. Of ROS's crashes, the fraction that never clear the first
gate: 34 % out of the box, 21 % one-process, 10 % timer-driven, 5 % hand-pinned, 4 % on eight
cores, 2 % with QoS 1. Untuned ROS dies at or before gate 1; tuned ROS dies mid-course, where we
do. `scripts/crash_comparison.py` re-derives all of it.

So the claim the flights support is precise, and it is narrower than "XPU-RT flies where ROS 2
crashes": ROS 2 as written fails, and so does ROS 2 with the idiomatic fixed-rate controller and
the whole machine; one QoS argument, or hand-pinning, or a second model instance brings it level.
What separates the systems at that point is not the flight outcome but what it cost to get there
and what the timing table still shows -- 28.2 ms camera-to-goal against 30.2 at best, a chain p95
of 40.5 ms against 58 to 62, and no frames discarded against 4 to 19 %.

### 1.8h Figure 10, sized to the paper's slot

The paper's Figure 10 prints 5.2 in wide in a 7.0 in text column because its aspect ratio is set by the height the
float allows (3.6 in). `refined/warehouse_showdown_paper10.pdf` (`scripts/showdown_paper10_figure.py`) takes the full
text width — 7.0 in × 5.1 in, one row of results taller than the slot — at the paper's type size (6 pt ticks, 6.6 pt
titles; the caption is 8 pt), with eight panels and the four moments. Row 1: the specific case at the aisle's true
aspect (A, a–d), the breaking point (B: CP-SAT 63/158, vanilla 14/158, hand-pinned 23/76, greedy 0/36 in the 1.7 m
scene), the camera-rate envelope on the K1 (C). Row 2: the second scene at 1.4 m/s with all 36 flights (S: CP-SAT 4/12,
ROS 2 vanilla 0/12, greedy 0/12; the pair — ROS 2 clears G1 and G2 and hits the gate frame — bold), the paired seeds
(F: XPU-RT further on 120, ROS 2 on 21, 27 tied) and the pair's body rate and nav goal heading (G: mean |ω| 1.50 vs
0.44 rad/s at the same ground speed). Row 3: the four measured schedule rows across the full width with each arm's
chain latency and control period in its row (D: 57 ms / 10.0 ms, 748 / 13.2, 447 / 16.6, 242 / 25.6) and where the
latency goes (E). The script's `--audit` pass re-reads every drawn text after layout and reports anything off the
page, under 5 pt or colliding with another panel; the shipped render reports none. Kept only in the final figure and
the atlas: the rate-injected envelope and the course-B generalisation (superseded by C for the frequency claim), the
speed and quiver telemetry, the 1.0 m/s scene runs, the 2.4 m success curve, the forest, the added-load and hart-map
panels. Every number is read at render time; the sidecar lists them, the display strings (latencies, control
rates) come from the registry `scripts/figure_constants.py`, and `verify_showdown_figure.py --all` re-derives the
sidecar together with the atlas's, the final figure's, the v3 composite's and the story figures'.

### 1.9 The runtime feedback loop with the board in it

`scripts/hil_feedback_study.sh` runs the loop on the figure's own chain specs where the frame is tight
(90 Hz camera, 120 Hz camera, 90 Hz plus the heavier stack), three rounds each, every table executed
three times on the K1. Round 0 solves both tables from the isolated profile alone — the costs a
scheduler has before anything has run — and executes them. Each later round fits a per-dispatch
calibration from the previous round's executed CP-SAT traces and re-solves both solvers on it.

What the executed trace shows, decomposed per hart (`scripts/calibration_from_executed.py` joins every
trace record to the schedule through the IR's slot map, `k1_trace.ir_slot_map`, since the runner
numbers records by kernel-call slot and the schedule by IR dispatch id): on the K1 every dispatch
executes faster than the isolated profile predicts, while idle the plan did not leave — dispatch
launch and cross-hart dependency waits — accumulates between dispatches, so the timeline slides
behind the plan and a table that is on time in the Gantt misses its control windows on the board.
The calibration therefore measures each dispatch's service time on its hart, execution plus the
unplanned idle that followed it, as the median of service over predicted across the warm instances
of every run (`per_dispatch_multiplier`, keyed by IR dispatch id; an op-kind tier for dispatches
below the 0.05 ms floor; the aggregate as the ratio of sums). The table is the format
`--board-calibration` reads, so the re-solve is the ordinary solve on measured costs. The
`emit_board_calibration.py` path, which excludes queue delay by design and drops a network's
per-dispatch keys when the trace's numbering does not match the schedule's, is kept for the
execution-only question it answers; its rounds on the 90 Hz chain are preserved under
`hil_feedback/misaligned_emitter/` and are not the study's.

What the study establishes on the two tight specs (120 Hz, and the 90 Hz chain on its 200 ms
table). Solved from the isolated profile alone, the table executes 150-180 ms behind a 200 ms plan
with every window missed, and the harts sit idle roughly half the time (`fba120hr0`: 48 % busy,
418 ms span): the isolated profile costs YOLO at its nominal 46 ms, the board runs it near 56 ms
under load, downstream harts wait on a hand-off the plan timed as free, and the slide compounds.
The same spec, same board and same solver costed with the per-dispatch YOLO cost measured under
load (the yolo110 table of §1.5, `a120h`) executes on time: drift about zero, harts 69 % busy,
every window held. `scripts/hil_feedback_closeup.py` draws the two executed side by side with the
drift curves (`refined/hil_feedback_close_120.{png,pdf}`); this is the loop closing, and the
board-calibrated CP-SAT table holds its windows where greedy on the same spec is late everywhere
(120 Hz camera: 566 ms, 36 of 36 late).

A global service-time multiplier fitted from the isolated run's own trace does not, by itself,
bridge the two, and the study records why: the median of the per-dispatch service ratios re-solves
a table only 7 % heavier (`fba120hr1`, `fba120hr2`), the execution-time ratio of sums 14 %
(`fba120er1`), and neither reshapes the placement, because the sliding schedule under-contends YOLO
so its own trace under-weights exactly the op whose true cost must go up. The naive closed loop can
fail to leave the basin it is measured in; the per-dispatch cost measured under representative load
is the feedback that moves it. The same bias degrades a schedule that already meets its
deadlines: on the slack half-second 90 Hz table the isolated-profile schedule executes on time
(`fba90r0`: 69 ms camera->control, near-zero misses), and re-solving on the fitted service-time
calibration makes it worse (`fba90r1`: 206-291 ms), because the calibration inflates `fused_full`
by a quarter from idle that belonged to the first schedule, not to the op. The unifying result:
a service-time calibration fitted from one schedule's own trace conflates an op's cost with that
schedule's idle, so re-solving on it neither closes an infeasible table nor is safe on a feasible
one; the feedback that helps is the per-dispatch op cost measured under representative load,
separated from idle (the yolo110 table, which produces the on-time `a120h`). Both fitted variants are kept
(`scripts/hil_feedback_study_cal.sh`, tags `a120e`/`a90e`) next to the median study, and
`scripts/hil_feedback_figure.py` draws each round's solved table, executed table and per-hart drift.

The 90 Hz chain is also run on its 200 ms table (`a90h`, `wh_chain90_solve_h200`, the hyperperiod
certificate's size, as the 120 Hz spec is), where the hard-window solve is found well inside the limit;
the half-second table is re-solved with a 9000 s limit. Board use is serialised on the host by a lock
every `board_stage2.sh` holds over its runs, so two study instances never share the board.
The calibration's per-dispatch statistic is a choice the study records (`--stat median | mean`,
`--execution-only`): on the 120 Hz spec the round-0 table, solved on the isolated profile, executes
150–180 ms behind its 200 ms plan although each dispatch runs only 2 % slower than profiled — a table
with no slack lets every late hand-off cascade — and the median of the service-time ratios (about
1.02 for YOLO) re-solves a table with 7 % more cost that slides the same way, while the same spec
costed with 11–18 % more (the table of §1.5, or the execution-time ratio of sums, `a120e`) executes
inside its windows. The variant `a120e` (`scripts/hil_feedback_study_cal.sh` with
`CALARGS="--execution-only --stat mean"`) shares round 0 with `a120h` and re-solves rounds 1 and 2
on the execution-time ratio of sums; both variants are kept.
A round's CP-SAT table is the hard-window one. The soft-window CP-SAT (HEFT warm start, misses
then lateness then makespan) is solved alongside on the same costs; when the hard certificate is
not found within the round's limit, the soft table is the one executed and calibrated from, and
the round's log says so. Per round and solver the study records the solver's own prediction
(makespan, window misses, lateness), the executed camera→control latency, the control gaps, the
frames late and the worst lateness, and the per-hart drift; `hil_feedback/study.log` and the
master table carry them. `scripts/hil_feedback_figure.py --tag a90 --spec wh_chain90_solve_500` draws
the rounds one above the other: the table as solved, the same window as the K1 executed it, and the
executed-minus-planned start of every dispatch against planned time per hart, with the round's frames
late, control-window misses and camera→control median (`refined/hil_feedback_<tag>.png` + metrics). The study covers the 90 Hz and 120 Hz chains on their 200 ms tables, the 90 Hz chain on its 500 ms table, the execution-mean calibration variants (`a120e`, `a90e`), and the heavier-stack chain (`b5`, `wh_chain90_rich_solve_500` with `ffn_block` and `dronet`): every hard CP-SAT round is feasible (the rich and half-second tables need the 9000 s limit), and on none of them does the trace-fitted global recalibration close the loop, while CP-SAT stays ahead of greedy each round.

## 2. Ablation of the experiments: which rungs, and why

### 2.1 The problem the ladders solve

The 24 runnable K1 specs are bimodal, and neither mode tests a scheduler. Twelve have a
baseline that already meets every deadline — nothing at stake, so the loop can only win
on terms nobody reads. Most of the rest ask for something no schedule can deliver:
`yolov8_nano_64x96` needs 23.95 ms at 8 cores against a 22 ms window and its core scaling
has saturated (4→8 cores buys 1.6%), so 44 of the 50 residual misses in the full ablation
were infeasible by construction. Almost nothing sat in the band where a scheduler decides
whether the deadline is met.

`scripts/make_scaling_workloads.py` builds rungs that do. Every rung is constructed so
the singleton baseline **misses**, and a schedule that meets every deadline **exists**
using implementations already measured on the board — so a failure is the loop's, not
physics'.

```bash
.venv/bin/python scripts/make_scaling_workloads.py --out-dir data/toplevel/scaling --check
```

### 2.2 Five ladders, each isolating one variable

| ladder | varies | sized from | why it exists |
|---|---|---|---|
| `LADDER` (w2–w5) | net count | profiles | the first ladder; confounded (see below) |
| `LADDER_COMPOSITION` (c2–c5) | net count, `dronet` at the top only | profiles | isolates `dronet` to one row |
| `LADDER_BOARD` (b4, b5) | net count | **board** | profiles were wrong about what fits |
| `LADDER_BOARD_SLACK` / `_HOLDS` (b5x, b2y–b5y) | slack, cold start | board, 1-core cold | a window is only meetable at the width the solver *picks* |
| `LADDER_SOLVABLE` (s5) | dispatch count | board + CP-SAT tractability | reachable *and* solvable |

Each ladder isolates one variable the previous one left confounded; the docstrings in
`make_scaling_workloads.py` carry the arithmetic. In brief:

* **`LADDER` confounded two things.** It introduces `dronet` at rung 3, and `dronet`'s conv
  dispatches take different core widths across their instances, so `shard` — the only
  lever in the whole ablation that ever clears a deadline — was refused as *unbuildable*
  on w4 and w5. All ten contract violations were `dronet`. The ladder therefore varied
  "more networks" and "contains the network that blocks our best lever" together, and the
  flat w4/w5 rungs were read as a scaling limit when the evidence pointed at codegen.
  `LADDER_COMPOSITION` puts `dronet` in exactly one rung so its effect is a single row.
* **Profile-sized windows do not survive execution.** `ffn_block` misses a 10 ms window on
  the board at its *fastest* measured width (10.03–10.34 ms against a 7.72 ms profile),
  and yolo misses 26 ms by 1.64×. The "0 infeasible misses" classification said otherwise
  only because it trusted profiles. `LADDER_BOARD` re-sizes from measurements.
* **A window is meetable only at the width the loop actually converges on.** `b5x` reaches
  0 predicted misses and does not survive execution: its windows were sized from
  measurements taken while `dronet` and `yolo` were *widened*, but the loop converged on
  `shard:ffn_block` alone and left both at one core. `LADDER_BOARD_HOLDS` sizes from
  1-core cold measurements — the conservative width, the one the loop falls back to.
* **Cold start is not free and is not a scheduling failure.** The first instance of a
  network pays it: `fused_full` runs 2.66–2.74× its warm median on instance 0 (11.7 ms
  against 4.28 warm). A 5 ms period cannot absorb that in any schedule. Deploying at that
  rate needs a warm-up pass, not a better scheduler; the rungs state it as a window.

### 2.3 Sizing for the solver, not only for the board

A scan of the 204 CP-SAT certificates in this repo says model size decided more of our
results than any scheduling question:

* a phase-1 objective of **0 is OPTIMAL in 55 of 55 runs** — when a zero-miss schedule
  exists the bound is matched trivially. All 121 FEASIBLE runs have objective > 0 and a
  genuine open gap. CP-SAT is fast at *confirming* an achievable target and slow at
  *disproving* an unachievable one;
* **n = 492 dispatches is never OPTIMAL (0/33)**, and above 300 only 2 of 73 files ever
  proved phase 1. The single yolo instance — 98 dispatches on its own — is exactly what
  takes w4 (394) to w5 (492);
* between those, node count is a weak predictor: n = 214 is mostly FEASIBLE while the
  larger n = 217 and n = 254 are mostly OPTIMAL. Hardness, not node count.

Dispatches per instance, measured: `mlp_control` 7, `fused_full` 15, `ffn_block` 5,
`dronet` 21, `yolov8_nano_64x96` 98. `s5_solvable_reveal` is sized to 215 dispatches from
these — the size at which the working rung sits.

**The consequence for reading w5:** its re-solve does not pay off, and the honest
statement is not that the outer loop fails at five networks. It is that at 492 dispatches
CP-SAT never converges, so the re-solve is optimising against an unproven bound. `s5` is
the same five networks sized so the solver can answer, which is why it is the rung to use.

### 2.4 Gate a rung before running it

```bash
.venv/bin/python scripts/gate_rung.py \
    --spec data/toplevel/scaling/s5_solvable_reveal.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --sweep-net ffn_block --windows 34,28,24,21,20,18 \
    --pin yolov8_nano_64x96=30.0 \
    --shard-sets yolov8_nano_64x96 yolov8_nano_64x96,ffn_block
```

Two gates, both empirical:

1. **at stake** — the solved baseline, on profile costs, misses at least one deadline;
2. **reachable** — some lever set, on *measured board* costs, reaches zero.

Both must pass on the same row. Gate 2 also buys tractability, per §2.3: a reachable rung
is also a solvable one.

**Why this is empirical and not arithmetic.** Three analytical gates are available and
each rests on a model of the baseline that the measurements do not bear out:

* `one > window` ignored that a window may be wide for an unrelated reason (absorbing the
  t=0 cold-start burst), which then also clears the baseline;
* `one > period` is false on a multicore machine — successive *instances* are independent
  and the scheduler puts them on different harts. CP-SAT duly returned a zero-miss
  baseline by spreading `ffn_block`'s three instances;
* comparing `net_times()` to the window ignores that the scheduler parallelises a
  network's *dispatches* across cores even at width 1 each, so yolo's 47.73 ms serial sum
  never lands on a single core at all.

Solve it and look.

One more distinction the tool enforces, because hand-tuning kept breaking it: **period and
window are different knobs.** The period is what puts a rung at stake — `ffn_block` at
period 20 ms against a 26.61 ms single-core time means one core cannot sustain the release
rate, so widening is forced whatever the window says. The window is what makes it
reachable — 34 ms, wide enough that the widened schedule survives the t=0 burst when all
five networks release together. An earlier version sized the window at 28 to absorb the
burst, which also put it above the 26.61 ms single-core cost, so the baseline fitted on
one core and the band check rejected the rung as "nothing at stake".

---

## 3. Ablation of the feedback: which loop is doing the work

### 3.1 The design

Showing one workload where the pair helps does not separate them. `ablate_feedback_loops.py`
runs the 2×2, per solver:

| cell | inner (levers) | outer (solve on measured costs) | |
|---|---|---|---|
| **A** | no | no | the naive deployment |
| **B** | **yes** | no | offline co-design, deployed blind |
| **C** | no | **yes** | measure-and-re-solve, no co-design |
| **D** | **yes** | **yes** | both |

* **inner** = AOT co-design with ModelBlaster. Graph rewrites and the per-dispatch
  implementation choice, decided offline against isolated per-dispatch profiles. The board
  may be used here, but only as a **profiler**.
* **outer** = HIL. Costs observed while the whole schedule runs in real time, returned as
  measured multipliers and re-solved against. The board here is the **runtime**.

**Every cell is scored on board costs**, because that is what silicon does. The cells
differ in what the *scheduler knew*, not in how they are judged: A and B are solved
against predicted costs and then re-cost on the measured multipliers with their assignment
held fixed — exactly what deploying them means — while C and D are solved with
`--board-calibration`, so re-costing them again would apply the multiplier twice.

Scoring is instance-level (`xpu-rt/schedule_eval.py`): an instance misses when its last
dispatch ends past `inst*period + window`.

### 3.2 Running it

```bash
.venv/bin/python scripts/ablate_feedback_loops.py \
    --workloads data/toplevel/scaling/s5_solvable_reveal.json \
                data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --calibration results/codesign_feedback/k1_cal_s5_measured.json \
    --solvers cpsat,greedy --cpsat-time-limit 150 --repeats 3 \
    --out-dir results/loop_ablation_postfix
```

`--repeats` applies only to a solve whose result can *move*: a CP-SAT solve returning
OPTIMAL has a unique objective and runs once; one returning FEASIBLE is budget-truncated
and is repeated, reported as median with spread. Greedy is deterministic and always runs
once. `--one-per-family` collapses the 25 K1 specs, which contain byte-identical
duplicates and variants that return bit-identical results — the family is the real unit.

### 3.3 Which grid to read

There are **thirteen** ablation directories in `results/`, one per revision of the
accept path; the earlier grids are kept as the record of how the answer moved with it.

**Read `results/loop_ablation_postfix/`.** It is the grid run with the current accept path
— a zero tolerance on the deterministic miss count (`DETERMINISTIC_TOLERANCES`), the winner
ranked by misses rather than by makespan (term 7 of 9), and the CP-SAT budget split. The others: `loop_ablation` is the broad 24-workload greedy sweep,
`_ladder*` are the scaling rungs, `_v4_*`/`_v5_big` vary model size, and `_pair`,
`_stable` and `_final*` are intermediate grids kept so a changed number can be traced to
the change that caused it.

Each `ablation_summary.json` carries its own `cpsat_time_limit_s`, `cpsat_workers`,
`solve_env` and `codegen_contract`, so a cell cannot be silently compared against one
solved under different conditions. Check them before comparing across directories.

### 3.4 What it found

Instance misses on board costs, `loop_ablation_postfix`:

| workload | solver | A | B | C | D | inner levers chosen |
|---|---|---:|---:|---:|---:|---|
| `s5_solvable_reveal` | greedy | 1 | **0** | 1 | **0** | `shard` |
| `s5_solvable_reveal` | cpsat | 1 | 1 | 1 | 1 | *(none accepted)* |
| `w4_ffn_dronet_sensor` | greedy | 10 | **4** | 10 | 8 | `shard:ffn_block` |
| `w4_ffn_dronet_sensor` | cpsat | 23 | 10 | 28 | 26 | `shard:dronet`, `shard:fused_full` |

Three things this says, stated as plainly as they deserve:

**The inner loop carries this result.** B beats A on every row that moves. On these two
at-stake workloads, offline co-design against isolated profiles is what clears deadlines.

**The outer loop alone does approximately nothing here.** C ≈ A throughout. Re-solving
against measured costs without any lever to pull does not help, which is unsurprising in
retrospect: better cost estimates do not create an implementation that fits.

**D is not better than B, and on `w4`/greedy it is worse (8 vs 4).** We do not have an
explanation we are confident in, and it is reported rather than smoothed. Note the
tension with §1.4, where the *arcs* show the outer loop recovering board-revealed misses
on `sensor_evo_auto` and `s5` — the arc and the ablation ask different questions (does the
re-solve recover what the board revealed, vs. does the pair beat the inner loop alone),
and the honest summary of the pair, on this population, is that it does not.

**CP-SAT loses to greedy in every cell** (`exact_vs_greedy_per_cell`: B, C, D all 0 wins /
2 losses; A 1–1). At these sizes the exact solver does not earn its cost against the
heuristic. That is a statement about these two workloads at this budget and this
dispatch count, not about exact scheduling.

**n = 2.** The at-stake population is two workloads. These are directional findings from
a small sample, and the ablation reports `aggregate_at_stake_only` separately from
`aggregate_achievable_only` precisely so a reader can see how few rows carry the claim.

### 3.5 Separating a miss the loop could have cleared from one it could not

`achievable_means` in the summary: *a miss is ACHIEVABLE when the net's fastest measured
implementation fits its window; otherwise no schedule can meet that deadline and the miss
is a compiler gap, not a scheduling one.* `total_infeasible_misses` is 0 across every cell
in `postfix`, which is what makes those rungs fair tests — and is exactly the property the
profile-sized rungs lacked (§2.2).

---

### 3.6 Two gain policies for the flight envelope

The envelope holds `moment_scale` at 0.0055 on every cell — one controller, as deployed, at
whatever cadence it is given — so a cell at a low rate is also a cell with less authority.
`scripts/gain_controlled_grid.sh` runs the same 5 speeds × 4 rates × 12 seeds with the gain
calibrated per cell (0.5/eff_hz, `results/codesign_feedback/gain_controlled/`), and
`scripts/hil_envelope_panel.py` draws both when both exist, computing the floor's Fisher
p-value from the cells at plot time. A floor that holds under both policies is a property of
the rate.

## 4. What these measurements do not establish

* **Single board, single unit.** All nine runs are one SpaceMiT K1. Nothing here separates
  silicon-to-silicon variation from anything else.
* **The calibration is not transferable.** Measured per rung, and demonstrated not to
  transfer: `b5z`'s table mispredicted `s5` by 1.66×. Treat a multiplier as a statement
  about the workload it was measured on.
* **Op-kind and aggregate tiers are predictions.** Only `coverage.nets_exact` is measured
  per dispatch. A net costed by its op kinds is being predicted, and the JSON labels it.
* **Queueing is excluded from the calibration by construction.** It is in the trace and in
  the attribution, but not in the multipliers.
* **The ablation population is two at-stake workloads.** See §3.4.
* **The analytical baseline is a serialization policy model (Tier A in `docs/Baselines/ros_baseline_tiers.md`).**
  Naming is precise throughout: the arm is **static per-node serial pinning**, and it comes in three forms.

  1. *Scheduling* (`scripts/ros_pinning_generic.py`, `ros_pinning_periodic.py`) — takes
     XPU-RT's measured per-dispatch durations and re-lays them under a one-node-per-network,
     one-hart, sequential-graph, periodic-timer policy. Every `schedules/cmp_*_board.json`
     is this; the `_board` suffix means board-*calibrated costs*, not board-*executed*.
  2. *Flight sim* (envelope, showdown) — in-sim ZOH latency injection, parameterised by
     `sched_latency_ms` and the command-refresh rule `ceil(latency / control_dt)`.
  3. *micro-ROS on FireSim* — a different target, reported as such and never as K1.

  **What this supports.** The baseline pays the *same measured per-op costs* as XPU-RT, so
  the comparison isolates placement policy and cannot be dismissed as a slow build or
  unoptimised kernels. On the compute side it is a best case for the baseline. The claim is
  *"static per-node serial pinning loses to global scheduling at equal per-op cost"*.

  **What it does not support on its own.** *"ROS 2 loses to XPU-RT on the K1"* — that claim rests on
  the measured arms (Tier C: `ros_traced/`, `docs/Baselines/ros_arms_catalog.md`, the warehouse census earlier on this page).
  The model charges nothing for DDS serialization, message copies, executor wake-up or
  callback jitter, which would make a real deployment **slower** than this arm; it also
  assumes single-threaded per-node execution, and a multithreaded executor, callback groups
  or composed nodes would be **faster**. Only the first direction is bounded.

  **Bounding it.** `scripts/ros2_middleware_tax_k1.py` measures the middleware cost of a
  3-hop chain on the board under both executors; `scripts/run_ros_tax_when_free.sh` runs it
  once the board is idle. The static-pin arm's 35.58 ms end-to-end on the coupled chain is
  exactly its serial compute sum (30.592 + 4.905 + 0.081), so it carries no queueing or
  middleware term, and a positive tax makes it a lower bound.

* **`yolov8_nano_64x96` core scaling is measured only in contention.** In the five-network
  runs yolo was sharded per dispatch (48 dispatches at width 4, 17 at width 2, 33 at width
  1 — never width 8) while contending with four other networks. That is not the standalone
  1/2/4/8-core sweep that Figure 14's extrapolated multipliers would need to be confirmed
  against, and it should not be presented as one. `MB_CORES` in
  `ModelBlaster/scripts/run_model_k1.sh` supports the clean experiment; it has not been run.

---

## 5. Environment and gotchas

| variable | what it does |
|---|---|
| `XPURT_PY` | interpreter for the scheduler and analysis (default `$REPO/.venv/bin/python`) |
| `ISAAC_PY` | interpreter for the flight sims (HIL figures only) |
| `XPURT_NO_COMPACT=1` | keep the schedule as solved; the miss count is the solver's answer, not a post-pass's |
| `XPURT_UNIFORM_PACKED_WIDTH=1` | a packed-weight dispatch takes one width across its instances — the codegen contract's requirement |
| `XPURT_CPSAT_WORKERS` | CP-SAT workers. 4 for cells, 1 under `--replay` for bit-exact reruns; **mixing the two inside one row confounds inner-vs-outer with worker count** |
| `XPURT_SHARD_ONLY_NETS` | restrict the shard lever to named nets. Published *before* any solve — a lever published only at the CP-SAT call site never reaches `--solver greedy`, which made per-net shard candidates come out byte-identical |
| `MODELBLASTER_K1_HOST` | ssh config entry for the board (default `k1`) |
| `CROSS` | SpaceMiT cross toolchain, set by `scripts/setup_spacemit_toolchain.sh` |

Two more that cost real time:

**Do not use `pgrep -f` to wait on your own jobs.** It matches its own command line. It
killed the shell mid-heredoc three times in this study and lost a script. Use PID files.

**Absolute paths under `results/` are records, not instructions.** They say what was
actually run. The paths in `scripts/` are repo-relative, and machine-specific locations
come from `scripts/env.sh` (`docs/Artifact/environment.md` §Environments).

---

## 6. Where the artifacts are

The end-to-end recipe — host, simulator and board, in order — is `docs/Artifact/reproduction_full.md`.

**The master table.** `scripts/build_master_csv.py` writes `results/codesign_feedback/master_runs.csv`,
one row per run of everything above — every board run of a ROS 2 layout (`ros_traced/summary.csv`
rows), every executed XPU-RT table (its trace re-read for camera→control, control gaps and frames
late), every solver certificate and table solve, every flight of every campaign (with the rotor-model
power and commanded moment where the flight was recorded) and every energy flight — each with
`config` and `config_description` saying in words what that configuration is, plus `source`, the
artifact the row was read from. Each flight row also carries the scene (course and gate positions, density, people height,
walking speed, layout seed, nav weights), the outcome in full (gates, steps, flight time, crash type,
crash position and time, progress, path length, the recorded path file), the rotor-model power and
energy over the flight and up to the moment its comparable run crashed (`pair_run_id`: the other
main deployment flown in the same scene, speed, gain and seed), the navigation command statistics,
and — through the cadence trace's board run — the deployment behind it: the ROS 2 layout's executor,
processes, pinning, pool, QoS and measured cadence, or the XPU-RT table the board executed with its
solver, the solver configuration, the spec's windows, the predicted metrics, the board metrics and the
counterpart tables for the same spec (greedy, CP-SAT, the HIL feedback rounds). `family` groups runs
by campaign or study, `cell` by direct comparability, `cell_runs` and `related_runs` name the rows a
run is related to. `master_configs.csv` lists every configuration once with its run count. The table is rebuilt every half hour while campaigns run (`scripts/refresh_master_csv.sh`)
and can be rebuilt at any time from the artifacts on disk.

```
results/codesign_feedback/
  k1_cal_<rung>_measured.json     the nine board calibrations (§1.3)
  <rung>_miss_attribution.json    execution-bound vs queueing (§1.4)
  <arc>/                          per-arc loop output, panels, Gantts
  loop_overview_2band.{png,pdf}   the figure, with its _metrics.json sidecar
results/loop_ablation_postfix/    the 2x2 grid to read (§3.3)
results/loop_ablation*/           twelve superseded grids, kept as the record
results/loop_sweep/<workload>/    25 workloads taken through the loop end to end
data/toplevel/scaling/            the generated rungs
```

Every `loop_report.json` names the round, the lever, the objective terms that decided and
whether the candidate was accepted — so a claim of the form "the loop chose `shard` here"
can be checked against the round that made the choice rather than taken on trust. Every
figure has a `_metrics.json` sidecar carrying the numbers it drew.
