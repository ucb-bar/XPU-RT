# The warehouse showdown with the ANALYTICAL ROS 2 pinning baseline

This page reproduces the form of the showdown whose baseline is a **schedule model** (Tier A in [`ros_baseline_tiers.md`](../Baselines/ros_baseline_tiers.md))
rather than a board measurement.

| stem | where | baseline | sidecar |
|---|---|---|---|
| `warehouse_showdown_story` | `results/codesign_feedback/refined/` | modelled per-node pinning | none (allowlisted) |
| `warehouse_showdown_board` | `results/codesign_feedback/` | modelled per-node pinning, board-recost | `warehouse_showdown_board_metrics.json` |
| `warehouse_schedule_board` | `results/codesign_feedback/` | the same pair, schedule panel alone | `warehouse_schedule_board_metrics.json` |

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md),
[`figure_runbook.md`](figure_runbook.md) §3,
[`warehouse_hil_figure_reproduction.md`](warehouse_hil_figure_reproduction.md) for the schedule
solve this model re-lays, [`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md) for every **measured** ROS 2
arrangement, and
[`showdown_submitted_config_reproduction.md`](showdown_cam45_ros_out_of_box_reproduction.md) for the
same figure configuration flown against a measured baseline instead of this one.

---

## 1. What the model is

One ROS 2 node per network. Each node owns a **static partition** of harts and is released by its
own periodic timer at `k*T`. Inside the node the network's whole dispatch graph runs
**sequentially** — a node is a process with one executor thread, so its graph is not spread across
harts the way the scheduler spreads it. Per-dispatch durations are the **same measured K1 numbers
the XPU-RT arm pays**, so the two arms differ only in placement policy, never in kernel cost.

For a node with one-hart serial compute `C`, period `T`, instance `k`:

```
release_k  = k * T
start_k    = k * T                            # executor policy "release"
start_k    = max(k * T, finish_{k-1})         # executor policy "queued"
finish_k   = start_k + C
response_k = finish_k - release_k
late_k     = response_k > T
```

and the command cadence a response of that length sustains under the zero-order hold at the
simulator's `dt = 10 ms` control step:

```
rate_hz = 1000 / (dt * ceil(response / dt))
```

Two executor policies are offered: the committed schedule places instances by release time, and the
script that shares its name serialises them. Both are computed so their difference is visible:

* **`release`** places every instance at its release time even when the partition is still busy.
  This is what `schedules/scheduled_ros_partition_deployed.json` — the schedule the Tier A showdown
  drew — contains: YOLO starts at 0, 22, 44, 66, 88 ms while each frame takes 24.353 ms, so
  consecutive frames overlap on the same partition. The per-frame overrun is real and drawn; the
  growing queue is not modelled.
* **`queued`** serialises instances on the partition, which is what a single-threaded executor does
  and what `scripts/ros_pinning_periodic.py` writes. It is the stricter reading and it makes the
  baseline slower (makespan 121.767 ms, worst perception response 33.767 ms) than the `release`
  reading the Tier A showdown drew.

### What it assumes

1. Per-dispatch durations are the K1-calibrated numbers of the source schedule. The model adds no
   compute and removes none.
2. A node's graph is serial — no intra-node sharding, no cross-hart split of one network.
3. Partitions are static for the whole horizon. Nothing migrates.
4. Releases are strictly periodic and jitter-free.
5. **Middleware is free.** No DDS serialisation, no message copy, no executor wake-up, no callback
   jitter. `scripts/ros2_middleware_tax_k1.py` exists to bound exactly this omission and measures
   the tax on the board. The omission makes the modelled arm *faster* than a real deployment, so
   every "the baseline misses" claim built on it is conservative.
6. **But** the model also assumes one executor thread per node. A real deployment with a
   `MultiThreadedExecutor`, callback groups or composed nodes can overlap what this serialises and
   would be *faster* than the model. That direction is **not** bounded by the tax measurement, and
   it is why the measured ROS 2 arms exist at all
   ([`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md), `measured_timing.ROS_VANILLA`).
7. A wider perception partition helps only through a **measured** whole-net speedup table, never
   through core count. Sharding this detector is inefficient (1.42× at eight harts; four harts are
   slower than two), and that inefficiency is applied in the baseline's favour.
8. Deadline = period. A frame is late when its response exceeds its own release period.

### Analytical, measured, neither

| quantity | which |
|---|---|
| per-dispatch durations, perception sharding speedup table, middleware tax | **measured** (K1) |
| placement, release/queue arithmetic, every response and makespan that follows, chain latency, command cadence | **analytical** |
| the two worst-case control responses behind the "100 Hz vs 50 Hz" labels (4.89 / 12.40 ms) | **neither** — see §6 |

---

## 2. Where each input comes from

The model's inputs are the three deployed nets as ROS 2 nodes. They are held once, in
`scripts/ros_pinning_model.py:DEPLOYED_NODES`, and `--verify` re-reads each of them out of the
schedule they were taken from:

| node | role | one-hart compute `C` | period `T` | instances | partition |
|---|---|---|---|---|---|
| `mlp_control` | control | 0.083000 ms | 10 ms | 12 | `CPU_E#0` |
| `fused_full` | nav | 3.621874 ms | 20 ms | 6 | `CPU_E#1` |
| `yolov8_nano_64x96` | perception | 24.353374 ms | 22 ms | 5 | `CPU_P#0..3` |

* **Compute** is the sum of that instance's dispatch durations in
  `schedules/scheduled_ros_partition_deployed.json`, which carries the K1-calibrated per-dispatch
  numbers produced for the deployed spec
  (`data/toplevel/networks_k1_flight_deployed.json`; the solve is
  [`warehouse_hil_figure_reproduction.md`](warehouse_hil_figure_reproduction.md) §3). It is uniform
  across instances, which `--verify` checks.
* **Period** is the deployed rate spec: CTRL 100 Hz, NAV 50 Hz, YOLO pipelined at 22 ms.
* **Partition** is the ROS-style fixed assignment: nav and control each on one E hart, the detector
  on the whole P cluster. Six of eight harts carry work; the two the Tier A showdown labels
  "idle — core unused" are `CPU_E#2` and `CPU_E#3`.
* **Camera period** enters as the perception node's release period (`--camera-hz`).

The perception sharding table (`PERCEPTION_SPEEDUP`) is the one
`scripts/ros_pinning_generic.py:63` applies: measured standalone on this board at 1/2/4/8 harts
(55.870 / 50.384 / 52.294 / 39.386 ms).

---

## 3. Re-deriving the numbers

No render and no hardware:

```bash
.venv/bin/python scripts/ros_pinning_model.py                 # the deployed stack, release policy
.venv/bin/python scripts/ros_pinning_model.py --executor queued
.venv/bin/python scripts/ros_pinning_model.py --camera-hz 30 --perception-width 4
.venv/bin/python scripts/ros_pinning_model.py --json
```

The default run prints exactly what the Tier A showdown's panel I draws:

```
executor release   harts 6/8 used, 2 idle   makespan 112.353 ms
node                  role           C (ms)   T (ms)  worst resp    late  harts
mlp_control           control         0.083    10.00       0.083   0/12   CPU_E#0
fused_full            nav             3.622    20.00       3.622   0/6    CPU_E#1
yolov8_nano_64x96     perception     24.353    22.00      24.353   5/5    CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3

camera->control 28.058 ms   commands 33.3 Hz under the 10 ms ZOH   perception sustains 41.1 Hz
```

An arbitrary stack can be modelled without touching the module:

```bash
.venv/bin/python scripts/ros_pinning_model.py \
  --node perception:yolo:24.353:22:5:CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3 \
  --node nav:nav:3.622:20:6:CPU_E#1 \
  --node control:ctrl:0.083:10:12:CPU_E#0 \
  --executor queued --control-on-timer
```

---

## 4. The render command

The modelled figure is drawn by `sims/scripts/compose_warehouse_showdown.py`. It needs no board and
no Isaac interpreter — the host `.venv` is enough — but it does need the two committed flight dumps
under `results/codesign_feedback/crash_demo/` (provenance in that directory's
`trace_manifest.json`).

```bash
.venv/bin/python sims/scripts/compose_warehouse_showdown.py \
  --xpu-dir results/codesign_feedback/crash_demo/complete_figdata \
  --ros-dir results/codesign_feedback/crash_demo/crash_figdata --rot 0 \
  --x-resp-ms 4.89 --r-resp-ms 12.40 \
  --sched-xpu schedules/scheduled__flight_deployed_matched_board_cpsat_profiled.json \
  --sched-ros schedules/scheduled_ros_partition_deployed_matched_board.json \
  --out    <new-stem> \
  --schedule-out <new-schedule-stem>
```

`--sched-xpu` / `--sched-ros` are shown explicitly although they are the defaults, because they are
what makes the baseline modelled rather than measured. `--x-resp-ms` and `--r-resp-ms` are the panel
A/B command-rate inputs of §6 — they are **required** arguments with no default, deliberately.
`--frame-instance` (default 2) selects the interior perception frame both arms are compared on, and
`--frame-budget-ms` (default 23) is the period/deadline both are held to.

Re-rendered to a scratch stem against the committed inputs, the schedule panel's sidecar comes back
**identical on all 17 keys** to the committed `warehouse_schedule_board_metrics.json`. Pick a new
stem rather than the committed one: existing figures are not overwritten.

---

## 5. The verification block

```bash
.venv/bin/python scripts/ros_pinning_model.py --verify
.venv/bin/python scripts/verify_panel_i.py \
    --spec data/toplevel/_flight_deployed_2frame.json \
    --xpu  schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json \
    --ros  schedules/scheduled_ros_partition_deployed.json \
    --out  "$TMPDIR/ros_truncated.json"
bash artifact/verify_no_hardware.sh
```

`ros_pinning_model.py --verify` re-derives, at **0 DRIFT**:

| check | value |
|---|---|
| ROS pinning makespan, from the artifact **and** from the model | 112.353 ms — panel I's "112 ms" |
| ROS perception response, artifact and model | 24.353 ms against a 22 ms deadline, 5/5 frames late |
| harts carrying work / idle, artifact and model | 6 / 2 — panel I's "static · 6 cores" and its two "idle — core unused" lanes |
| XPU-RT makespan drawn beside it | 40.379 ms — panel I's "40 ms" |
| every model input, re-read from the schedule it was taken from | compute, instance count and partition per node |
| board-recost frame response, XPU-RT / ROS | 22.9998 / 49.7648 ms against the 23 ms budget, matching both committed sidecars |
| ROS pinning makespan, board-recost | 153.524 ms |
| the ZOH arithmetic behind the drawn labels | 4.89 ms → 100 Hz, 12.40 ms → 50 Hz |

`verify_panel_i.py` is the separate horizon check: the two schedules the Tier A panel I draws do
**not** cover the same releases (XPU-RT 4/2/2 against ROS 12/6/5 CTRL/NAV/YOLO instances), so it
truncates the longer one and rescores. The horizon-independent statement is the deadline outcome —
the baseline misses 100 % of perception instances at both horizons, the scheduled arm none — not the
makespan ratio. It prints:

```
schedule                              makespan         work   disp  misses
XPU-RT (CP-SAT)                        40.38ms      57.4cms    254  0 {}
ROS truncated (same horizon)           46.35ms      56.3cms    238  2 {'yolov8_nano_64x96': 2}
ROS as plotted                        112.35ms     144.5cms    624  5 {'yolov8_nano_64x96': 5}
```

so the same-horizon makespan ratio is **1.15×** (the two drawn bars span 2.8× because their horizons differ), and the
claim to carry is the miss count. `--out` is required because the default landing place
(`results/codesign_feedback/tmp/`) is not in the tree. The board-recost pair used by
`warehouse_showdown_board` is matched by construction (5/6/12 on both sides) and its sidecar records
`horizons_matched: true`.

`artifact/verify_no_hardware.sh` now runs `ros_pinning_model.py --verify` alongside
`measured_timing.py --verify`, so a modelled number drifts as loudly as a measured one.

---

## 6. Caveats — which panels rest on a model rather than a measurement

* **Panel I is a model.** The composer's form prints the footer "Calibrated model, not a board trace" (`sims/scripts/compose_warehouse_showdown.py:379`). Every bar in the baseline row is placed by the
  arithmetic of §1, not stamped by a board. The *durations* inside those bars are measured; their
  *placement* is not.
* **The "100 Hz vs 50 Hz" labels on panels A and B are neither measured nor derived here.** They
  come from two worst-case control responses — 4.89 ms for XPU-RT and 12.40 ms for the baseline —
  that enter the composer as inputs (`--x-resp-ms` / `--r-resp-ms`) and are recorded in
  `results/codesign_feedback/refined_src/warehouse_regen_metrics.json`. They come from
  `results/codesign_feedback/microros_baseline_k1/microros_baseline_k1.json`, a prediction of the
  per-network-pinning policy on the four-network `networks_k1_tri_exact_100ms.json` workload (dronet
  12.403 ms, XPU-RT shard 4.890542 ms), not from either schedule named in that sidecar. The ZOH step
  that turns them into 100 Hz and 50 Hz is re-derived by `--verify`; the two responses are inputs.
* **The board-recost schedule is not re-derivable from the recurrence.**
  `schedules/scheduled_ros_partition_deployed_matched_board.json` carries per-frame compute of
  30.705 ms, from which the queued model predicts 46.114 ms and the release model 30.705 ms for
  frame 2 — against the 49.765 ms the artifact carries and the panel prints. The recost re-times an
  existing schedule by per-dispatch multipliers rather than re-running the policy, so it opens
  intra-frame gaps the recurrence does not model. `--verify` prints all three numbers so the
  divergence is visible and any drift in it is caught; the artifact remains the authority for what
  the panel drew.
* **No committed script regenerates the two ROS schedule JSONs.** `ros_pinning_periodic.py` pins
  each net to a single P hart and `ros_pinning_generic.py` numbers its partitions from
  `--control-hart-base` upward, so neither emits the nav-and-control-on-E, detector-on-the-whole-P-
  cluster map those files contain. Their 90-dispatch-per-frame detector graph also has no source
  schedule left in the tree (every committed solve of this workload carries 98). The files are
  therefore inputs, not outputs; `ros_pinning_model.py --verify` treats them as the artifact of
  record and checks that the model reproduces their timing, which it does exactly for the deployed
  one.
* **`warehouse_showdown_story` is assembled from several producers.** Panel I's text lives in
  `results/codesign_feedback/refined_src/showdown_recovered.py`, panels B/C/D come from
  `scripts/hil_envelope_panel.py` and `scripts/hil_story_figure.py`, and the composer named beside
  it on `verify_showdown_figure.REFINED_ALLOWLIST` draws four panels rather than that layout. §4
  reproduces the composer's form, which is the one carrying a sidecar; `warehouse_showdown_story`
  itself stays allowlisted.
* **Panel I's two schedules do not cover the same horizon** (§5). Compare deadline outcomes, not
  makespans.
* **Middleware is free in this model** (§1, assumption 5) and single-threaded execution is assumed
  (assumption 6). The first direction is bounded by `scripts/ros2_middleware_tax_k1.py`; the second
  is not, and is the reason the measured arms of
  [`ros_arms_catalog.md`](../Baselines/ros_arms_catalog.md) exist. Any claim that needs a ROS 2 number to be
  *tight* rather than *indicative* should cite a measured arm, not this page.
* **The flights beside the schedule panel are simulator rollouts.** `complete_figdata` and
  `crash_figdata` are Isaac Lab runs replaying a commanded rate; the model supplies the rate, not
  the flight.
