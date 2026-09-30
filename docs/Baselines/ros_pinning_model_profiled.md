# The same analytical pinning model, costed from the board's own ROS 2 runs

[`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md) re-derives the Tier A
showdown's ROS 2 baseline from its stated inputs: the per-node-pinning **policy model**, evaluated on the
per-node costs it was given. Those costs did not come from ROS. They are XPU-RT's measured
per-dispatch durations, summed per network and divided by a standalone shard-speedup table
(`scripts/ros_pinning_generic.py:63`). The baseline then pays the same per-op compute as XPU-RT, so
the comparison isolates placement policy. This page separates the two contributions: **how much of
that baseline is the model, and how much is the input?** (Tier A and Tier B in
[`ros_baseline_tiers.md`](ros_baseline_tiers.md).)

This document changes exactly one thing and answers it. Same model, same recurrence, same partitions,
same periodic release; per-node costs read from the K1's own ROS 2 C++ runs. Every prediction below
has a board measurement standing next to it, because the arrangement the model describes is one the
board actually ran.

```bash
scripts/ros_pinning_profiled.py                       # all three sections below
scripts/ros_pinning_profiled.py --costs               # the profiled inputs
scripts/ros_pinning_profiled.py --table               # predicted vs measured
scripts/ros_pinning_profiled.py --submitted-spec      # the Tier A showdown's own metric, both ways
scripts/ros_pinning_profiled.py --json out.json
```

No hardware, no GPU, no render. It reads `results/codesign_feedback/ros_traced/` and two schedule
JSONs already in the tree.

---

## 1. Where the profiled inputs come from

`results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp` is the ROS 2 Jazzy C++ harness
every `ros_traced/` run was produced by. It brackets each node's kernel call with `rdtime()` and
nothing else — not the publish, not the DDS take:

| node | bracketed at | what is inside the bracket |
|---|---|---|
| perception | `ros_mb_chain_traced.cpp:288` | `run_model_yolov8_nano_64x96(...)` |
| nav | `ros_mb_chain_traced.cpp:323` | `run_model_fused_full(...)` |
| control | `ros_mb_chain_traced.cpp:346` (chained), `:357` (timer) | `run_model_mlp_control(...)` |

Each bracket is written to that run's `trace.csv` as one `op=node_callback` row with
`actual_start_cycles` / `actual_end_cycles`, in rdtime ticks at 24 MHz (`:67`). That is precisely the
quantity the pinning model calls *a node's cost*, which is what makes the substitution like-for-like.

**The camera is the one input that is not traced.** `:424` stamps the camera timer's `rdtime` into
`released.csv` and publishes, but never brackets the callback, so the camera's cost has to be derived
— see §3.

`scripts/ros_pinning_profiled.py:run_costs` reads one run; `profile()` pools the per-run medians
across runs, split by the perception node's **pool width** (a node's cost is a property of the kernel
and the harts it was given, not of the camera rate — the min/max spread below is the evidence for
pooling across rates). Median, not mean: a callback distribution with a rare 35 ms tail does not have
a mean that is its cost.

### The table of inputs

```
node          pool  runs   med ms      min      max      p95 | submitted  /prof |   recost  /prof
camera           -    27    1.028    0.808    1.232    1.232 |    0.000  0.00x |    0.000  0.00x
perception       -    75   47.737   47.001   48.580   48.171 |        -      - |        -      -
perception       4   124   24.547   24.337   25.253   25.049 |   24.353  0.99x |   30.705  1.25x
perception       8     3   25.098   24.987   25.171   25.705 |        -      - |        -      -
nav              -   202    4.143    3.611    5.727    4.234 |    3.622  0.87x |    4.905  1.18x
control          -   202    0.061    0.045    0.191    0.065 |    0.083  1.36x |    0.081  1.33x
```

`runs` is the number of board runs each figure is pooled over, not the number of callbacks; the runs
themselves are listed per input in the `--json` output. The pool-4 perception figure is pooled over
**124 runs, 11 arms and 13 camera rates (5–120 Hz)** and its per-run medians stay inside
24.337–25.253 ms (−0.9 %/+2.9 % of the pooled median); the serial figure over 75 runs and 4 arms stays
inside 47.001–48.580 ms. That spread is the confidence statement: the node cost is a constant of the
kernel and its partition, not of the rate or the arm, which is what makes pooling legitimate.

The two assumed columns are both real and both in the paper's lineage, so both are carried:

| set | schedule it is read from | what uses it |
|---|---|---|
| `submitted` | `schedules/scheduled_ros_partition_deployed.json` | panel I of the Tier A showdown |
| `recost` | `schedules/scheduled_ros_partition_deployed_matched_board.json` | the 49.76 ms in `warehouse_showdown_board_metrics.json` |

### The shard-speedup assumption, separately

Both assumed sets divide a serial cost by the width-4 credit in `ros_pinning_generic.py:63`, which is
**1.068×** — measured standalone with OC-sharding at codegen, where four harts are slower than two.
The board's own ROS pool does not use that mechanism, and it buys **1.945×** (perception serial
47.737 ms → pool-4 24.547 ms, both measured here, 75 and 124 runs).

So the `recost` input carries two errors in opposite directions that partly cancel:

* the width-1 number the credit divides is **1.46× optimistic** for the baseline,
* the credit itself is **1.82× pessimistic**,
* net at width 4: **1.25× pessimistic** (30.705 against 24.547).

---

## 2. The model, unchanged

`scripts/ros_pinning_profiled.py:predict`, the same recurrence as `ros_pinning_generic.py:84-96`:

```
release_k = k * T
start_k   = max(arrival_k, partition_free)
finish_k  = start_k + C
partition_free = finish_k
```

One node per network, a static partition each, the graph serial inside the node, nothing charged for
DDS, for executor wake-up or for a queue.

The board stamps `camera→goal` at the **control node's callback entry** (`ros_mb_chain_traced.cpp:344`),
before `mlp_control` runs, so the modelled chain is `C(perception) + C(nav)` and the control cost is
deliberately not in it.

Two terms the model does not have are computed **separately** and never folded into the model column:

* **`queue`** — when the camera timer and the perception subscription share one executor thread and
  that process is saturated, the thread fires the timer once per pass, so exactly one unread frame
  stands in front of perception at all times. Gated on `qos_depth >= 2`: a depth-1 history overwrites
  that frame instead of queueing it. The `p3_q1` arm is the falsification test and it is in the table.
* **`dds`** — the three topic hops (`frame`, `detections`, `goal`; the first intra-process in these
  arms, the other two between processes). Not predicted at all; reported as the *residual* of
  measured minus (model + queue), which is the honest way to state the size of what the model omits.

`drift` is the model run out with no drop rule: what `max(k*T, free)` alone says the response reaches
after a 17 s warm run.

---

## 3. The camera cost, derived

Above saturation the camera timer no longer paces its process: it shares the executor thread with the
perception subscription, so it can fire only once per pass of that thread, and `released.csv`'s
inter-release gap *is* the process's service period. Subtracting the perception callback from it
leaves the camera callback plus the executor's own per-pass overhead
(`scripts/ros_pinning_profiled.py:camera_cost`, 27 saturated runs):

```
1.028 ms   [0.808, 1.232]
```

This is a residual, so it is an **upper** bound on the camera's compute and is used as one. The
assumed sets have no camera node at all.

---

## 4. Predicted vs measured

One row per (arm, camera rate) over every pinned arm the board ran. `subm` / `recost` are the model
on the two assumed input sets; `prof` is the same model on the profiled inputs; `+queue` adds the
one named term of §2; `meas` is the median of that cell's board runs (`summary.json:e2e_goal_med_ms`);
`resid` is `meas − (+queue)`. The Hz block is the control cadence: `sHz`/`rHz`/`pHz` predicted on the
three input sets, `mHz` measured (`1000 / gap_mean_ms`).

```
arm         hz  qos    ctrl   sat |   subm recost   prof +queue   meas  resid |   sHz   rHz   pHz   mHz |    drift
cp3         15   10 chained    no |  27.98  35.61  28.69  28.69  30.65  +1.96 |  15.0  15.0  15.0  15.0 |        0
cp3         25   10 chained    no |  27.98  35.61  28.69  28.69  30.62  +1.93 |  25.0  25.0  25.0  25.0 |        0
cp3         30   10 chained    no |  27.98  35.61  28.69  28.69  30.67  +1.98 |  30.0  30.0  30.0  29.9 |        0
cp3         45   10 chained   yes |  27.98  35.61  28.69  54.27  56.15  +1.88 |  41.1  32.6  39.1  38.7 |     2565
cp3         60   10 chained   yes |  27.98  35.61  28.69  54.27  56.18  +1.91 |  41.1  32.6  39.1  39.0 |     9078
cp3         75   10 chained   yes |  27.98  35.61  28.69  54.27  56.06  +1.80 |  41.1  32.6  39.1  39.1 |    15609
cp3         90   10 chained   yes |  27.98  35.61  28.69  54.27  56.02  +1.75 |  41.1  32.6  39.1  39.1 |    22131
cp3n4       30   10 chained    no |  27.98  35.61  28.69  28.69  30.13  +1.44 |  30.0  30.0  30.0  29.9 |        0
cp3n4_d     30   10 chained    no |  27.98  35.61  28.69  28.69  30.05  +1.36 |  30.0  30.0  30.0  29.9 |        0
p3           5   10   timer    no |  27.98  35.61  28.69  28.69  30.44  +1.75 | 100.0 100.0 100.0 100.0 |        0
p3           8   10   timer    no |  27.98  35.61  28.69  28.69  30.46  +1.77 | 100.0 100.0 100.0 100.0 |        0
p3          10   10   timer    no |  27.98  35.61  28.69  28.69  30.53  +1.84 | 100.0 100.0 100.0 100.0 |        0
p3          12   10   timer    no |  27.98  35.61  28.69  28.69  30.51  +1.82 | 100.0 100.0 100.0 100.0 |        0
p3          15   10   timer    no |  27.98  35.61  28.69  28.69  30.51  +1.82 | 100.0 100.0 100.0 100.0 |        0
p3          20   10   timer    no |  27.98  35.61  28.69  28.69  30.48  +1.79 | 100.0 100.0 100.0 100.0 |        0
p3          25   10   timer    no |  27.98  35.61  28.69  28.69  30.47  +1.78 | 100.0 100.0 100.0 100.0 |        0
p3          30   10   timer    no |  27.98  35.61  28.69  28.69  30.40  +1.71 | 100.0 100.0 100.0 100.0 |        0
p3          45   10   timer   yes |  27.98  35.61  28.69  54.27  56.39  +2.12 | 100.0 100.0 100.0 100.0 |     2565
p3          60   10   timer   yes |  27.98  35.61  28.69  54.27  55.85  +1.58 | 100.0 100.0 100.0 100.0 |     9078
p3          75   10   timer   yes |  27.98  35.61  28.69  54.27  56.08  +1.81 | 100.0 100.0 100.0 100.0 |    15609
p3          90   10   timer   yes |  27.98  35.61  28.69  54.27  56.19  +1.93 | 100.0 100.0 100.0 100.0 |    22131
p3         120   10   timer   yes |  27.98  35.61  28.69  54.27  56.39  +2.12 | 100.0 100.0 100.0 100.0 |    35157
p3_c50      45   10   timer   yes |  27.98  35.61  28.69  54.27  56.45  +2.18 |  50.0  50.0  50.0  50.0 |     2565
p3_q1       25    1   timer    no |  27.98  35.61  28.69  28.69  30.70  +2.01 | 100.0 100.0 100.0 100.0 |        0
p3_q1       45    1   timer   yes |  27.98  35.61  28.69  28.69  31.10  +2.41 | 100.0 100.0 100.0 100.0 |     2565
p8           5   10   timer    no |  27.98  35.61  28.69  28.69  30.43  +1.74 | 100.0 100.0 100.0 100.0 |        0
p8           8   10   timer    no |  27.98  35.61  28.69  28.69  30.41  +1.72 | 100.0 100.0 100.0 100.0 |        0
p8          10   10   timer    no |  27.98  35.61  28.69  28.69  30.38  +1.69 | 100.0 100.0 100.0 100.0 |        0
p8          12   10   timer    no |  27.98  35.61  28.69  28.69  30.38  +1.69 | 100.0 100.0 100.0 100.0 |        0
p8          15   10   timer    no |  27.98  35.61  28.69  28.69  30.39  +1.70 | 100.0 100.0 100.0 100.0 |        0
p8          20   10   timer    no |  27.98  35.61  28.69  28.69  30.32  +1.63 | 100.0 100.0 100.0 100.0 |        0
p8          25   10   timer    no |  27.98  35.61  28.69  28.69  30.35  +1.66 | 100.0 100.0 100.0 100.0 |        0
p8          30   10   timer    no |  27.98  35.61  28.69  28.69  30.27  +1.58 | 100.0 100.0 100.0 100.0 |        0
p8          45   10   timer   yes |  27.98  35.61  28.69  54.27  56.22  +1.95 | 100.0 100.0 100.0 100.0 |     2565
part8       45   10   timer   yes |  27.98  35.61  28.69  54.27  56.20  +1.93 | 100.0 100.0 100.0 100.0 |     2565

  residual over 35 rows: median +1.80 ms, min +1.36, max +2.41
```

The measured column is the roll-up in each run's own `summary.json`, the same artifact
`measured_timing.ROS_SENSITIVITY` and `docs/Baselines/ros_arms_catalog.md` read; the catalog's `cp3` rows
(30.6 / 30.6 / 30.7 / 56.2 / 56.2 / 56.1 / 56.0 ms and 15.0 / 25.0 / 29.9 / 38.8 / 39.0 / 39.1 /
39.1 Hz) are these numbers rounded.

### What the table says

**The residual is flat.** Measured − (model + queue) is `+1.80 ms` median with a full range of
`[+1.36, +2.41]` across 35 rows, eight arms, camera rates 5–120 Hz, both control modes and both QoS
depths. The pinning model's structural omission on this chain is a **constant ~1.8 ms**. That covers
the three topic hops the chain crosses — `frame` inside the perception process, `detections` and
`goal` between processes — plus the executor wake-ups on each, about 0.6 ms a hop. That it does not
grow with rate or with load is the interesting part: this chain's middleware cost is per-message, not
congestion-dependent. (It is not comparable to `results/codesign_feedback/ros_middleware_tax/`, which
is 17.6 ms for three hops — that harness is `rclpy`, and these arms are `rclcpp`.)

**The queue term is the whole saturated gap, and it is a QoS property.** Above the saturation rate the
model alone under-predicts by 27.5 ms; one service period of queueing closes it to 1.8 ms. `p3_q1`
falsifies the alternative reading: at 45 Hz with `qos_depth = 1` the same arm measures **31.10 ms**,
not 56, because a depth-1 history overwrites the queued frame instead of holding it. The term is
therefore the keep-last queue, not executor ordering, and it is switched by a QoS setting the model
has no knowledge of.

**Cadence is where profiled inputs win outright.** On the chained arm at saturation the profiled model
predicts **39.1 Hz** against a measured 38.7–39.1 Hz; the `recost` inputs predict **32.6 Hz** (16 %
low) and the `submitted` inputs **41.1 Hz** (6 % high). The profiled prediction is inside the
replicate spread. It is only right because the camera's 1.028 ms is in the service period —
`1000 / (24.547 + 1.028) = 39.1`; dropping the camera gives 40.7 Hz and misses.

**Where the model does not converge at all.** `drift` is the recurrence's own answer with no drop rule:
at 45 Hz it reaches 2565 ms over a 17 s run and at 120 Hz, 35 s. The board is flat at 56 ms at every
one of those rates, because the camera self-clocks and the keep-last queue drops. The Tier A
showdown's single response number is therefore **horizon-dependent** — it is finite only because it was
evaluated over five instances.

---

## 5. The Tier A showdown's own metric, both ways

`warehouse_showdown_board_metrics.json` reports one perception frame, release to output, instance 2 of
5, at a 22 ms period against a 23 ms budget. `--submitted-spec` runs that same recurrence on each
input set (no recost pass, so it will not reproduce the schedule JSON to the digit — that file was
board-recost *after* placement, which moves start times relative to durations):

```
  instance                             0        1        2        3        4
  submitted 24.353 ms/frame        24.35    26.71    29.06    31.41    33.77
  recost    30.705 ms/frame        30.70    39.41    48.11    56.82    65.52
  profiled 24.547 ms/frame         24.55    27.09    29.64    32.19    34.74
```

The sidecar's headline is **49.76 ms** (this recurrence puts the `recost` input at 48.11). The same
metric on the profiled input is **29.64 ms**. Against the XPU-RT arm's 22.999 ms in the same sidecar:

| input | instance-2 response | meets the 23 ms budget? | ratio to XPU-RT |
|---|---|---|---|
| `recost` (the sidecar's figure) | 49.76 ms | no | 2.16× |
| profiled | 29.64 ms | no | **1.29×** |

Under either input the pinned baseline misses the budget on every frame and XPU-RT meets it; the
ratio to XPU-RT is 2.16× on the `recost` input and 1.29× on the profiled one.

---

## 6. Model versus assumption

Reading the three columns against each other:

* **The policy model is sound on the unsaturated chain.** Given the right costs it predicts
  camera→goal to within a constant 1.8 ms at every camera rate from 5 to 120 Hz, and control cadence
  to within 0.4 Hz. Its error there is entirely the middleware it says up front it does not charge
  for.
* **The model is structurally incomplete above saturation.** Two mechanisms carry the whole gap and
  neither is in it: a keep-last queue (one service period, switched off by `qos_depth = 1`) and a
  self-clocked camera (which is what makes the board flat where the recurrence diverges). These are
  not input errors. No cost table fixes them.
* **The Tier A showdown's per-node perception cost is within 0.8 % of the board's.**
  `scheduled_ros_partition_deployed.json` costs perception at 24.353 ms; the board's ROS pool
  measures 24.547 ms. Over the 22 unsaturated rows the Tier A inputs give a mean chain residual
  of −2.46 ms against the board, the profiled inputs −1.74 ms and the `recost` inputs +5.18 ms.
* **The board-recost variant carries the largest cost difference.** `..._matched_board.json` costs the
  same node at 30.705 ms (+25 %), nav at 4.905 ms (+18 %), and that is the input behind the 49.76 ms
  headline. On the same recurrence that one number moves the instance-2 response from 29.64 to
  48.11 ms — 18.5 of the 20.1 ms between the profiled answer and the reported 49.76 ms, with the
  remaining 1.7 ms being the recost pass itself.
* **The camera was assumed away.** Neither assumed set has a camera node. It costs 1.028 ms, it sits
  on the perception partition, and it is the difference between predicting the saturated control
  cadence at 40.7 Hz and at 39.1 Hz.
* **The shard credit is the largest single difference between assumed and measured inputs.** 1.068×
  assumed against 1.945× measured. The assumed table is a real measurement of a different mechanism — OC-sharding at
  codegen, where four harts are slower than two — and the deployed ROS node does not use it; it uses
  a 4-hart worker pool, which nearly doubles. Its effect on the `recost` input is largely cancelled
  by the width-1 number it divides, which is 1.46× optimistic, so the net departure at width 4 is 1.25×.
  That cancellation is arithmetic, not design: either input alone departs from the board by a wider
  margin than the pair does.

Neither implementation should stay separate. `scripts/ros_pinning_model.py` (assumed inputs) and
`scripts/ros_pinning_profiled.py` (profiled inputs) were written against the same recurrence in the
same week without sharing code; the recurrence should live in one of them and the other should import
it, so that a change to the model cannot move one arm of this comparison without moving the other.

---

## 7. Why no new figure

The numbers do not support redrawing the showdown. The profiled baseline is *slower* than the
Tier A one on the deployed arrangement (56.2 ms measured against the Tier A model's 28.0 ms) and
*faster* on the five-instance response metric (29.64 against 49.76 ms), so there is no single
direction to redraw in; and the board-measured showdown figures
([`showdown_r30_reproduction.md`](../Evaluation/showdown_r30_reproduction.md),
[`showdown_submitted_config_reproduction.md`](../Evaluation/showdown_cam45_ros_out_of_box_reproduction.md)) already
present those same arms with every number measured rather than modelled, which is strictly better
evidence than a re-costed model would be. The contribution here is the predicted-vs-measured table,
and it belongs in a table.
