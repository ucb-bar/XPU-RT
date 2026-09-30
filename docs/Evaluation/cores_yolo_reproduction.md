# The cores × YOLO-service figure, and what each of its points is

The paper's cores × YOLO figure (`fig_cores_yolo`). This page says how it is produced and which schedule every drawn point comes from; a point with no
producing schedule is marked as such. The ROS arm is Tier A ([`ros_baseline_tiers.md`](../Baselines/ros_baseline_tiers.md)).

## 1. What the figure plots

One curve per scheme, against the number of harts allocated to YOLO, in two cost models: the
schedule as solved ahead of time (dashed) and the same schedule re-costed on K1-calibrated
per-dispatch costs (solid). A right-hand axis converts service into the fastest cruise it sustains,
against a chosen anchor.

The statistic is the **median per-frame YOLO response**: per YOLO instance, the last dispatch to
finish minus the first to start, then the median over the instances in the schedule. One definition
for all three schemes, so a sharded inference and a sequential one are the same measurement.

## 2. Producing it

```bash
# the published form, at the anchor the published figure used
scripts/cores_yolo_service.py --ros published --anchor 24.0
# the same, with the ROS arm drawn from its one schedule pair
scripts/cores_yolo_service.py --ros flat --anchor 24.0
# the same at the 22 ms frame the paper's prose names
scripts/cores_yolo_service.py --ros flat --anchor 22.0 \
    --out results/codesign_feedback/refined/cores_yolo_service_derived_flat_a22
# the counterpart of refined/cores_yolo_service.png, whose anchor is 24.5
scripts/cores_yolo_service.py --ros published --anchor 24.5 \
    --out results/codesign_feedback/refined/cores_yolo_service_derived_published_a24p5
```

Each writes `<out>.png`, `<out>.pdf` and `<out>_metrics.json`. The sidecar carries, per point, the
schedule it came from with that file's sha256, the kind of number it is, and — for CP-SAT — the
deadline the solve was given. `scripts/verify_cores_yolo.py` re-derives every sourced point from
those schedules and is in `artifact/verify_no_hardware.sh`.

The cruise labels at 8 cores reproduce both published forms exactly: **1.04× / 0.90× / 0.79×** at
the 24 ms anchor (the paper's copy) and **1.07× / 0.91× / 0.80×** at 24.5 (`refined/cores_yolo_service.png`).

## 3. Where each point comes from

| series | schedules | kind |
|---|---|---|
| `greedy_sched` | `scheduled_m_greedy_shard_K{4..8}_predicted_greedy_profiled.json` | achieved service |
| `greedy_board` | `scheduled_m_greedy_shard_K{4..8}_board_greedy_profiled.json` | achieved service |
| `cpsat_sched` | `scheduled_fine_K{4,6,8}_D{19.0,18.5,18.25}_cpsat_profiled.json` | tightest deadline attempted and accepted; K5 and K7 interpolated, never solved |
| `cpsat_board` | `scheduled_cbp_K{4..8}_board_D{24p5,24,23p25,23p25,23}_cpsat_profiled.json` | tightest deadline attempted and accepted |
| `ros_sched` | `scheduled_ros_pin_predicted.json` | achieved service of one 1-hart pin |
| `ros_board` | `scheduled_ros_pin_board.json` | achieved service of one 1-hart pin |

Three properties of that table decide how the figure should be read.

**The CP-SAT rows come from a deadline search, and "tightest" means tightest *attempted*.** The
whole search is tracked, so the claim is checkable rather than asserted:

| width | board deadlines attempted | AOT deadlines attempted |
|---|---|---|
| K4 | 24.5, 25.0, 27.0 | 19.0 |
| K5 | 24.0, 24.5, 25.0 | — (interpolated) |
| K6 | 23.25, 23.5, 23.75, 25.0 | 18.5, 18.75, 19.0 |
| K7 | 23.25 | — (interpolated) |
| K8 | 23.0 | 18.25, 18.5, 18.75, 19.0 |

The plotted point is the tightest row of each column, which `verify_cores_yolo.py` asserts. At
board K7, board K8 and AOT K4 only one deadline was ever attempted, so **nothing here says a
tighter one would have failed** — the verifier reports those three rather than implying otherwise.

The achieved median equals the budget to within 0.02 ms at every point, because CP-SAT schedules to
its deadline; the verifier asserts that coincidence too. So the level of the CP-SAT curve is set by
where the search stopped, not by how fast the solver can make the chain run. Four of the five
`cbp` solves also report a nonzero `deadline_miss_count` (max lateness 0.4 µs, numerical); the
sidecar records it per point.

**The ROS arm has one schedule, not a sweep.** `scheduled_ros_pin_{predicted,board}.json` is a
single 1-hart pin, which is what the caption means by core-independent: the node runs the inference
sequentially, so more harts do not help. `--ros flat` draws that one value at every width.
`--ros published` reproduces the published curve, whose four interior points (K4–K7 on both rows)
no schedule in this repository produces; the sidecar marks each one `as published; no schedule in
this repository produces it`, and `unsourced_points` counts them — 10 in the published mode
(8 ROS + the 2 CP-SAT interpolations), 2 in the flat mode.

A separate no-shard sweep does exist, `scheduled_m_greedy_noshard_K{4..8}_*`, giving 34.6–41.1 ms
AOT and 43.7–55.1 ms board. It is a different configuration — a pool the scheduler may not shard
across, not a 1-hart pin — and it is slower than either drawn ROS curve; it is not drawn in its
place.

**The 1.0× anchor is a choice.** Which curves fall inside the band follows from it, so
`cores_yolo_service.py` requires it as a flag, prints it on the figure and records it in the
sidecar with a note saying it is chosen. Three values are in circulation: 24.0 in the script that
drew the paper's copy (whose comment reads "the aspirational 22 ms isn't reachable on real
silicon"), 24.5 in `refined/cores_yolo_service.png`, and the 22 ms the paper's prose calls the YOLO
frame. At 22 ms no scheme is inside the band; at 24.0 only CP-SAT is; at 24.5 CP-SAT reads 1.07×.

## 4. The producers of the earlier renders

The paper's `plots/fig_cores_yolo.png` (sha256 `e0ee34e3…`) is the output of `fig_fair_v6.py`, which
lived untracked in a second checkout at `XPU-RT/results/codesign_feedback/refined_src/`. It is
preserved verbatim at `scripts/attic/fig_fair_v6.py` so the paper's figure has a recorded producer;
it holds all thirty points as literals and no build step runs it.

The producer of `refined/cores_yolo_service.png` — `fig_measured_first_v5.py`, with
`cpsat_board_percore_all.csv` and an `enrich_methods/ros_percore` table — was never persisted out of
the scratchpad it was written in and is not on disk in either checkout. That render is on
`REFINED_ALLOWLIST` in `verify_showdown_figure.py` for that reason;
`cores_yolo_service_derived_published_a24p5` reproduces its numbers from the schedules.

## 5. Verification

```bash
scripts/verify_cores_yolo.py          # every sourced point against its schedule; 0 failed
bash artifact/verify_no_hardware.sh   # includes the above
```
