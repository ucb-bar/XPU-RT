# Ranking every ROS 2 arm measured on the K1

The showdown figures are only honest if the ROS 2 side of them is the strongest ROS 2 we have
actually measured. This page ranks all of it and says, per figure, whether that holds.

[`ros_arms_catalog.md`](ros_arms_catalog.md) says what each arm *is* (processes, pinning, pool,
executor, QoS, control mode), generated from the run manifests. [`run_index.md`](../Artifact/run_index.md) says
what each run *measured* and which figure draws it. Neither orders them. This one does, and then
asks the adversarial question. (For the same exercise on the XPU-RT side of the comparison, see
`xpurt_arm_ranking.md`.)

**Verdict up front.** At 36 Hz the figures draw the strongest arm we have. At 30 Hz and 45 Hz they
do not: `p3` at 30 Hz and `p3_q1` / `vanilla4x2tm` at 45 Hz beat every drawn arm at their camera
rate on both axes at once, and none of the three is drawn anywhere. §2 states the margins.

---

## 0. Method

**Where the numbers come from.** Every row is re-derived, not copied:

* Latency and cadence: `results/codesign_feedback/ros_traced/summary.csv`, one row per run
  (`<hz>_<arm>_r<k>`), pooled over an arm's replicates at that camera rate the way
  `scripts/measured_timing.py:derive()` pools them — `gap_mean_ms` averaged, `e2e_goal_med_ms`
  median-of-medians, `n_goals`/`n_frames` summed.
* Layout: each run's own `manifest.json`, through `scripts/ros_arms_catalog.py:describe()`.
* Flights: `scripts/flight_quarantine.py:flight_rows` over every `campaign*.csv` under
  `results/codesign_feedback/` (recursive), so faulted PhysX batches are dropped before counting.
  A flight is attributed to an arm through `scripts/figure_constants.py:ARM_BY_TRACE[...].derive`,
  the registry that says which board run a replayed cadence trace came from.

**Columns.**

* **control Hz** = `1000 / gap_mean_ms`, the rate at which the deployment emits a new actuator
  command. This is what a flight replays and what decides the outcome; it is *not* the camera rate
  and *not* the inverse of the latency.
* **camera→goal ms** = `e2e_goal_med_ms`, camera release to the navigation goal reaching control.
* **worst gap ms** = the largest `gap_max_ms` over the arm's replicates: the longest the vehicle
  went without a command.
* **frames dropped** = `1 - Σn_goals / Σn_frames`: frames the camera published that never produced
  a goal.
* **harts busy** = how many of the eight harts the arm *actually* uses, counted as cores whose mean
  `busy_c<k>` over the arm's replicates exceeds 10 %. This is measured occupancy, not the mask the
  launcher handed out: `cp3` reserves six harts and keeps five of them above 10 % because its
  control node costs almost nothing.
* **flights** = completed / flown, after quarantine. "not flown" means no campaign replayed this
  arm's cadence trace; it is not a result.

**Ranking rule.** Within a camera rate, by control cadence descending, then camera→goal ascending.
Cadence first because that is the number the flight campaigns separate on; latency breaks ties.
Where an arm wins one axis and loses the other the table says so rather than hiding it in a scalar
score, and §3.4 gives the flight evidence for how the two axes actually trade.

**Scope.** 194 (arm, camera-rate) cells over 64 arms have board evidence. Cells whose runs recorded
no goals produce no cadence and no latency and are excluded, exactly as the catalog excludes them.
**Ranking across camera rates is not done**: a 30 Hz arm and a 90 Hz arm are carrying different
offered loads, so only within-rate order is meaningful.

---

## 1. The ranking

Arms marked **◀ drawn** are the ones a figure currently replays (from the `ros_trace` field of
`results/codesign_feedback/refined/*_metrics.json`).

### Strongest ROS 2 at each camera rate

| camera | strongest arm | camera→goal | control | harts | why | is it drawn? |
|---|---|---|---|---|---|---|
| 5–20 Hz | `p8` (`p3` within 0.1 ms) | 30.3–30.4 ms | 100 Hz | 4–5 | pinned three-process layout, control on its own timer; nothing at these rates saturates | no |
| 25 Hz | `yproc` | 30.26 ms | 100 Hz | 5 | as `p3` with nav and control sharing one hart; 0 % frames dropped | no |
| 30 Hz | **`p3`** (`p8` 30.27 ms) | 30.40 ms | 100 Hz | 5 | only arms at 30 Hz that hold 100 Hz control at minimum latency | **no** |
| 36 Hz | **`vanilla4x2`** | 32.16 ms | 36.0 Hz | 8 | Pareto-optimal: nothing measured at 36 Hz beats it on either axis | **yes** |
| 38 / 40 / 50 Hz | `vanilla4x2` (only arm measured) | 36.4 / 37.5 / 36.8 ms | 38 / 40 / 50 Hz | 8 | — | no |
| 45 Hz | **`p3_q1`** (`vanilla4x2tm` 37.79 ms / 100 Hz on 8 harts) | 31.10 ms | 100 Hz | 5 | lowest latency *and* highest sustainable cadence at 45 Hz; `p3_c200` reaches 200 Hz at 56.04 ms | **no** |
| 60 Hz | `p3` | 55.85 ms | 100 Hz | 6 | — | no |
| 75 Hz | `p3` | 56.08 ms | 100 Hz | 5 | — | no |
| 90 Hz | `p3` | 56.19 ms | 100 Hz | 5 | — | no |
| 120 Hz | `p3` | 56.39 ms | 100 Hz | 5 | camera-rate-independent: `p3` is flat from 45 to 120 Hz | no |

Two structural facts fall out of the whole table and are worth stating before the rows:

1. **`p3` is flat.** From 45 Hz to 120 Hz its camera→goal stays 55.8–56.4 ms and its cadence stays
   100 Hz. Its pipeline saturates at the 4-hart YOLO pool's throughput (~24.6 ms/frame,
   `YOLO_STANDALONE_MS["4hart_pool"]`) and then simply drops the surplus. No other single-instance
   arm does this without its latency exploding.
2. **The `vanilla*` family's 242 ms is a queue, not compute.** See §3.1; it is nine camera periods
   of keep-last-10 backlog, and three unrelated deployment choices each remove it.

### Per-rate tables

### Camera 5 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.43 | 10.3 | 7% | 1 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.44 | 10.6 | 8% | 4 | 3 | 10 | 1 | not flown |
| 3 | `smte` | 100.0 | 44.94 | 13.5 | 1% | 4 | 1 | 10 | 2 | not flown |
| 4 | `multi` | 100.0 | 53.24 | 10.9 | 1% | 0 | 1 | 10 | 2 | not flown |
| 5 | `spin` | 94.9 | 30.49 | 36.0 | 1% | 2 | 1 | 10 | 2 | not flown |
| 6 | `ship` | 82.4 | 53.79 | 50.3 | 1% | 1 | 1 | 10 | 2 | not flown |

### Camera 8 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.41 | 10.9 | 6% | 4 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.46 | 11.1 | 5% | 4 | 3 | 10 | 1 | not flown |
| 3 | `smte` | 100.0 | 42.74 | 16.3 | 1% | 4 | 1 | 10 | 2 | not flown |
| 4 | `multi` | 100.0 | 53.22 | 11.2 | 1% | 0 | 1 | 10 | 2 | not flown |
| 5 | `spin` | 88.0 | 30.47 | 34.7 | 1% | 4 | 1 | 10 | 2 | not flown |
| 6 | `ship` | 71.1 | 53.43 | 58.7 | 1% | 2 | 1 | 10 | 2 | not flown |

### Camera 10 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.38 | 12.2 | 6% | 4 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.53 | 10.8 | 5% | 4 | 3 | 10 | 1 | not flown |
| 3 | `smte` | 100.0 | 41.71 | 14.1 | 1% | 4 | 1 | 10 | 2 | not flown |
| 4 | `multi` | 100.0 | 53.18 | 11.3 | 1% | 2 | 1 | 10 | 2 | not flown |
| 5 | `spin` | 89.9 | 30.51 | 35.7 | 1% | 4 | 1 | 10 | 2 | not flown |
| 6 | `ship` | 69.2 | 53.38 | 50.2 | 1% | 2 | 1 | 10 | 2 | not flown |

### Camera 12 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.38 | 11.1 | 5% | 4 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.51 | 10.8 | 5% | 4 | 3 | 10 | 1 | not flown |
| 3 | `smte` | 100.0 | 38.63 | 13.6 | 0% | 4 | 1 | 10 | 2 | not flown |
| 4 | `multi` | 100.0 | 53.17 | 13.8 | 0% | 5 | 1 | 10 | 2 | not flown |
| 5 | `spin` | 81.1 | 30.42 | 46.2 | 0% | 4 | 1 | 10 | 2 | not flown |
| 6 | `ship` | 55.5 | 53.34 | 58.6 | 0% | 3 | 1 | 10 | 2 | not flown |

### Camera 15 Hz — 15 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.39 | 10.5 | 5% | 4 | 3 | 10 | 1 | not flown |
| 2 | `yproc` | 100.0 | 30.41 | 15.5 | 0% | 4 | 2 | 10 | 3 | not flown |
| 3 | `p3` | 100.0 | 30.51 | 11.4 | 6% | 4 | 3 | 10 | 3 | not flown |
| 4 | `smte` | 100.0 | 45.36 | 14.7 | 0% | 4 | 1 | 10 | 1 | not flown |
| 5 | `multi` | 100.0 | 53.84 | 12.5 | 0% | 8 | 1 | 10 | 3 | not flown |
| 6 | `spin` | 78.9 | 30.38 | 42.6 | 0% | 4 | 1 | 10 | 3 | not flown |
| 7 | `vanilla4t` | 75.0 | 30.88 | 34.8 | 0% | 4 | 1 | 10 | 3 | not flown |
| 8 | `ship` | 41.6 | 53.88 | 57.1 | 0% | 3 | 1 | 10 | 3 | not flown |
| 9 | `cspin` | 15.0 | 30.48 | 72.9 | 0% | 4 | 1 | 10 | 3 | not flown |
| 10 | `cp3` | 15.0 | 30.65 | 74.8 | 5% | 4 | 3 | 10 | 3 | 0/60 |
| 11 | `vanilla4` | 15.0 | 31.04 | 77.7 | 5% | 5 | 4 | 10 | 3 | not flown |
| 12 | `rvanilla4` | 15.0 | 31.46 | 77.0 | 5% | 7 | 6 | 10 | 3 | not flown |
| 13 | `vanilla` | 15.0 | 53.82 | 76.1 | 5% | 3 | 4 | 10 | 3 | not flown |
| 14 | `cship` | 15.0 | 53.94 | 68.5 | 0% | 3 | 1 | 10 | 3 | not flown |
| 15 | `rvanilla` | 12.0 | 721.20 | 89.4 | 27% | 5 | 6 | 10 | 3 | not flown |

### Camera 20 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.32 | 11.1 | 5% | 4 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.48 | 11.5 | 6% | 5 | 3 | 10 | 1 | not flown |
| 3 | `smte` | 100.0 | 41.80 | 14.6 | 1% | 4 | 1 | 10 | 1 | not flown |
| 4 | `multi` | 100.0 | 53.17 | 13.9 | 1% | 7 | 1 | 10 | 2 | not flown |
| 5 | `spin` | 80.0 | 30.39 | 29.7 | 0% | 4 | 1 | 10 | 2 | not flown |
| 6 | `ship` | 20.1 | 210.51 | 53.3 | 1% | 2 | 1 | 10 | 2 | not flown |

### Camera 25 Hz — 24 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `yproc` | 100.0 | 30.26 | 15.1 | 0% | 5 | 2 | 10 | 3 | not flown |
| 2 | `p8` | 100.0 | 30.35 | 11.9 | 5% | 5 | 3 | 10 | 1 | not flown |
| 3 | `p3` | 100.0 | 30.47 | 11.6 | 6% | 5 | 3 | 10 | 3 | not flown |
| 4 | `p3_q1` | 100.0 | 30.70 | 10.6 | 5% | 5 | 3 | 1 | 1 | not flown |
| 5 | `rp3` | 100.0 | 30.85 | 12.4 | 5% | 7 | 4 | 10 | 2 | not flown |
| 6 | `vanilla4x2tm` | 100.0 | 31.61 | 12.4 | 6% | 7 | 5 | 10 | 3 | not flown |
| 7 | `rmulti` | 100.0 | 33.96 | 13.3 | 0% | 8 | 1 | 10 | 2 | not flown |
| 8 | `multi_q1` | 100.0 | 73.59 | 11.1 | 20% | 8 | 1 | 1 | 1 | not flown |
| 9 | `smte` | 100.0 | 391.12 | 15.2 | 4% | 4 | 1 | 10 | 1 | not flown |
| 10 | `multi` | 100.0 | 433.76 | 14.4 | 21% | 8 | 1 | 10 | 3 | not flown |
| 11 | `vanilla4t` | 74.9 | 30.95 | 31.6 | 0% | 6 | 1 | 10 | 3 | not flown |
| 12 | `spin` | 74.8 | 30.41 | 31.5 | 0% | 4 | 1 | 10 | 3 | not flown |
| 13 | `spin_q1` | 74.8 | 30.43 | 30.4 | 0% | 4 | 1 | 1 | 1 | not flown |
| 14 | `rspin` | 34.9 | 96.29 | 65.9 | 1% | 4 | 1 | 10 | 2 | not flown |
| 15 | `ship_q1` | 27.3 | 107.69 | 54.4 | 11% | 1 | 1 | 1 | 1 | not flown |
| 16 | `cspin` | 25.0 | 30.36 | 90.1 | 0% | 4 | 1 | 10 | 3 | not flown |
| 17 | `cp3` | 25.0 | 30.62 | 60.4 | 5% | 5 | 3 | 10 | 3 | 0/60 |
| 18 | `vanilla4` | 25.0 | 30.99 | 57.3 | 5% | 5 | 4 | 10 | 3 | not flown |
| 19 | `rvanilla4` | 25.0 | 31.34 | 56.9 | 5% | 7 | 6 | 10 | 3 | not flown |
| 20 | `vanilla4x2` | 25.0 | 31.41 | 66.8 | 6% | 8 | 5 | 10 | 3 | not flown |
| 21 | `vanilla` | 20.7 | 433.29 | 53.0 | 23% | 2 | 4 | 10 | 3 | not flown |
| 22 | `ship` | 18.8 | 212.43 | 54.2 | 1% | 3 | 1 | 10 | 3 | not flown |
| 23 | `cship` | 18.7 | 212.63 | 54.0 | 1% | 3 | 1 | 10 | 3 | not flown |
| 24 | `rvanilla` | 11.9 | 470.02 | 92.3 | 56% | 4 | 6 | 10 | 3 | not flown |

### Camera 30 Hz — 20 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p8` | 100.0 | 30.27 | 11.6 | 5% | 4 | 3 | 10 | 1 | not flown |
| 2 | `p3` | 100.0 | 30.40 | 10.8 | 5% | 5 | 3 | 10 | 1 | 3/48 |
| 3 | `vanilla4x2tm` | 100.0 | 31.60 | 12.2 | 5% | 8 | 5 | 10 | 3 | 1/48 |
| 4 | `smte` | 100.0 | 364.89 | 15.6 | 23% | 4 | 1 | 10 | 1 | not flown |
| 5 | `multi` | 100.0 | 369.49 | 12.8 | 33% | 8 | 1 | 10 | 2 | not flown |
| 6 | `spin` | 56.2 | 30.39 | 46.2 | 0% | 4 | 1 | 10 | 2 | not flown |
| 7 | `vanilla4t` | 50.5 | 30.95 | 36.5 | 0% | 5 | 1 | 10 | 3 | not flown |
| 8 | `vanilla4x2c0_ime` | 30.0 | 23.96 | 60.1 | 6% | 4 | 5 | 10 | 3 | not flown |
| 9 | `vanilla4` | 30.0 | 31.09 | 35.8 | 5% | 5 | 4 | 10 | 3 | not flown |
| 10 | `vanilla4x2c0_rvv` | 30.0 | 31.10 | 76.0 | 6% | 4 | 5 | 10 | 3 | not flown |
| 11 | `vanilla4x2` **◀ drawn** | 30.0 | 31.38 | 57.9 | 10% | 8 | 5 | 10 | 3 | 0/228 |
| 12 | `rvanilla4` | 30.0 | 31.52 | 47.9 | 5% | 8 | 6 | 10 | 3 | not flown |
| 13 | `vanilla4x2d2` | 30.0 | 31.63 | 63.5 | 7% | 8 | 5 | 10 | 1 | not flown |
| 14 | `cp3n4_d` | 29.9 | 30.05 | 78.8 | 7% | 5 | 3 | 10 | 3 | not flown |
| 15 | `cp3n4` **◀ drawn** | 29.9 | 30.13 | 80.9 | 7% | 5 | 3 | 10 | 3 | 0/120 |
| 16 | `cp3` **◀ drawn** | 29.9 | 30.67 | 80.6 | 7% | 5 | 3 | 10 | 3 | 0/120 |
| 17 | `vanilla` | 20.7 | 370.55 | 53.0 | 36% | 4 | 4 | 10 | 3 | not flown |
| 18 | `ship` | 19.0 | 210.40 | 53.3 | 1% | 2 | 1 | 10 | 2 | not flown |
| 19 | `vanilla4x2_ime` | 15.0 | 24.07 | 90.7 | 53% | 4 | 4 | 10 | 3 | not flown |
| 20 | `rvanilla` | 11.9 | 406.30 | 89.9 | 63% | 5 | 6 | 10 | 3 | not flown |

### Camera 36 Hz — 6 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `vanilla4x2` **◀ drawn** | 36.0 | 32.16 | 60.3 | 6% | 8 | 5 | 10 | 3 | 2/84 |
| 2 | `vanilla4x2_q1` | 36.0 | 32.34 | 55.6 | 5% | 8 | 5 | 1 | 3 | not flown |
| 3 | `vanilla4x2d2` **◀ drawn** | 36.0 | 32.83 | 52.3 | 7% | 8 | 5 | 10 | 1 | not flown |
| 4 | `vanilla4x2ns4c` **◀ drawn** | 36.0 | 37.76 | 54.0 | 6% | 8 | 5 | 10 | 1 | 0/12 |
| 5 | `vanilla4x2ns4a` | 36.0 | 38.71 | 53.1 | 7% | 8 | 5 | 10 | 1 | not flown |
| 6 | `vanilla4x2ns4b` | 36.0 | 39.17 | 58.1 | 7% | 8 | 5 | 10 | 1 | not flown |

### Camera 38 Hz — 1 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `vanilla4x2` | 38.0 | 36.42 | 57.2 | 5% | 8 | 5 | 10 | 3 | not flown |

### Camera 40 Hz — 1 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `vanilla4x2` | 40.0 | 37.45 | 52.1 | 5% | 8 | 5 | 10 | 3 | 7/60 |

### Camera 45 Hz — 54 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p3_c200` | 200.0 | 56.04 | 7.0 | 5% | 6 | 3 | 10 | 1 | not flown |
| 2 | `multi_c200` | 200.0 | 264.98 | 9.1 | 56% | 8 | 1 | 10 | 1 | not flown |
| 3 | `p3_q1` | 100.0 | 31.10 | 10.4 | 5% | 5 | 3 | 1 | 1 | 81/528 |
| 4 | `vanilla4x2tm` | 100.0 | 37.79 | 14.8 | 5% | 8 | 5 | 10 | 3 | not flown |
| 5 | `yproc` | 100.0 | 55.65 | 16.3 | 0% | 5 | 2 | 10 | 3 | not flown |
| 6 | `p3_hog2` | 100.0 | 55.87 | 11.7 | 8% | 7 | 3 | 10 | 1 | not flown |
| 7 | `part8` | 100.0 | 56.20 | 12.0 | 5% | 5 | 3 | 10 | 6 | not flown |
| 8 | `p8` | 100.0 | 56.22 | 11.5 | 5% | 5 | 3 | 10 | 3 | not flown |
| 9 | `p3` | 100.0 | 56.39 | 14.5 | 5% | 5 | 3 | 10 | 6 | 109/636 |
| 10 | `rp8` | 100.0 | 56.94 | 15.4 | 5% | 8 | 5 | 10 | 3 | not flown |
| 11 | `rp3` | 100.0 | 57.36 | 11.7 | 5% | 7 | 4 | 10 | 2 | not flown |
| 12 | `x2p` | 100.0 | 62.73 | 14.6 | 15% | 8 | 4 | 10 | 2 | not flown |
| 13 | `x2rp3` | 100.0 | 63.28 | 14.5 | 16% | 8 | 5 | 10 | 2 | not flown |
| 14 | `multi_q1` | 100.0 | 64.92 | 12.7 | 55% | 7 | 1 | 1 | 1 | not flown |
| 15 | `rvanilla4x2tm` | 100.0 | 89.70 | 15.7 | 5% | 8 | 7 | 10 | 3 | not flown |
| 16 | `vanilla8tm` | 100.0 | 241.93 | 12.5 | 18% | 5 | 4 | 10 | 3 | 1/12 |
| 17 | `vanilla4tm` | 100.0 | 242.49 | 14.2 | 19% | 5 | 4 | 10 | 3 | 50/552 |
| 18 | `rvanilla8tm` | 100.0 | 244.02 | 15.8 | 22% | 8 | 6 | 10 | 3 | not flown |
| 19 | `x2rmulti` | 100.0 | 247.90 | 14.6 | 67% | 8 | 1 | 10 | 2 | not flown |
| 20 | `smte` | 100.0 | 258.65 | 15.5 | 49% | 4 | 1 | 10 | 3 | not flown |
| 21 | `multi` | 100.0 | 264.68 | 13.3 | 56% | 8 | 1 | 10 | 3 | 2/48 |
| 22 | `multi_hog2` | 100.0 | 265.43 | 13.7 | 57% | 8 | 1 | 10 | 1 | not flown |
| 23 | `rmulti` | 99.9 | 248.01 | 47.0 | 34% | 8 | 1 | 10 | 4 | not flown |
| 24 | `vanilla4x2tm_c50` | 50.0 | 39.80 | 24.7 | 6% | 8 | 5 | 10 | 3 | not flown |
| 25 | `p3_c50` | 50.0 | 56.45 | 22.0 | 5% | 5 | 3 | 10 | 3 | not flown |
| 26 | `vanilla8tm_c50` | 50.0 | 242.88 | 22.0 | 20% | 7 | 4 | 10 | 3 | not flown |
| 27 | `spin_q1` | 46.8 | 60.08 | 39.4 | 18% | 4 | 1 | 1 | 1 | not flown |
| 28 | `vanilla4x2` | 45.0 | 37.04 | 48.5 | 4% | 8 | 5 | 10 | 3 | 35/228 |
| 29 | `vanilla4_q1` | 39.1 | 42.04 | 35.5 | 17% | 5 | 4 | 1 | 3 | 20/156 |
| 30 | `cp3` **◀ drawn** | 38.8 | 56.15 | 42.9 | 6% | 5 | 3 | 10 | 3 | 25/204 |
| 31 | `vanilla8f` | 38.8 | 242.28 | 75.5 | 19% | 6 | 4 | 10 | 3 | not flown |
| 32 | `vanilla4` **◀ drawn** | 38.6 | 242.39 | 70.3 | 19% | 6 | 4 | 10 | 3 | 87/1116 |
| 33 | `vanilla8_q1` | 38.5 | 42.49 | 72.3 | 19% | 6 | 4 | 1 | 3 | not flown |
| 34 | `vanilla8` | 38.5 | 242.47 | 73.1 | 20% | 6 | 4 | 10 | 3 | not flown |
| 35 | `vanilla4f` | 38.3 | 242.45 | 72.7 | 20% | 6 | 4 | 10 | 3 | not flown |
| 36 | `rvanilla4` | 37.8 | 243.39 | 36.1 | 21% | 8 | 6 | 10 | 3 | 9/120 |
| 37 | `rvanilla8` | 37.5 | 243.62 | 76.3 | 21% | 8 | 6 | 10 | 3 | not flown |
| 38 | `spin` | 33.4 | 119.03 | 33.8 | 1% | 4 | 1 | 10 | 3 | 0/60 |
| 39 | `spin_hog2` | 33.2 | 119.48 | 42.1 | 1% | 6 | 1 | 10 | 1 | not flown |
| 40 | `spin_c200` | 33.2 | 119.66 | 39.4 | 1% | 4 | 1 | 10 | 1 | not flown |
| 41 | `vanilla4t` | 32.9 | 121.91 | 40.2 | 1% | 5 | 1 | 10 | 3 | 10/216 |
| 42 | `cspin` | 32.8 | 120.85 | 44.2 | 1% | 4 | 1 | 10 | 3 | not flown |
| 43 | `ship_q1` | 27.9 | 107.19 | 54.1 | 19% | 1 | 1 | 1 | 1 | not flown |
| 44 | `vanilla_q1` | 20.7 | 64.91 | 53.1 | 56% | 1 | 4 | 1 | 1 | not flown |
| 45 | `vanilla` | 20.7 | 264.77 | 52.2 | 57% | 3 | 4 | 10 | 3 | 0/48 |
| 46 | `vanilla_smoke` | 20.7 | 264.99 | 49.6 | 63% | 1 | 4 | 10 | 1 | not flown |
| 47 | `vanilla_c50` **◀ drawn** | 20.3 | 265.90 | 51.8 | 58% | 3 | 4 | 10 | 3 | 0/72 |
| 48 | `rspin` | 18.8 | 198.24 | 81.1 | 1% | 4 | 1 | 10 | 2 | not flown |
| 49 | `ship` | 18.8 | 213.21 | 54.8 | 1% | 2 | 1 | 10 | 3 | not flown |
| 50 | `cship` | 18.7 | 212.69 | 53.9 | 1% | 2 | 1 | 10 | 3 | not flown |
| 51 | `ship_c200` | 18.7 | 213.30 | 54.0 | 1% | 1 | 1 | 10 | 1 | not flown |
| 52 | `x2spin` | 16.7 | 417.72 | 69.3 | 51% | 4 | 1 | 10 | 2 | not flown |
| 53 | `rvanilla` | 11.9 | 300.70 | 90.9 | 75% | 6 | 6 | 10 | 3 | not flown |
| 54 | `x2rspin` | 10.8 | 623.77 | 100.0 | 52% | 4 | 1 | 10 | 2 | not flown |

### Camera 50 Hz — 1 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `vanilla4x2` | 50.0 | 36.78 | 51.1 | 6% | 8 | 5 | 10 | 3 | not flown |

### Camera 60 Hz — 13 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p3` | 100.0 | 55.85 | 10.3 | 5% | 6 | 3 | 10 | 1 | not flown |
| 2 | `smte` | 100.0 | 207.75 | 14.2 | 62% | 4 | 1 | 10 | 1 | not flown |
| 3 | `multi` | 100.0 | 212.48 | 11.2 | 67% | 8 | 1 | 10 | 1 | not flown |
| 4 | `vanilla4x2` | 60.1 | 68.44 | 44.1 | 6% | 8 | 5 | 10 | 3 | not flown |
| 5 | `cp3` | 39.0 | 56.18 | 29.5 | 5% | 5 | 3 | 10 | 1 | not flown |
| 6 | `vanilla4` | 39.0 | 189.39 | 32.7 | 39% | 6 | 4 | 10 | 3 | not flown |
| 7 | `rvanilla4` | 38.0 | 190.84 | 39.2 | 40% | 8 | 6 | 10 | 3 | not flown |
| 8 | `spin` | 33.3 | 119.38 | 34.0 | 1% | 4 | 1 | 10 | 1 | not flown |
| 9 | `vanilla4t` | 32.9 | 121.84 | 34.1 | 1% | 6 | 1 | 10 | 3 | not flown |
| 10 | `cspin` | 32.5 | 121.87 | 36.3 | 1% | 4 | 1 | 10 | 1 | not flown |
| 11 | `vanilla` | 20.7 | 212.05 | 50.3 | 68% | 2 | 4 | 10 | 3 | not flown |
| 12 | `ship` | 18.7 | 213.25 | 58.0 | 1% | 1 | 1 | 10 | 1 | not flown |
| 13 | `rvanilla` | 11.9 | 247.17 | 90.0 | 81% | 6 | 6 | 10 | 3 | not flown |

### Camera 75 Hz — 9 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p3` | 100.0 | 56.08 | 12.2 | 5% | 5 | 3 | 10 | 1 | not flown |
| 2 | `smte` | 100.0 | 174.29 | 14.5 | 69% | 4 | 1 | 10 | 1 | not flown |
| 3 | `multi` | 100.0 | 180.06 | 13.7 | 73% | 8 | 1 | 10 | 1 | not flown |
| 4 | `vanilla4x2tm` | 100.0 | 291.51 | 15.8 | 23% | 8 | 5 | 10 | 3 | not flown |
| 5 | `vanilla4x2` | 61.8 | 291.95 | 43.4 | 23% | 8 | 5 | 10 | 3 | not flown |
| 6 | `cp3` | 39.1 | 56.06 | 29.4 | 5% | 5 | 3 | 10 | 1 | not flown |
| 7 | `spin` | 33.1 | 119.77 | 39.7 | 1% | 4 | 1 | 10 | 1 | not flown |
| 8 | `cspin` | 32.6 | 122.06 | 36.9 | 1% | 4 | 1 | 10 | 1 | not flown |
| 9 | `ship` | 18.7 | 213.04 | 53.8 | 1% | 1 | 1 | 10 | 1 | not flown |

### Camera 90 Hz — 16 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p3` | 100.0 | 56.19 | 12.5 | 5% | 5 | 3 | 10 | 1 | 74/480 |
| 2 | `rp3` | 100.0 | 56.33 | 12.6 | 12% | 6 | 4 | 10 | 2 | not flown |
| 3 | `vanilla4tm` | 100.0 | 136.62 | 12.9 | 59% | 7 | 4 | 10 | 3 | 62/480 |
| 4 | `smte` | 100.0 | 154.77 | 14.3 | 75% | 4 | 1 | 10 | 1 | not flown |
| 5 | `multi` | 100.0 | 159.38 | 12.8 | 78% | 8 | 1 | 10 | 1 | not flown |
| 6 | `vanilla4x2` | 61.7 | 252.76 | 43.4 | 36% | 8 | 5 | 10 | 3 | 12/120 |
| 7 | `vanilla4_q1` | 39.3 | 36.67 | 31.4 | 59% | 7 | 4 | 1 | 3 | not flown |
| 8 | `cp3` | 39.1 | 56.02 | 29.6 | 6% | 5 | 3 | 10 | 1 | not flown |
| 9 | `vanilla4` | 39.0 | 136.86 | 27.8 | 59% | 5 | 4 | 10 | 3 | not flown |
| 10 | `rvanilla4` | 38.0 | 137.49 | 38.4 | 60% | 8 | 6 | 10 | 3 | 0/60 |
| 11 | `spin` | 33.3 | 119.47 | 37.0 | 1% | 4 | 1 | 10 | 1 | not flown |
| 12 | `cspin` | 33.2 | 119.64 | 39.6 | 1% | 4 | 1 | 10 | 1 | not flown |
| 13 | `vanilla4t` | 32.9 | 121.16 | 37.5 | 1% | 6 | 1 | 10 | 3 | not flown |
| 14 | `vanilla` | 20.7 | 159.25 | 50.2 | 78% | 3 | 4 | 10 | 3 | not flown |
| 15 | `ship` | 18.6 | 214.38 | 54.2 | 1% | 1 | 1 | 10 | 1 | not flown |
| 16 | `rvanilla` | 11.9 | 195.03 | 90.9 | 87% | 5 | 6 | 10 | 3 | not flown |

### Camera 120 Hz — 4 arms

| # | arm | control Hz | camera→goal ms | worst gap ms | frames dropped | harts busy | proc | QoS | runs | flights completed/flown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `p3` | 100.0 | 56.39 | 12.4 | 6% | 5 | 3 | 10 | 3 | not flown |
| 2 | `vanilla4x2tm` | 100.0 | 200.35 | 19.0 | 53% | 8 | 5 | 10 | 3 | not flown |
| 3 | `vanilla4x2` | 61.0 | 199.99 | 44.1 | 53% | 8 | 5 | 10 | 3 | not flown |
| 4 | `vanilla4` | 38.3 | 110.61 | 72.3 | 70% | 6 | 4 | 10 | 3 | not flown |


---

## 2. Is the baseline we draw the best one?

**First, the one thing that holds across all of it.** Of the 72 (deployment, camera-rate) pairs whose
control node is `chained`, **none commands faster than its camera** — 31 command at the camera rate
and the remaining 41 slower, because a chain that cannot keep up emits commands more slowly than
frames arrive. The 124 `timer` pairs are not bound that way: they re-send a goal they are still
holding, so their cadence says nothing about how fresh the goal under it is. That census is generated
by `scripts/build_implementation_bundles.py`, which fails if a counterexample ever appears, and is
printed at the top of `artifact/implementations/ros_deployments/README.md`. It is the difference
between an example and a rule, and it is why the comparison below is about *which* baseline rather
than about whether the mechanism is real.

Nine (arm, camera-rate) cells are replayed by a figure. For each, the arms below are those that
were measured at the **same camera rate** and are **at least as good on both axes and strictly
better on one** (tolerance 0.05 ms / 0.05 Hz) — i.e. there is no trade-off to argue about; they are
simply better ROS 2.

### 2.1 `vanilla4` @ 45 Hz — 242.39 ms, 38.6 Hz, 6 harts, 87/1116 flights — drawn in 9 figures

**22 of the 53 other arms measured at 45 Hz dominate it.** The ones that matter:

| arm | camera→goal | vs drawn | control | vs drawn | harts | runs | flights |
|---|---|---|---|---|---|---|---|
| `p3_q1` | 31.10 ms | **7.79× lower** | 100 Hz | **2.59×** | 5 | 1 | 81/528 |
| `vanilla4x2tm` | 37.79 ms | 6.41× lower | 100 Hz | 2.59× | 8 | 3 | not flown |
| `vanilla4x2` | 37.04 ms | 6.54× lower | 45.0 Hz | 1.16× | 8 | 3 | 35/228 |
| `p3` | 56.39 ms | 4.30× lower | 100 Hz | 2.59× | 5 | 6 | 109/636 |
| `yproc` | 55.65 ms | 4.36× lower | 100 Hz | 2.59× | 5 | 3 | not flown |
| `vanilla4_q1` | 42.04 ms | 5.77× lower | 39.1 Hz | 1.01× | 5 | 3 | 20/156 |
| `p3_c200` | 56.04 ms | 4.33× lower | 200 Hz | 5.18× | 6 | 1 | not flown |

The flights agree with the table. Restricted to the campaign cells the two arms actually share
(same course, prop density, cruise speed, moment scale), `p3` completes **16.7 % (58/348)** against
`vanilla4`'s **6.8 % (52/768)** over 17 shared cells; `vanilla4x2` **15.5 % (26/168)** against
**6.7 % (40/600)** over 9; `vanilla4_q1` **12.8 % (20/156)** against **6.9 % (38/552)** over 5.

This is defensible *only* under the label the figures give it — "ROS 2 unpinned, as one writes it".
As the naive arm it is correct and it is the right thing to draw in a figure whose caption says
"default". It is indefensible as *the* ROS 2 number, and nine figures is a lot of figures for one
deliberately naive arm.

### 2.2 `cp3` @ 45 Hz — 56.15 ms, 38.8 Hz, 5 harts, 25/204 flights — drawn in 4 figures

This is the "hand-tuned, statically pinned" arm, so it is the one that has to be the strongest.
**It is not: 9 arms dominate it.**

| arm | camera→goal | vs drawn | control | vs drawn | harts | runs | flights |
|---|---|---|---|---|---|---|---|
| `p3_q1` | 31.10 ms | **1.81× lower** | 100 Hz | **2.58×** | 5 | 1 | 81/528 |
| `vanilla4x2tm` | 37.79 ms | 1.49× lower | 100 Hz | 2.58× | 8 | 3 | not flown |
| `vanilla4x2` | 37.04 ms | 1.52× lower | 45.0 Hz | 1.16× | 8 | 3 | 35/228 |
| `vanilla4_q1` | 42.04 ms | 1.34× lower | 39.1 Hz | 1.01× | 5 | 3 | 20/156 |
| `yproc` | 55.65 ms | 1.01× lower | 100 Hz | 2.58× | 5 | 3 | not flown |
| `part8` | 56.20 ms | = | 100 Hz | 2.58× | 5 | 6 | not flown |
| `p3_c200` | 56.04 ms | = | 200 Hz | 5.15× | 6 | 1 | not flown |

`p3` is `cp3` with one line changed — control on its own 100 Hz timer instead of chained to the
goal (§3.4) — and it is the *same three pinned processes on the same five harts*. It reaches the
same latency (56.39 vs 56.15 ms, +0.4 %) and **2.58× the command rate**. Cell-matched, `p3`
completes **19.6 % (40/204)** against `cp3`'s **12.3 % (25/204)** over the 5 shared cells.

**This is the finding.** `cp3` is drawn as the strong ROS 2 baseline in four showdown figures, and
a strictly better arm — same hardware, same pinning, same process count, one flag — was measured,
flown, and is drawn nowhere.

### 2.3 `vanilla_c50` @ 45 Hz — 265.90 ms, 20.3 Hz, 0/72 flights — drawn in 1 figure

**46 of the 53 dominate it**; it is the weakest measured cell at 45 Hz on both axes at once. It is
drawn only in `warehouse_showdown_paper_submitted`, where it is the measured form of the configuration
the Tier A showdown models ([`ros_baseline_tiers.md`](ros_baseline_tiers.md)). New
figures draw a current baseline instead; `run_index.md` marks it as the out-of-the-box arm.

### 2.4 `vanilla4x2` @ 36 Hz — 32.16 ms, 36.0 Hz, 8 harts, 2/84 flights — drawn in 5 figures

**Nothing measured at 36 Hz dominates it, on either axis.** This one is clean.

The caveat is the *evidence*, not the arm: only six cells exist at 36 Hz, and all six are the same
`vanilla4x2` five-process graph with chained control. **`vanilla4x2tm` — the timer variant, which
holds 100 Hz control at 25, 30, 45, 75 and 120 Hz for at most +0.75 ms of latency (§3.4) — was
never run at 36 Hz.** Its 30 Hz cell is 31.60 ms at 100 Hz and its 45 Hz cell is 37.79 ms at
100 Hz, so a 36 Hz cell would land near 32–34 ms at 100 Hz and would dominate the drawn arm on
cadence by 2.8×. The 36 Hz figures are a control-rate story; they are told against the only arm at
that rate whose control rate is pinned to the camera. **One run
(`RATES=36 scripts/ros_traced_matrix.sh vanilla4x2tm`, replicates 1–3) closes this.** Until it is
run we cannot claim the 36 Hz baseline is the strongest ROS 2 at 36 Hz — only that it is the
strongest of the six arms we happened to measure there.

### 2.5 `vanilla4x2ns4c` @ 36 Hz — 37.76 ms, 36.0 Hz, 8 harts, 0/12 flights — drawn in 4 figures

Dominated by three arms at the same rate: `vanilla4x2` (32.16 ms, **1.17× lower**, same 36.0 Hz),
`vanilla4x2_q1` (32.34 ms), `vanilla4x2d2` (32.83 ms).

This is by construction and the figures say so: `ns4c` adds a four-way worker pool under the
*navigation* net as well as YOLO, and `measured_timing.py` calls it "the most capable ROS 2
arrangement measured here". It is capable in the sense of *most parallelism given to the baseline*,
not in the sense of *fastest*. Those are different claims and a caption that says "ROS 2 with every
stage pooled" is fine, while one that says "the best ROS 2 can do at 36 Hz" is not — `vanilla4x2`
is 5.6 ms faster at the same cadence. Both are drawn, in different figures, which is the right
outcome; the risk is a caption, not the data.

### 2.6 `vanilla4x2d2` @ 36 Hz — 32.83 ms, 36.0 Hz — drawn in 3 (ladder) figures

Dominated by `vanilla4x2` (32.16 ms) and `vanilla4x2_q1` (32.34 ms) at the same cadence, by
0.5–0.7 ms. `d2` is the same arm with the second YOLO pool's per-shard detail traced, which is why
it is drawn in the ladder figures that need those Gantt rows; the instrumentation costs the
0.7 ms and a little tail (52.3 ms worst gap against 60.3 un-instrumented). Not a fairness problem.

### 2.7 `vanilla4x2` @ 30 Hz — 31.38 ms, 30.0 Hz, 8 harts, 0/228 flights — drawn in 8 figures

**7 arms dominate it**, and two of them by a lot on the axis that matters:

| arm | camera→goal | control | harts | note |
|---|---|---|---|---|
| `p3` | 30.40 ms | **100 Hz (3.33×)** | 5 | flown 3/48 |
| `p8` | 30.27 ms | **100 Hz (3.33×)** | 4 | not flown |
| `spin` | 30.39 ms | 56.2 Hz (1.87×) | 4 | one process, 4-hart pool |
| `vanilla4t` | 30.95 ms | 50.5 Hz (1.68×) | 5 | one process, timer |
| `vanilla4x2c0_ime` | **23.96 ms (1.31× lower)** | 30.0 Hz | 4 | IME kernels, cluster 0 only |
| `vanilla4` | 31.09 ms | 30.0 Hz | 5 | one pool instead of two |

`vanilla4x2tm` at 30 Hz (31.60 ms, **100 Hz**, 8 harts, flown 1/48) is not in the dominance list
only because it is 0.22 ms slower; on cadence it is 3.33× the drawn arm.

`vanilla4x2c0_ime` deserves its own sentence: **23.96 ms is the lowest camera→goal any ROS 2 arm
has produced on this board**, lower than the XPU-RT arm the 30 Hz figures draw against it
(`p30freer`, 26.75 ms). It is ROS 2 running the IME kernels, confined to cluster 0 because
`smt.vmadot` SIGILLs on cluster 1 — the trade [`ros_with_ime.md`](ros_with_ime.md) exists to
document. Its cadence is 30 Hz, so the comparison is not lost; but any caption claiming XPU-RT wins
the 30 Hz *latency* is false against an arm we measured.

### 2.8 `cp3` @ 30 Hz — 30.67 ms, 29.9 Hz, 0/120 — and `cp3n4` @ 30 Hz — 30.13 ms, 29.9 Hz, 0/120

`cp3` is dominated by 6 arms, again including `p3` and `p8` at **3.34× the command rate** for
0.3 ms less latency. `cp3n4` is dominated only by `cp3n4_d` (0.08 ms) and `vanilla4x2c0_ime`, so on
latency it is close to the best there is — but `p3` still commands 3.34× as often at +0.27 ms.
At 30 Hz the same one-line change (§3.4) is worth 3.3× the cadence, and neither figure takes it.

### 2.9 Summary of the audit

| drawn cell | strongest at that rate | drawn arm is… |
|---|---|---|
| `vanilla4` @45 | `p3_q1` (31.10 ms / 100 Hz) | dominated by 22 arms — correct only as the *naive* arm |
| `cp3` @45 | `p3_q1`; `p3` is `cp3`+timer | **dominated by 9 arms; the "tuned" baseline is not the tuned optimum** |
| `vanilla_c50` @45 | — | dominated by 46; out-of-the-box figure only |
| `vanilla4x2` @36 | itself | **the strongest measured — but no timer arm exists at 36 Hz** |
| `vanilla4x2ns4c` @36 | `vanilla4x2` | dominated by 3; deliberate (nav pooled), caption-sensitive |
| `vanilla4x2d2` @36 | `vanilla4x2` | dominated by 2 by 0.7 ms; instrumentation cost, fine |
| `vanilla4x2` @30 | `p3` / `p8` (100 Hz); `vanilla4x2c0_ime` on latency | **dominated by 7** |
| `cp3` @30 | `p3` / `p8` | **dominated by 6** |
| `cp3n4` @30 | `cp3n4_d`; `p3` on cadence | dominated by 2 |

**What to do about it.** Either draw `p3` (30 Hz and 45 Hz) and `vanilla4x2tm` (45 Hz, and at 36 Hz
once run) as the strong ROS 2 arm, or keep the current arms and let every caption say exactly what
the arm is — "unpinned default", "statically pinned with control chained to the goal" — and state
in the text that a measured ROS 2 arrangement reaches 100 Hz control at the same latency. What is
not defensible is a caption of the form "ROS 2, hand-tuned" over `cp3`.

---

## 3. What each deployment choice buys

Every row below is a pair of arms at the same camera rate whose manifests differ in one thing.
Latency is `e2e_goal_med_ms`, cadence is `1000/gap_mean_ms`, both pooled over the arms' replicates.

### 3.1 QoS depth: the 200 ms that is not compute

`p3` and `p3_q1` differ in **exactly one manifest field**. Diffing
`ros_traced/45_p3_r1/manifest.json` against `ros_traced/45_p3_q1_r1/manifest.json`: same three
processes, same `nodes`, same `affinity_mask` (`0x1`/`0x10`/`0x20`), same `yolo_pool: 4` on harts
`0,1,2,3`, same `executor: single`, same `ctrl_mode: timer` at `ctrl_hz: 100`, same
`rmw_fastrtps_cpp`, same `ir=0c783539626c`. The only difference is `qos_depth: 10` → `qos_depth: 1`.
And the measurement is **56.39 ms → 31.10 ms** (`p3` is the median over six replicates:
56.41/56.41/56.37/56.41/56.34/56.36; `p3_q1` is one run, 31.10).

*(The `kernels_sha` field differs — `871ec7e9`/`a87bda63`/`d9ce8dcd` vs `6fbda91` — but the IR hash
is identical and `p3` measures 30.44–30.54 ms at 25 Hz across three different shas, so the build is
not what moved.)*

**Why.** At a 45 Hz camera the period is 22.22 ms and the 4-hart YOLO pool takes ~24.58 ms per
frame (`YOLO_STANDALONE_MS["4hart_pool"]`). Service time exceeds the period, so one frame is always
waiting behind the one in flight. Keep-last-10 lets it wait; keep-last-1 throws it away and the
chain measures its own unqueued latency again. 56.39 − 31.10 = 25.29 ms ≈ one YOLO service time.
At 25 Hz, where the period is 40 ms and the pool keeps up, the same flag does nothing
(30.47 → 30.70 ms).

The unpinned arms back up to the queue's full depth, and the arithmetic is exact:

| arm @45 Hz | depth 10 | depth 1 | Δ | Δ / camera period (22.22 ms) |
|---|---|---|---|---|
| `vanilla4` | 242.39 ms | 42.04 ms | −200.34 (5.77×) | **9.02** |
| `vanilla8` | 242.47 ms | 42.49 ms | −199.98 (5.71×) | **9.00** |
| `vanilla` | 264.77 ms | 64.91 ms | −199.87 (4.08×) | **8.99** |
| `multi` | 264.68 ms | 64.92 ms | −199.76 (4.08×) | **8.99** |
| `multi` @25 Hz | 433.76 ms | 73.59 ms | −360.17 (5.89×) | **9.00** (period 40 ms) |
| `ship` | 213.21 ms | 107.19 ms | −106.02 (1.99×) | 4.77 |
| `spin` | 119.03 ms | 60.08 ms | −58.95 (1.98×) | 2.65 |
| `p3` | 56.39 ms | 31.10 ms | −25.29 (1.81×) | 1.14 |
| `vanilla4x2` @36 Hz | 32.16 ms | 32.34 ms | +0.19 (none) | 0 |

**Nine camera periods is `depth − 1`.** A saturated ROS 2 subscriber with keep-last-10 hands the
callback a frame that is nine periods stale, and the chain reports that as latency. The figure's
242 ms baseline is 200 ms of DDS queue and 42 ms of work.

Cadence is essentially unaffected, except where the dropped frames free the executor thread:
`spin` 33.4 → 46.8 Hz (1.40×) and `ship` 18.8 → 27.9 Hz (1.49×), both single-threaded arms where
each surviving YOLO callback no longer blocks the control timer.

### 3.2 One YOLO node vs two pools

`vanilla4` → `vanilla4x2` (one perception process with a 4-hart pool → two perception processes,
pools on harts 0–3 and 4–7, camera alternating frames):

| camera | `vanilla4` | `vanilla4x2` | Δ latency | Δ cadence |
|---|---|---|---|---|
| 25 Hz | 30.99 ms / 25.0 Hz | 31.41 ms / 25.0 Hz | +0.43 | — |
| 30 Hz | 31.09 ms / 30.0 Hz | 31.38 ms / 30.0 Hz | +0.28 | — |
| **45 Hz** | 242.39 ms / 38.6 Hz | **37.04 ms / 45.0 Hz** | **−205.35 (6.54×)** | **1.16×** |
| 90 Hz | 136.86 ms / 39.0 Hz | 252.76 ms / 61.7 Hz | +115.90 (0.54×) | 1.58× |
| 120 Hz | 110.61 ms / 38.3 Hz | 199.99 ms / 61.0 Hz | +89.38 (0.55×) | 1.59× |

The second instance is worth 6.5× **only in the band where one pool saturates and two do not**.
Below 30 Hz one pool keeps up and it buys nothing; above ~62 Hz both saturate and the two-instance
arm is *worse* on latency (it holds two frames in flight instead of one) while buying 1.6× cadence.
Its own ceiling is visible in its sweep: cadence is `min(camera, ~62 Hz)`, and latency jumps from
68.4 ms at 60 Hz to 291.9 ms at 75 Hz.

**Widening one pool is not a substitute.** `vanilla4` → `vanilla8` (the same single perception node
given all eight harts) at 45 Hz: 242.39 → 242.47 ms, 38.6 → 38.5 Hz. Nothing, at any QoS or control
mode (`vanilla4tm`→`vanilla8tm` −0.56 ms; `vanilla4_q1`→`vanilla8_q1` +0.45 ms). The kernel stops
scaling past four harts — 24.58 ms at 4, 25.03 ms at 8 (`YOLO_STANDALONE_MS`) — so the win is
*pipelining*, not width.

**Pinning the pool's workers buys nothing either.** `vanilla4`→`vanilla4f` (pool harts unpinned):
+0.07 ms. `vanilla8`→`vanilla8f`: −0.19 ms. The pool pins its own workers; letting the OS place
them is free.

### 3.3 A nav pool

| pair | camera | without | with | Δ |
|---|---|---|---|---|
| `cp3n4` → `cp3n4_d` (nav 4-way over E-cluster) | 30 Hz | 30.13 ms / 29.9 Hz | 30.05 ms / 29.9 Hz | −0.08 ms |
| `vanilla4x2` → `vanilla4x2ns4a` (pool pinned 0–3) | 36 Hz | 32.16 ms / 36.0 Hz | 38.71 ms / 36.0 Hz | **+6.56 ms** |
| `vanilla4x2` → `vanilla4x2ns4b` (pool pinned 4–7) | 36 Hz | 32.16 ms | 39.17 ms | **+7.01 ms** |
| `vanilla4x2` → `vanilla4x2ns4c` (pool unpinned) | 36 Hz | 32.16 ms | 37.76 ms | **+5.60 ms** |

A nav pool buys nothing and costs up to 7 ms. Where there are spare harts (`cp3n4_d`: YOLO on
cluster 0, nav's four workers on cluster 1) it is free and pointless. Where there are none
(`vanilla4x2*`: both clusters already carry a YOLO pool) it takes harts from YOLO, and the cost is
the perception stage getting slower. Leaving the pool to the OS (`c`) is the cheapest of the three,
which is why the figures draw `c`.

### 3.4 Chained control vs a timer — the cadence lever

Control chained to the goal topic emits one command per goal, so the command rate is the
*pipeline's throughput*. Control on its own 100 Hz timer **in its own process** emits at 100 Hz
whatever the pipeline does.

| pair | camera | chained | timer | Δ latency | **Δ cadence** |
|---|---|---|---|---|---|
| `cp3` → `p3` | 15 Hz | 30.65 / 15.0 Hz | 30.51 / 100 Hz | −0.14 ms | **6.67×** |
| `cp3` → `p3` | 25 Hz | 30.62 / 25.0 Hz | 30.47 / 100 Hz | −0.15 ms | **4.00×** |
| `cp3` → `p3` | 30 Hz | 30.67 / 29.9 Hz | 30.40 / 100 Hz | −0.27 ms | **3.34×** |
| `cp3` → `p3` | 45 Hz | 56.15 / 38.8 Hz | 56.39 / 100 Hz | +0.24 ms | **2.58×** |
| `vanilla4x2` → `vanilla4x2tm` | 25 Hz | 31.41 / 25.0 Hz | 31.61 / 100 Hz | +0.19 ms | **4.00×** |
| `vanilla4x2` → `vanilla4x2tm` | 30 Hz | 31.38 / 30.0 Hz | 31.60 / 100 Hz | +0.22 ms | **3.33×** |
| `vanilla4x2` → `vanilla4x2tm` | 45 Hz | 37.04 / 45.0 Hz | 37.79 / 100 Hz | +0.75 ms | **2.22×** |
| `vanilla4` → `vanilla4tm` | 45 Hz | 242.39 / 38.6 Hz | 242.49 / 100 Hz | +0.10 ms | **2.59×** |

**This is the biggest single deployment lever, and it is free**: 2.2×–6.7× the command rate for at
worst +0.75 ms of latency, no extra harts, no rebuild — `--ctrl-mode chained` removed and the
control node launched as its own process.

Two conditions, both visible in the data:

* **The control node must have a thread YOLO cannot block.** `vanilla4t` is the same 4-hart graph
  with a 100 Hz timer but everything in *one* process behind one single-threaded executor: 32.9 Hz
  at 45 Hz camera. `spin` (also one process, timer): 33.4 Hz. `cship`→`ship` and `cspin`→`spin`
  move the cadence by 1.00× and 1.02× — chaining vs a timer makes no difference at all when the
  timer cannot fire. The lever is *control in its own process*, and "timer" is how you spend it.
* **Asking for more than 100 Hz only works where the thread is free.** `p3`→`p3_c200`: 100 → 200 Hz
  exactly, latency −0.35 ms. `spin`→`spin_c200`: 33.4 → 33.2 Hz, i.e. nothing — the timer cannot
  preempt the YOLO callback. Downward it always works: `p3_c50` 50.0 Hz, `vanilla4x2tm_c50`
  50.0 Hz.

**What the cadence is worth in flight, honestly.** Cell-matched over shared campaign cells:

| comparison | what changes | completed |
|---|---|---|
| `p3` vs `cp3` @45 | cadence 100 vs 38.8 Hz, latency equal (56 ms) | **19.6 % (40/204)** vs 12.3 % (25/204) |
| `vanilla4_q1` vs `vanilla4` @45 | latency 42 vs 242 ms, cadence equal (39 Hz) | **12.8 % (20/156)** vs 6.9 % (38/552) |
| `vanilla4tm` vs `vanilla4` @45 | cadence 100 vs 38.6 Hz, latency equal (242 ms) | 8.7 % (22/252) vs 7.6 % (51/672) |
| `p3_q1` vs `p3` @45 | latency 31 vs 56 ms, cadence equal (100 Hz) | 15.7 % (81/516) vs 17.5 % (105/600) |

Read together: **neither axis alone is sufficient.** Cadence buys 1.6× the completion rate when the
latency is already 56 ms, and nothing at all when it is 242 ms. Latency buys 1.9× when the cadence
is 39 Hz, and nothing (within the single-seed noise this campaign has) when it is already 100 Hz. The strongest baseline is the one that is
good on both, which is exactly `p3_q1` / `vanilla4x2tm`.

### 3.5 Process count

| pair | camera | Δ latency | Δ cadence |
|---|---|---|---|
| `p3` (3 proc) → `yproc` (2 proc, nav+control together) | 45 Hz | −0.74 ms | 1.00× |
| `p3` → `p8` (nav/control 2 harts each instead of 1) | 45 Hz | −0.17 ms | 1.00× |
| `p3` → `part8` (+ 2-hart pools under nav and control) | 45 Hz | −0.19 ms | 1.00× |
| `rp3` (4 proc) → `rp8` (5 proc, heavier stack) | 45 Hz | −0.42 ms | 1.00× |
| `ship` (1 proc) → `vanilla` (4 proc), no pool | 45 Hz | **+51.57 ms** | 1.10× |
| `spin` (1 proc) → `vanilla4` (4 proc), 4-hart pool | 45 Hz | **+123.36 ms** | 1.16× |

Process count *per se* is not a lever: 2, 3, 4 and 5 processes land within 0.8 ms of each other
once the nodes are pinned. Splitting an *unpinned* graph into more processes makes it **worse**,
because each new process is another unpinned scheduler entity competing for the same harts and each
new topic hop is another keep-last-10 queue to back up in (§3.1). What the three-process layouts
buy is not the processes, it is the pinning that goes with them.

**Per-node pinning**, measured against the same graph unpinned:

| pair | camera | unpinned | pinned | Δ |
|---|---|---|---|---|
| `vanilla4` → `cp3` (chained both) | 45 Hz | 242.39 ms / 38.6 Hz | 56.15 ms / 38.8 Hz | **−186.24 (4.32×)** |
| `vanilla4tm` → `p3` (timer both) | 45 Hz | 242.49 ms / 100 Hz | 56.39 ms / 100 Hz | **−186.10 (4.30×)** |
| `vanilla4` → `cp3` | 30 Hz | 31.09 ms | 30.67 ms | −0.43 ms |

Pinning is worth 4.3× at 45 Hz and nothing at 30 Hz, for the §3.1 reason: it is another way of
stopping the queue from forming, not a separate effect. Note that §3.1, §3.2 and this row are three
different single changes that each remove *the same* ~200 ms, and they do not compose — `p3_q1`
(pinned **and** depth 1) is 31.10 ms, not 31.10 − 186.

### 3.6 Executor

`ship` → `multi` (`MultiThreadedExecutor`, otherwise identical): at 15 Hz camera, cadence 41.6 →
100 Hz (2.40×) at identical latency (53.88 → 53.84 ms). At 45 Hz, cadence 18.8 → 100 Hz (5.32×)
but latency 213.21 → 264.68 ms (+51 ms): the threads fix the timer and then all fight for the same
harts, so the frame queue grows instead. `spin` → `smte` is the same shape, worse: 33.4 → 100 Hz,
119.03 → 258.65 ms. The multi-threaded executor buys cadence and pays for it in latency — the
opposite trade to §3.4, which buys cadence for free.

### 3.7 Kernels, and the one place the baseline is not given our best

At 30 Hz, with the whole deployment confined to cluster 0 (the placement an IME build has no choice
about, `ros_with_ime.md` §1):

| arm | kernels | camera→goal | control | harts |
|---|---|---|---|---|
| `vanilla4x2c0_rvv` | RVV | 31.10 ms | 30.0 Hz | 4 |
| `vanilla4x2c0_ime` | table-guided IME | **23.96 ms (1.30× lower)** | 30.0 Hz | 4 |

Giving the ROS arm the matrix engine is worth 7.14 ms — more than any layout change at 30 Hz — and
produces the lowest ROS 2 latency on record here. Spreading the IME build across both clusters
instead (`vanilla4x2_ime`, one perception instance, mask `0xf`) reaches 24.07 ms but collapses to
15.0 Hz control with 53 % of frames dropped, because the cluster-1 processes cannot run
`smt.vmadot`. See [`ros_with_ime.md`](ros_with_ime.md); this is not a gap, it is a documented trade,
but the 30 Hz figures should not claim a latency win over ROS 2 without it.

### 3.8 Things that buy nothing, for the record

Two unpinned CPU hogs (`HOGS=2`) alongside the run: `p3` −0.52 ms, `spin` +0.45 ms, `multi`
+0.75 ms — all inside replicate scatter. Every pinned arm is immune, and the unpinned ones are
already saturated. A 200 Hz control timer on a starved executor (`spin_c200`, `ship_c200`): 0.99×
and 1.00× the cadence. An 8-hart YOLO pool instead of 4 (§3.2): 1.00×.

### 3.9 The levers, ranked

| rank | choice | best measured effect | cost |
|---|---|---|---|
| 1 | control in its own process on a timer, not chained | **2.2×–6.7× command rate** | ≤ +0.75 ms latency, 0 harts |
| 2 | remove the standing frame queue — *any one of*: second YOLO instance (−205.35 ms, 6.54×), QoS depth 1 (−200.34 ms, 5.77×), per-node pinning (−186.24 ms, 4.32×) | **≈ −200 ms** = 9 camera periods | second instance needs 4 more harts; the other two are free; **they do not compose** |
| 3 | IME kernels for the baseline's YOLO | −7.14 ms (1.30×) | cluster 0 only, i.e. half the machine |
| 4 | a nav worker pool | −0.08 ms where harts are spare | **+5.6 to +7.0 ms** where they are not |
| 5 | 8-hart pool instead of 4; pinning pool workers; more processes; CPU hogs | ≤ 0.8 ms, either sign | — |

---

## 4. Reproduction

Every arm is one invocation of `scripts/ros_traced_matrix.sh <arm> [replicate]`. The camera rates
are the `RATES` environment variable (**not** `HZ`; the default is `5 8 10 12 15 20 25 30`), and the
sensitivity knobs are environment variables whose value is recorded in the run tag through `SUFFIX`,
so the tag is `<hz>_<arm><SUFFIX>_r<rep>`. Replicates are separate invocations:
`scripts/ros_traced_matrix.sh p3 1`, `... 2`, `... 3`.

The script runs on the K1 over ssh (`MODELBLASTER_K1_HOST`, default `k1`), archives the run on the
board, pulls it back in digest-checked chunks and writes
`results/codesign_feedback/ros_traced/<tag>/`. Board prerequisites (staging the kernels, the
`MB_WITH_POOL` / `_nav4` / `_ime` binaries) are in
[`ros_baseline_reproduction.md`](ros_baseline_reproduction.md) §1–2 and
[`ros_with_ime.md`](ros_with_ime.md) §2. **Never compile on the board while an arm is running.**

**The complete enumeration is generated.** The table below is written by hand and covers the arms
this ranking discusses. `artifact/implementations/ros_deployments/` enumerates every deployment that
has a run directory — 65 of them over 443 runs — and derives each one's command from
`ros_traced_matrix.sh` and the run manifests rather than restating it, so a new arm gets an entry the
day it is first run. `scripts/build_implementation_bundles.py --check` fails when the two disagree;
all 63 commands the two share are identical today, and the generated page declines to print one for
the single run whose manifest cannot support it (`vanilla4x2ns4`, its row below).

**Where each run's evidence lives.** `.gitignore` excludes
`results/codesign_feedback/ros_traced/*/trace.csv` — the raw per-dispatch traces are 3–12 MB each
and re-measurable — and archives them under `results/codesign_feedback/archive_v3/` with their
sha256 in the tracked `MANIFEST.sha256`. None is tracked; the traces a documented check opens are in
`board_traces_ros_traced_2026-09-29.tar.gz` (`docs/Artifact/external_data.md`, "Raw traces").
The evidence that is *not* re-measurable — `manifest.json`, `summary.json`, `ctrl_gaps.csv`,
`chain.csv`, `cpu.csv`, `kernel_sha256.txt` — stays tracked for all 445 runs. So every number in
§1–§3 above re-derives from tracked files alone; only the per-dispatch Gantt rows need the archive.

**Two gaps, listed so they are not discovered later.** 12 run traces are neither tracked nor in
`archive_v3` — `30_vanilla4x2_ime_r{1,2,3}`, `30_vanilla4x2c0_ime_r{1,2,3}`,
`30_vanilla4x2c0_rvv_r{1,2,3}`, `30_cp3n4_d_r{2,3}`, `36_vanilla4x2ns4_r1`. Their derived files are
tracked, so their rows in §1 stand; their dispatch-level traces would have to be re-measured. Two
runs have no `ctrl_gaps.csv` at all (`45_rmulti_r1`, `36_vanilla4x2ns4_r1`) — both recorded no
goals and are the runs the catalog already excludes.


### Per-arm commands and evidence

| arm | camera rates | command | `trace.csv` | `ctrl_gaps.csv` | `manifest.json` |
|---|---|---|---|---|---|
| `cp3` | 15 25 30 45 60 75 90 | `RATES="15 25 30 45 60 75 90" scripts/ros_traced_matrix.sh cp3` | tracked 12, archived 3 | tracked 15 | tracked 15 |
| `cp3n4` | 30 | `RATES="30" scripts/ros_traced_matrix.sh cp3n4` | archived 3 | tracked 3 | tracked 3 |
| `cp3n4_d` | 30 | `RATES="30" SUFFIX=_d scripts/ros_traced_matrix.sh cp3n4`<br>*the `--nav-pool 4 --nav-harts 4,5,6,7` build — the plain `cp3n4` rows predate that line, so the script as it stands reproduces `cp3n4_d`* | archived 1, **neither 2** | tracked 3 | tracked 3 |
| `cship` | 15 25 45 | `RATES="15 25 45" scripts/ros_traced_matrix.sh cship` | tracked 9 | tracked 9 | tracked 9 |
| `cspin` | 15 25 45 60 75 90 | `RATES="15 25 45 60 75 90" scripts/ros_traced_matrix.sh cspin` | tracked 12 | tracked 12 | tracked 12 |
| `multi` | 5 8 10 12 15 20 25 30 45 60 75 90 | `RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh multi` | tracked 24 | tracked 24 | tracked 24 |
| `multi_c200` | 45 | `RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh multi` | tracked 1 | tracked 1 | tracked 1 |
| `multi_hog2` | 45 | `RATES="45" HOGS=2 SUFFIX=_hog2 scripts/ros_traced_matrix.sh multi` | tracked 1 | tracked 1 | tracked 1 |
| `multi_q1` | 25 45 | `RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh multi` | tracked 2 | tracked 2 | tracked 2 |
| `nproc` | 15 25 45 | `RATES="15 25 45" scripts/ros_traced_matrix.sh nproc` | tracked 9 | tracked 9 | tracked 9 |
| `p3` | 5 8 10 12 15 20 25 30 45 60 75 90 120 | `RATES="5 8 10 12 15 20 25 30 45 60 75 90 120" scripts/ros_traced_matrix.sh p3` | tracked 24 | tracked 24 | tracked 24 |
| `p3_c200` | 45 | `RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh p3` | tracked 1 | tracked 1 | tracked 1 |
| `p3_c50` | 45 | `RATES="45" CTRL_HZ=50 SUFFIX=_c50 scripts/ros_traced_matrix.sh p3` | archived 3 | tracked 3 | tracked 3 |
| `p3_hog2` | 45 | `RATES="45" HOGS=2 SUFFIX=_hog2 scripts/ros_traced_matrix.sh p3` | tracked 1 | tracked 1 | tracked 1 |
| `p3_q1` | 25 45 | `RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh p3` | tracked 2 | tracked 2 | tracked 2 |
| `p8` | 5 8 10 12 15 20 25 30 45 | `RATES="5 8 10 12 15 20 25 30 45" scripts/ros_traced_matrix.sh p8` | tracked 11 | tracked 11 | tracked 11 |
| `part8` | 45 | `RATES="45" scripts/ros_traced_matrix.sh part8` | tracked 6 | tracked 6 | tracked 6 |
| `rmulti` | 25 45 | `RATES="25 45" scripts/ros_traced_matrix.sh rmulti` | tracked 4, archived 2 | tracked 5, **neither 1** | tracked 6 |
| `rp3` | 25 45 90 | `RATES="25 45 90" scripts/ros_traced_matrix.sh rp3` | tracked 6 | tracked 6 | tracked 6 |
| `rp8` | 45 | `RATES="45" scripts/ros_traced_matrix.sh rp8` | tracked 3 | tracked 3 | tracked 3 |
| `rspin` | 25 45 | `RATES="25 45" scripts/ros_traced_matrix.sh rspin` | tracked 4 | tracked 4 | tracked 4 |
| `rvanilla` | 15 25 30 45 60 90 | `RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh rvanilla` | tracked 18 | tracked 18 | tracked 18 |
| `rvanilla4` | 15 25 30 45 60 90 | `RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh rvanilla4` | tracked 18 | tracked 18 | tracked 18 |
| `rvanilla4x2tm` | 45 | `RATES="45" scripts/ros_traced_matrix.sh rvanilla4x2tm` | tracked 3 | tracked 3 | tracked 3 |
| `rvanilla8` | 45 | `RATES="45" scripts/ros_traced_matrix.sh rvanilla8` | tracked 3 | tracked 3 | tracked 3 |
| `rvanilla8tm` | 45 | `RATES="45" scripts/ros_traced_matrix.sh rvanilla8tm` | tracked 3 | tracked 3 | tracked 3 |
| `ship` | 5 8 10 12 15 20 25 30 45 60 75 90 | `RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh ship` | tracked 24 | tracked 24 | tracked 24 |
| `ship_c200` | 45 | `RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh ship` | tracked 1 | tracked 1 | tracked 1 |
| `ship_q1` | 25 45 | `RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh ship` | tracked 2 | tracked 2 | tracked 2 |
| `smte` | 5 8 10 12 15 20 25 30 45 60 75 90 | `RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh smte` | tracked 18 | tracked 18 | tracked 18 |
| `spin` | 5 8 10 12 15 20 25 30 45 60 75 90 | `RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh spin` | tracked 24 | tracked 24 | tracked 24 |
| `spin_c200` | 45 | `RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh spin` | tracked 1 | tracked 1 | tracked 1 |
| `spin_hog2` | 45 | `RATES="45" HOGS=2 SUFFIX=_hog2 scripts/ros_traced_matrix.sh spin` | tracked 1 | tracked 1 | tracked 1 |
| `spin_q1` | 25 45 | `RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh spin` | tracked 2 | tracked 2 | tracked 2 |
| `vanilla` | 15 25 30 45 60 90 | `RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh vanilla` | tracked 18 | tracked 18 | tracked 18 |
| `vanilla4` | 15 25 30 45 60 90 120 | `RATES="15 25 30 45 60 90 120" scripts/ros_traced_matrix.sh vanilla4` | tracked 21 | tracked 21 | tracked 21 |
| `vanilla4_q1` | 45 90 | `RATES="45 90" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4` | tracked 6 | tracked 6 | tracked 6 |
| `vanilla4f` | 45 | `RATES="45" scripts/ros_traced_matrix.sh vanilla4f` | tracked 3 | tracked 3 | tracked 3 |
| `vanilla4t` | 15 25 30 45 60 90 | `RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh vanilla4t` | tracked 18 | tracked 18 | tracked 18 |
| `vanilla4tm` | 45 90 | `RATES="45 90" scripts/ros_traced_matrix.sh vanilla4tm` | tracked 6 | tracked 6 | tracked 6 |
| `vanilla4x2` | 25 30 36 38 40 45 50 60 75 90 120 | `RATES="25 30 36 38 40 45 50 60 75 90 120" scripts/ros_traced_matrix.sh vanilla4x2` | tracked 11, archived 22 | tracked 33 | tracked 33 |
| `vanilla4x2_ime` | 30 | `RATES="30" BINSUF=_ime SUFFIX=_ime scripts/ros_traced_matrix.sh vanilla4x2` | **neither 3** | tracked 3 | tracked 3 |
| `vanilla4x2_q1` | 36 | `RATES="36" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4x2` | archived 3 | tracked 3 | tracked 3 |
| `vanilla4x2c0_ime` | 30 | `RATES="30" BINSUF=_ime SUFFIX=_ime scripts/ros_traced_matrix.sh vanilla4x2c0` | **neither 3** | tracked 3 | tracked 3 |
| `vanilla4x2c0_rvv` | 30 | `RATES="30" SUFFIX=_rvv scripts/ros_traced_matrix.sh vanilla4x2c0` | **neither 3** | tracked 3 | tracked 3 |
| `vanilla4x2d2` | 30 36 | `RATES="30 36" SUFFIX=d2 scripts/ros_traced_matrix.sh vanilla4x2`<br>*the second YOLO pool's per-shard rows were traced for this one; the current script has no knob for that* | archived 2 | tracked 2 | tracked 2 |
| `vanilla4x2ns4` | 36 | *not recoverable from the run* — it recorded no goal and only the camera process wrote a manifest, so the knobs it ran with are not in it. The `ns4a`/`ns4b`/`ns4c` rows below show the form the family takes. | `camera/` only, tracked 1 | **none** | tracked 1 |
| `vanilla4x2ns4a` | 36 | `RATES="36" BINSUF=_nav4 NAVPOOL=4 NAVHARTS=0,1,2,3 SUFFIX=ns4a scripts/ros_traced_matrix.sh vanilla4x2` | archived 1 | tracked 1 | tracked 1 |
| `vanilla4x2ns4b` | 36 | `RATES="36" BINSUF=_nav4 NAVPOOL=4 NAVHARTS=4,5,6,7 SUFFIX=ns4b scripts/ros_traced_matrix.sh vanilla4x2` | archived 1 | tracked 1 | tracked 1 |
| `vanilla4x2ns4c` | 36 | `RATES="36" BINSUF=_nav4 NAVPOOL=4 SUFFIX=ns4c scripts/ros_traced_matrix.sh vanilla4x2` | archived 1 | tracked 1 | tracked 1 |
| `vanilla4x2tm` | 25 30 45 75 120 | `RATES="25 30 45 75 120" scripts/ros_traced_matrix.sh vanilla4x2tm` | tracked 3, archived 12 | tracked 15 | tracked 15 |
| `vanilla4x2tm_c50` | 45 | `RATES="45" CTRL_HZ=50 SUFFIX=_c50 scripts/ros_traced_matrix.sh vanilla4x2tm` | archived 3 | tracked 3 | tracked 3 |
| `vanilla8` | 45 | `RATES="45" scripts/ros_traced_matrix.sh vanilla8` | tracked 3 | tracked 3 | tracked 3 |
| `vanilla8_q1` | 45 | `RATES="45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla8` | tracked 3 | tracked 3 | tracked 3 |
| `vanilla8f` | 45 | `RATES="45" scripts/ros_traced_matrix.sh vanilla8f` | tracked 3 | tracked 3 | tracked 3 |
| `vanilla8tm` | 45 | `RATES="45" scripts/ros_traced_matrix.sh vanilla8tm` | tracked 3 | tracked 3 | tracked 3 |
| `vanilla8tm_c50` | 45 | `RATES="45" CTRL_HZ=50 SUFFIX=_c50 scripts/ros_traced_matrix.sh vanilla8tm` | archived 3 | tracked 3 | tracked 3 |
| `vanilla_c50` | 45 | `RATES="45" CTRL_HZ=50 SUFFIX=_c50 scripts/ros_traced_matrix.sh vanilla` | archived 3 | tracked 3 | tracked 3 |
| `vanilla_q1` | 45 | `RATES="45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla` | tracked 1 | tracked 1 | tracked 1 |
| `vanilla_smoke` | 45 | `RATES="45" SECS=8 SUFFIX=_smoke scripts/ros_traced_matrix.sh vanilla`<br>*the 8 s smoke run that checked the traced binary before the matrix; not used by any figure* | tracked 1 | tracked 1 | tracked 1 |
| `x2p` | 45 | `RATES="45" scripts/ros_traced_matrix.sh x2p` | tracked 2 | tracked 2 | tracked 2 |
| `x2rmulti` | 45 | `RATES="45" scripts/ros_traced_matrix.sh x2rmulti` | tracked 2 | tracked 2 | tracked 2 |
| `x2rp3` | 45 | `RATES="45" scripts/ros_traced_matrix.sh x2rp3` | tracked 2 | tracked 2 | tracked 2 |
| `x2rspin` | 45 | `RATES="45" scripts/ros_traced_matrix.sh x2rspin` | tracked 2 | tracked 2 | tracked 2 |
| `x2spin` | 45 | `RATES="45" scripts/ros_traced_matrix.sh x2spin` | tracked 2 | tracked 2 | tracked 2 |
| `yproc` | 15 25 45 | `RATES="15 25 45" scripts/ros_traced_matrix.sh yproc` | tracked 9 | tracked 9 | tracked 9 |

---

## 5. Two defects in the generated catalog, found while ranking

Both are in `scripts/ros_arms_catalog.py`, not in the measurements. Neither changes a number in
§1–§3, which read the manifests and `summary.csv` directly, but both mislead a reader of
`ros_arms_catalog.md`.

**5.1 Eight timer arms are labelled "chained".** `describe()` renders a missing `ctrl_mode` as
"control None", and the page then prints "(chained to the goal)" for it. The earliest runs of
`p3`, `p8`, `yproc`, `nproc`, `multi`, `spin`, `ship` and `smte` predate the field; later runs of
the *same* arms record `ctrl_mode: timer`, `ctrl_hz: 100`, and
[`ros_baseline_reproduction.md`](ros_baseline_reproduction.md) §3 lists all eight as timer arms.
The measurement settles it: `p3` emits a command every 10.00 ms at a 45 Hz camera, which a chained
control node cannot do. Because `describe()` reads only the *lowest-sorted* rate's manifest
(`10_p3_r1` before `45_p3_r1`), the stale label wins for the whole arm. These eight are exactly the
arms §2 finds are stronger than the drawn ones, so the mislabel hides the finding: it makes `p3`
look like `cp3` with a different pinning rather than `cp3` with the control timer switched on.

**5.2 The flights column misses four arms.** The catalog looks for a replay trace named
`ros_<arm><hz>.csv`. Four traces do not follow that convention, so their arms read "not flown" when
they were flown: `ros_vanilla4x236ns4.csv` → `vanilla4x2ns4c` @36 (0/12),
`ros_spin45_r1.csv` → `spin` @45 (0/60), and `ros_vanilla4x230d2.csv` / `ros_vanilla4x236d2.csv` →
`vanilla4x2d2` (which turns out genuinely never to have been flown). The catalog also globs
`campaign*/campaign*.csv`, one level deep, which misses the 14 `campaign_scene/*/campaign.csv`
display runs — 192 further episodes. §1 uses `ARM_BY_TRACE[...].derive` and a recursive glob
instead, which is why a few of its flight counts are larger than the catalog's (`vanilla4` 87/1116
here against 86/1068 there).

---

## 6. What would settle the open questions

1. `RATES=36 scripts/ros_traced_matrix.sh vanilla4x2tm` ×3 — the one missing cell. Without it we
   cannot say the 36 Hz baseline is the strongest ROS 2 at 36 Hz (§2.4).
2. `RATES=45 QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh p3 2` and `... 3` — `p3_q1` @45, the
   strongest 45 Hz arm, rests on a single run (§2.1).
3. Fly `vanilla4x2tm` @45. It is the arm `measured_timing.py` calls "the strongest arrangement of
   ROS 2 we have been able to build" and it has never been flown.
