# The warehouse showdown against out-of-the-box ROS 2 at 45 Hz

This page rebuilds `fig_hil_showdown` in its paper layout with both arms measured on the K1, the ROS 2
arm being ROS 2 as it ships (`vanilla_c50`). The Tier A form ([`ros_baseline_tiers.md`](../Baselines/ros_baseline_tiers.md)) draws both arms modelled.
Its panel I titles the baseline "ROS serial on 1 hart backs up (~15 Hz) → control
starves". The deployment matching that title is ROS 2 as it ships -- one process per stage, unpinned,
serial YOLO, and **control running in the goal callback** (`scripts/ros_traced_matrix.sh:53`), so the
command rate is the goal-arrival rate of a chain that backs up. (The same panel's lanes draw a static
6-core partition; `scripts/campaign_static6.sh` flies that reading as `cp3`.) This is that same
figure — same layout, same panels, same camera rate, same cell — with every number coming from the
SpaceMiT K1 and from flights.

It is the Tier A pairing, with both arms measured (Tier C).

## The two arms

Both at a 45 Hz camera, each replaying its own board run, goal hold 0 on both sides.

| arm | board runs | control | camera→control | harts busy | replay trace |
|---|---|---|---|---|---|
| XPU-RT · CP-SAT | `xpurt_long/trace_acpsat_hardr{1,2,3}` | 100 Hz (10.00 ms gap) | 56.8 ms | 8 (22.6–50.7 %) | `ctrl_traces/xpu_a_cpsat_hard.csv` |
| ROS 2, serial YOLO, control in the goal callback | `ros_traced/45_vanilla_c50_r{1,2,3}` | **20.3 Hz** (49.16 ms mean gap) | 265.9 ms camera→goal | 1 (one core pegged at 100 %, migrating between runs) | `ctrl_traces/ros_vanilla_c5045.csv` |

Both run the same staged YOLO IR (`ir=0c783539626c`), so the comparison is two runtimes over one
network, not two networks. `scripts/measured_timing.py --verify` re-derives both rows (zero DRIFT);
`scripts/figure_constants.py` holds them as registered `ReplayArm`s, so no figure types them in.

The baseline's board side is produced by `scripts/board_ctrl50_arms.sh` (needs the board).

## 1. Flight census (GPU)

```bash
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_submitted_config.sh
```

Course a, prop density 0.30, people 2.4 m, gain 0.0055, cruise 1.0–1.8 m/s, twelve seeds per speed,
one flight per (cruise, seed) per arm. Flights are admitted one at a time under
`results/codesign_feedback/gpu_admit.lock`. No batch log carries a simulator fault, so nothing from
this campaign is quarantined.

```bash
.venv/bin/python scripts/census_submitted.py \
    --json results/codesign_feedback/campaign_submitted/census_submitted.json
```

`census_submitted.py` is the rule of `showdown_rate_sweep_reproduction.md` §3 as one command: rows
only through `flight_quarantine.flight_rows`, a cell only when both arms flew it, completions over
the non-timeout flights, a Newcombe (Wilson) interval on the completion difference, a percentile
bootstrap (4000 resamples, seed 11) over the PAIRED per-cell gate differences. It reproduces the
§3 r36 row from the same CSVs, which is what makes it usable here:

```bash
.venv/bin/python scripts/census_submitted.py \
    --csv results/codesign_feedback/campaign_rate3640/campaign.csv \
    --xpu xpu_w2pg36.csv --ros ros_vanilla4x236.csv    # +0.88 [+0.55, +1.22] gates, as §3 prints
```

Over 60 paired cells (5 cruise speeds × 12 seeds):

| arm | completed | non-timeout | mean gates |
|---|---|---|---|
| XPU-RT · CP-SAT | **9** | 59 | **2.03** |
| ROS 2, serial YOLO, control in the goal callback | **0** | 35 | **0.23** |

* completion difference **+15.3 pts**, Newcombe 95 % **[+3.1, +26.5]**
* gate difference **+1.80**, bootstrap 95 % **[+1.48, +2.13]**
* per cell: XPU-RT ahead on **51**, level on **9**, behind on **0**

| cruise | XPU-RT k/n | ROS 2 k/n | XPU-RT gates | ROS 2 gates |
|---|---|---|---|---|
| 1.0 | 1/12 | 0/5 | 1.92 | 0.17 |
| 1.2 | 2/11 | 0/4 | 2.00 | 0.08 |
| 1.4 | 3/12 | 0/7 | 2.25 | 0.17 |
| 1.6 | 2/12 | 0/11 | 1.92 | 0.33 |
| 1.8 | 1/12 | 0/8 | 2.08 | 0.42 |

The baseline never completes the course at any speed, and it times out 25 of its 60 cells before it
can crash — that is why its denominator is 35 rather than 60, and why the Newcombe interval is the
one statistic here that is not overwhelming.

## 2. The displayed pair (GPU)

```bash
MAX_SIMS=3 NEED_MB=10000 scripts/display_search_submitted.sh        # env CRUISES, OD
```

The display script flies ONE episode with `layout_seed = seed` through `record_sensor_demo.py`, so a
census cell is a different scene and a census outcome never carries over. A pair is kept only when,
on the scene it flies, XPU-RT completes the course and the baseline crashes after one or two gates
(`ROS_GATES='[12]'` — `verify_showdown_figure.py`'s `_compare_display_pair` accepts exactly that, and
a baseline that never enters the course is not a pair). Seeds are ordered per cruise with the
census's completing seeds first: a search order, not a selection — every attempt is logged and the
pair is accepted on what the display flight itself does. `CRUISES` and `OD` split the search over
several GPU-admitted processes.

The pair drawn: **cruise 1.4 m/s, seed 1000**.

| arm | control rate in flight | camera→control | outcome |
|---|---|---|---|
| XPU-RT · CP-SAT | 96.2 Hz | 56.8 ms | **completes, 4/4 gates in 1130 steps** |
| ROS 2, serial YOLO, control in the goal callback | **20.2 Hz** | 265.9 ms | **crashes after gate 1, at step 521** |

The baseline commands at 20.2 Hz in flight because that is the cadence the board measured: its control
node fires once per goal (`n_ctrl_fires == n_goals == 385` in the run's manifest), so the command rate is
the goal-arrival rate of the serial chain. The `CTRL_HZ` setting the run records is not what paces it.

The attempts before it, all logged in `campaign_submitted/display*/search_c*.log`: at cruise 1.4
seeds 1001 (2 gates), 1003 (1), 1009 (1) did not complete for XPU-RT; at 1.6, seeds 1000 (3), 1003
(1), 1007 (2); at 1.8, seed 1000 completed but the baseline timed out at 0 gates (not a pair),
1001 (3), 1007 (2).

## 3. Panel A's per-scene census (GPU)

Panel A's legend counts the same scene the pair flies, twelve seeds per arm:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
CELL=submitted_l1000 LAYOUT_SEED=1000 CRUISE=1.4 \
  ARMS="xpu:$T/xpu_a_cpsat_hard.csv:56.8:0 ros8:$T/ros_vanilla_c5045.csv:265.9:0" \
  scripts/scene_runs_pair.sh
```

→ `results/codesign_feedback/campaign_scene/submitted_l1000/campaign.csv`:
**XPU-RT 3/12, the baseline 0/12.**

## 4. Panel D's energy runs (GPU)

Panel D re-flies six seeds per arm with the commanded wrench logged, cadence only (no latency, no hold),
so the panel isolates what the command rate alone does:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces; R=$PWD/results/codesign_feedback
CONDS="xpu_cpsat_sub:$T/xpu_a_cpsat_hard.csv ros_serial50_sub:$T/ros_vanilla_c5045.csv" \
  CRUISE=1.4 ER=$R/energy_runs_submitted OUTCSV=$R/flight_energy_submitted.csv \
  scripts/run_energy_pair.sh
```

→ `flight_energy_submitted.csv`, n = 6 per arm; relative to XPU-RT the baseline draws
**71.6× the mean commanded body moment and 136.3× the modelled propulsive power**.

## 5. Panel I's Gantt (no hardware)

```bash
D=results/codesign_feedback
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$D/xpurt_long/trace_acpsat_hardr1_other_run1.csv:$D/xpurt_long/cpu_acpsat_hardr1_other_run1.csv:$D/xpurt_long/manifest_acpsat_hardr1_other_run1.json:schedules/fig_a_cpsat_hard_clamped.json" \
  --arm "ros:ros:$D/ros_traced/45_vanilla_c50_r1/trace.csv:$D/ros_traced/45_vanilla_c50_r1/cpu.csv:$D/ros_traced/45_vanilla_c50_r1/manifest.json" \
  --spec data/toplevel/wh_chain45_solve.json --window-ms 160 \
  --out-prefix $D/refined/submitted/measured_gantt_sub
```

| row | lanes with bars | chain median | control gap | frames late |
|---|---|---|---|---|
| XPU-RT · CP-SAT | 8 (22.6–50.7 % busy) | 56.86 ms | 10.00 ms mean | 0 / 44 |
| ROS 2 | 3 (`CPU_E#3` at 100 %) | 266.12 ms | 49.57 mean | **401 / 401** |

Every frame of the baseline misses its window, and the frame start lag is 211 ms at the median: the
serial YOLO never catches up, which is the backlog the Tier A showdown described.

## 6. Render and verify

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_submitted.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_submitted/display/search_c1.4/xpu_s1000_figdata \
  --ros-dir $R/campaign_submitted/display/search_c1.4/ros_s1000_figdata \
  --scene-records $R/campaign_scene/submitted_l1000 --display-cruise 1.4 \
  --xpu-trace xpu_a_cpsat_hard.csv --ros-trace ros_vanilla_c5045.csv \
  --xpu-label "XPU-RT · CP-SAT" \
  --ros-label "ROS 2 · serial YOLO, control in the goal callback" \
  --camera-hz 45 --gantt-prefix $R/refined/submitted/measured_gantt_sub --gantt-rows xpu,ros \
  --out $R/refined/warehouse_showdown_paper_submitted
```

`--xpu-label` must name the solver recorded in the board manifest (`cpsat`); the verifier checks it.
`ENERGY_CSV` is an environment variable, not a flag.

## What does not reproduce from a clean checkout

* The display dumps (`{xpu,ros}_s1000_figdata`, 154 MB together) are ignored by pattern, so panels A, a–d
  and the telemetry row cannot be re-rendered from a clean checkout without them. They are archived beside
  the other display sets with their sha256 in the tracked `archive_v3/MANIFEST.sha256`.
* Flights are not bit-reproducible: a seed re-flown can differ from the census, which is why the display
  search re-verifies each candidate on the scene it flies.
* The board half needs the K1: `scripts/board_ctrl50_arms.sh` for the baseline's runs and
  `docs/Baselines/ros_baseline_reproduction.md` for staging the kernels and building the node.
* `verify_showdown_figure.py --metrics` does not re-derive the panel B and C statistical annotations, the
  panel D bar labels, panel A's backdrop and clearance value, the a–d imagery, panels F and H, or the Gantt
  drawing itself -- only the sidecar numbers behind it.
* Panel D has no wattmeter: "moment (measured)" is the commanded body moment logged by the simulated flight
  replaying the board-measured cadence, and "power (modeled)" puts that same wrench through an X-quad mixer
  and momentum-theory rotor model. Only the ratio between arms is meaningful.
