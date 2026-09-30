# The warehouse showdown at a 30 Hz camera, end to end

This page reproduces `results/codesign_feedback/refined/warehouse_showdown_paper_r30.{png,pdf}`: the
warehouse gate-course comparison between XPU-RT and a ROS 2 deployment of the same three networks on the
same SpaceMiT K1, with every drawn number measured on the board or flown in the simulator.

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md) for what is stored versus generated,
[`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) for building and deploying the baseline,
[`showdown_rate_sweep_reproduction.md`](showdown_rate_sweep_reproduction.md) for the same figure at a
36 Hz camera.

---

## 1. The two arms

Both arms run the **same generated ModelBlaster kernels** for the same three networks
(`yolov8_nano_64x96`, `fused_full`, `mlp_control`) on the same eight harts. Only the orchestrator differs.

| | XPU-RT | ROS 2 |
|---|---|---|
| arrangement | CP-SAT schedule, control on its own slot | two model instances across both clusters, default executor |
| board run | `xpurt_long/trace_a30cpsat_hardr1_other_run1.csv` | `ros_traced/30_vanilla4x2_r1/` |
| camera→control | 55.155 ms | **31.441 ms** |
| control cadence | 9.97 ms (100 Hz) | 33.31 ms (30 Hz) |
| frames late | 0 | **0** |
| cadence trace replayed | `ctrl_traces/xpu_a30_cpsat.csv` | `ctrl_traces/ros_vanilla4x230.csv` |

Two properties of the baseline decide how this figure should be read:

* **Its pipeline is not backed up.** Zero frames late, 31.4 ms camera→goal, YOLO spread over eight harts.
  A 30 Hz camera is the rate at which this arrangement keeps up; at higher camera rates the same
  arrangement backs up (see `ros_arms_catalog.md`).
* **It has the lower camera→control latency of the two arms**, and it occupies more of the board:
  mean hart occupancy 37.1 % against XPU-RT's 23.8 %, every one of the eight lanes in use on both sides.

What differs is *when control runs*. ROS 2 chains the control node to the perception output, so its command
rate is the camera rate, 30 Hz. XPU-RT schedules control in its own recurring slot, so it commands at 100 Hz
regardless of the camera. The flight consequence of that difference is the figure.

**Board-measured timing, simulated flight consequence**: the drone never leaves Isaac Lab. Each flight
replays its arm's measured control-output series (`--ctrl_trace`), its measured camera→control latency
(`--percep_latency_ms`) and one camera period of goal hold (`--percep_hold_ms 33.3`).

---

## 2. The census (needs a GPU)

```bash
scripts/campaign_rate30.sh
```

Five cruise speeds (1.0–1.8 m/s) × 12 seeds × both arms, course a, prop density 0.30, people 2.4 m,
controller gain 0.0055 → `results/codesign_feedback/campaign_rate30/campaign.csv`, 180 rows (the file also
holds a third arm, `xpu_fb30r1`, from the per-rate feedback experiment).

Read it only through `scripts/flight_quarantine.py:flight_rows`, which drops GPU-faulted batches.

| arm | completed | mean gates |
|---|---|---|
| XPU-RT · CP-SAT | 6/60 | 1.87 |
| ROS 2 · two instances | 0/60 | 0.32 |

Equal counts per arm, same cells, no exclusions.

---

## 3. The displayed pair (needs a GPU)

```bash
CRUISES="1.8" scripts/display_search_rate30.sh
```

The display script flies one episode per seed with `layout_seed = seed`, so a census cell is a different
scene and a census outcome never carries over — the pair is accepted on what the display flight itself
does, and every attempt is logged under `campaign_rate30/display30*/`. `ROS_GATES='[12]'` requires the
baseline to have entered the course, because `verify_showdown_figure._compare_display_pair` accepts a
baseline that clears one or two gates and rejects one that clears none.

Accepted pair — **cruise 1.8 m/s, layout seed 1001**:

| arm | outcome | gates | steps | control rate replayed |
|---|---|---|---|---|
| XPU-RT | success | 4/4 | 854 | 94.6 Hz |
| ROS 2 | crash | 1/4 | 319 | 30.0 Hz |

Dumps: `campaign_rate30/display30b/search_c1.8/{xpu,ros}_s1001_figdata/`.

Flights are not bit-reproducible: the same seed re-flown can differ from the census, which is why the
search re-verifies on the scene it flies. The census cell at cruise 1.4 seed 1001 has the same shape and
is the alternate if this pair is re-flown.

---

## 4. Panel A's scene census and panel D's energy runs (need a GPU)

Panel A's legend counts the same scene the pair flies, twelve seeds per arm:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
CELL=r30_l1001 LAYOUT_SEED=1001 CRUISE=1.8 \
  ARMS="xpu:$T/xpu_a30_cpsat.csv:55.2:33.3 ros8:$T/ros_vanilla4x230.csv:31.4:33.3" \
  scripts/scene_runs_pair.sh
```

→ `results/codesign_feedback/campaign_scene/r30_l1001/campaign.csv`: **XPU-RT 2/12, the baseline 0/12.**
No log from either arm matches the GPU-fault signature, so nothing is quarantined.

Panel D re-flies six seeds per arm with the commanded wrench logged, cadence only (no latency, no
hold), so the panel isolates what the command rate alone does:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces; R=$PWD/results/codesign_feedback
CONDS="xpu_cpsat_r30:$T/xpu_a30_cpsat.csv ros_x2_30:$T/ros_vanilla4x230.csv" \
  CRUISE=1.8 ER=$R/energy_runs_r30 OUTCSV=$R/flight_energy_r30.csv \
  scripts/run_energy_pair.sh
```

→ `flight_energy_r30.csv`, n = 6 per arm; relative to XPU-RT the baseline draws **34.1× the mean
commanded body moment and 26.9× the modelled propulsive power**.

---

## 5. Panel I, the onboard schedule (no hardware)

Both Gantt rows are built from the board runs by `scripts/make_measured_gantt_pair.py` into
`results/codesign_feedback/refined/rate30/measured_gantt_r30_{xpu,ros8}.json` and their `_metrics.json`
sidecars. The window drawn is the one whose own latency distribution matches the whole run's, so the
picture is representative rather than a transient.

Per-hart occupancy over the run, from the board's own sampler:

| | P#0 | P#1 | P#2 | P#3 | E#0 | E#1 | E#2 | E#3 |
|---|---|---|---|---|---|---|---|---|
| XPU-RT | 50.3 | 13.5 | 24.4 | 13.3 | 33.3 | 13.3 | 26.8 | 15.8 |
| ROS 2 | 39.8 | 37.6 | 52.4 | 33.5 | 5.6 | 44.3 | 50.8 | 33.0 |

---

## 6. Render (no hardware)

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_r30.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_rate30/display30b/search_c1.8/xpu_s1001_figdata \
  --ros-dir $R/campaign_rate30/display30b/search_c1.8/ros_s1001_figdata \
  --scene-records $R/campaign_scene/r30_l1001 --display-cruise 1.8 \
  --xpu-trace xpu_a30_cpsat.csv --ros-trace ros_vanilla4x230.csv \
  --xpu-label "XPU-RT · CP-SAT" \
  --ros-label "ROS 2 · two model instances on all 8 cores" \
  --camera-hz 30 --gantt-prefix $R/refined/rate30/measured_gantt_r30 --gantt-rows xpu,ros8 \
  --out $R/refined/warehouse_showdown_paper_r30
```

`--xpu-label` must name the solver recorded in the board manifest (`cpsat`); the verifier checks it.
`ENERGY_CSV` is an environment variable, not a flag.

---

## 7. Verification (no hardware)

```bash
.venv/bin/python scripts/verify_showdown_figure.py --metrics \
    results/codesign_feedback/refined/warehouse_showdown_paper_r30_metrics.json
.venv/bin/python scripts/measured_timing.py --verify
.venv/bin/python scripts/flight_quarantine.py
.venv/bin/python -m pytest tests xpu-rt/tests -q
```

`verify_showdown_figure.py --metrics` reports **0 FAIL over 42 checks**;
`artifact/verify_no_hardware.sh` runs this and the repo-wide checks in one command.

---

## 8. What this figure's verification does and does not cover

`verify_showdown_figure.py --metrics` re-derives, from the recorded inputs: every input file's sha256;
panel A's displayed outcomes, gate counts, gains and per-arm latencies against the registry, plus the
per-scene tally from the scene census; the a–d moment timestamps; panels B and C's k/n per control rate
from the ablation CSVs; panel D's ratios by re-importing `energy_ratios()`; panels E and G's means from the
dumps; and, for each Gantt row, the chain median, frames-late, hart placement and the provenance of every
trace, sampler and manifest behind it, including that both arms ran the same staged YOLO IR.

It does **not** re-derive: the panel B and C statistical annotations (the points-difference text, the Wilson
bands and the Fisher p-value), the panel D bar labels and ordering, panel A's photographic backdrop and its
clearance value, the a–d imagery itself, panels F and H (which record no number), or the Gantt drawing —
only the sidecar numbers behind it.

Panel D's "moment (measured)" is the commanded body moment logged by the *simulated* flight that replays the
arm's board-measured cadence, and "power (modeled)" puts that same logged wrench through an X-quad mixer and
momentum-theory rotor model (`scripts/flight_energy_model.py`). There is no wattmeter and no torque sensor in
this path; only the ratio between arms is meaningful.

Two further notes on what will not reproduce from a clean checkout:

* The display dumps (`{xpu,ros}_s1001_figdata`, 288 MB together) are ignored by pattern, so panels A,
  a–d and the telemetry row need them alongside. They are archived with the other display sets, their
  sha256 in the tracked `archive_v3/MANIFEST.sha256`.
* Flights are not bit-reproducible. The census cell at cruise 1.4 seed 1001 has the same shape and is
  the alternate if this pair is re-flown; when the search re-flew that cell it came out differently,
  which is why acceptance is on the display flight itself rather than on the census row.
