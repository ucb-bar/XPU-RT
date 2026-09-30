# The warehouse showdown at a 45 Hz camera, against an unpinned ROS 2 deployment

This page reproduces `results/codesign_feedback/refined/showdown_45hz_pinned_vs_rosdefault_s1000.{png,pdf}`
(named `warehouse_showdown_paper_c14` until 2026-09-19; the old name recorded the display cruise speed,
which is the least important thing about it). Every drawn number is measured on the board or flown in
the simulator.

Companion pages: [`artifact_checklist.md`](../Artifact/artifact_checklist.md) for what is stored versus generated,
[`ros_baseline_reproduction.md`](../Baselines/ros_baseline_reproduction.md) for building and deploying the baseline,
[`showdown_cam30_solver_placed_reproduction.md`](showdown_cam30_solver_placed_reproduction.md) for the
30 Hz form in which the solver derives the placement.

---

## 1. The two arms

Both run the **same generated ModelBlaster kernels** for the same three networks on the same eight
harts of the same SpaceMiT K1 -- the verifier checks that both board runs carry the same staged YOLO
IR, `ir=0c783539626c`. Only the orchestrator differs.

| | XPU-RT | ROS 2 vanilla |
|---|---|---|
| arrangement | CP-SAT schedule, control on its own slot | 4 processes, single executor, **unpinned**, control chained to the goal |
| board run | `xpurt_long/trace_acpsat_hardr1_other_run1.csv` | `ros_traced/45_vanilla4_r1/` |
| camera→control | 56.86 ms | **242.10 ms** |
| control cadence | 10.00 ms (100 Hz) | 25.96 ms (39 Hz) |
| frames late | 0 of 44 | **760 of 766** |
| cadence trace replayed | `ctrl_traces/xpu_a_cpsat_hard.csv` | `ctrl_traces/ros_vanilla445.csv` |

**Read this baseline for what it is.** Unlike the 30 Hz forms, this one *is* backed up: 242 ms
camera→goal and 760 of 766 frames late. Its per-hart occupancy is lopsided -- `CPU_E#3` at 99.3 % and
`CPU_P#1..3` at 85-87 %, against `CPU_P#0` 3.3 %, `CPU_E#0` 1.0 % and `CPU_E#2` 2.8 % -- because
nothing pins the callbacks. A reviewer is entitled to ask whether a better-deployed ROS 2 would fare
better, and the answer is yes on command rate: see §7.

**Board-measured timing, simulated flight consequence**: the drone never leaves Isaac Lab. Each flight
replays its arm's measured control-output series (`--ctrl_trace`) and its measured camera→control
latency (`--percep_latency_ms`). This form uses **no goal hold** (`percep_hold_ms = 0`).

---

## 2. The displayed pair (needs a GPU)

```bash
scripts/queue_v3c.sh          # -> scripts/display_pair_search.sh, XLAT=56.8 RLAT=242 ROS_GATES=2
```

The search writes `results/codesign_feedback/campaign_v2/display_v3s_c<cruise>/` and logs every
attempt beside the accepted pair. `results/codesign_feedback/campaign_v2/campaign.csv` only **orders**
the search; nothing drawn depends on it, and it is not tracked.

Accepted pair -- **cruise 1.4 m/s, scene (layout seed) 1000**:

| arm | episode seed | outcome | gates | steps | control rate replayed |
|---|---|---|---|---|---|
| XPU-RT | **1100** | success | 4/4 | 1181 | 96.2 Hz |
| ROS 2 vanilla | **1000** | crash, hits a gate frame | **2/4** | 555 | 39.1 Hz |

Both flights are drawn from the same scene, but at **two different episode seeds**. That is a property
of the older `display_pair_search.sh`, which searches each arm separately; the newer
`display_same_env.sh` flies both arms at one seed. The panel states both seeds and the verifier
re-derives them from the dumps, so the difference is on the figure rather than behind it.

This is the only form in which the baseline clears **two** gates. At a 30 Hz camera it never has: 0 of
72 baseline flights across `campaign_static6` and `campaign_free30_eq` reach a second gate.

Dumps: `campaign_v2/display_v3s_c1.4/{xpu,ros}_s1000_figdata` (171 MB together, ignored by pattern).
They are archived in `results/codesign_feedback/archive_v3/display_dumps_v3.tar`, whose sha256 is in
the tracked `archive_v3/MANIFEST.sha256`; the archived copies are byte-identical to the ones the
render read, which `scripts/verify_archived_dumps.py` checks.

---

## 3. Panel A's scene census and panel D's ladder (need a GPU)

Panel A's legend counts the same scene the pair flies, twelve seeds per arm
(`results/codesign_feedback/campaign_scene/tall1000s/campaign.csv`, tracked):

| arm | completed | mean gates |
|---|---|---|
| XPU-RT · CP-SAT | **4/12** | 2.58 |
| ROS 2 vanilla | **0/12** | 1.08 |
| *(also in this census, not drawn)* XPU-RT · greedy | 0/12 | 0.00 |
| *(also in this census, not drawn)* ROS 2, control on its own timer | 3/12 | 2.42 |

The two rows the panel draws are the two arms the figure is about; the other two are recorded here
because they are in the same file and a reader of the CSV will see them. The timer variant is the
same point §7 makes: give ROS 2 an independent control timer and its flight outcome moves.

Panel D is a **ladder over scheduling quality**, four arms, six flights each, cadence only (no
latency, no hold), so the panel isolates what the command rate alone does
(`results/codesign_feedback/flight_energy_v2.csv`, tracked):

| arm | control | moment | modelled power |
|---|---|---|---|
| XPU-RT · CP-SAT | 96 Hz | 1× | 1× |
| XPU-RT · greedy | 75 Hz | 4.1× | 2.7× |
| ROS 2 vanilla | 39 Hz | 19.1× | 10.9× |
| ROS 2 as shipped | 21 Hz | 43.3× | 97.4× |

Two of the four rungs are ours, so the panel reads as a spectrum of scheduling quality rather than a
two-horse race. Both bars of every arm come from the same logged wrench array: there is no wattmeter
and no torque sensor in this path, and only the ratio between arms is meaningful.

---

## 4. Panel I, the onboard schedule (no hardware)

Both Gantt rows are built from the board runs into `schedules/measured_gantt_v3_{xpu,ros}.json` and
their `_metrics.json` sidecars. The window drawn is the one whose own latency distribution matches the
whole run's, so the picture is representative rather than a transient.

Per-hart occupancy over the run, from the board's own sampler:

| | P#0 | P#1 | P#2 | P#3 | E#0 | E#1 | E#2 | E#3 |
|---|---|---|---|---|---|---|---|---|
| XPU-RT | 50.7 | 32.8 | 42.3 | 29.2 | 46.4 | 25.7 | 41.4 | 22.6 |
| ROS 2 vanilla | 3.3 | 87.2 | 85.9 | 85.4 | 1.0 | 20.4 | 2.8 | **99.3** |

Each row also draws its own **command instants** as stripes across its lanes, with the count in the
window: 11 commands one every 10.0 ms against 4 one every 26.0 ms. That contrast is the figure's
claim, so it is drawn rather than left to be counted off the arrow rail.

---

## 5. Render (no hardware)

```bash
R=results/codesign_feedback
ENERGY_CSV=$PWD/$R/flight_energy_v2.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_v2/display_v3s_c1.4/xpu_s1000_figdata \
  --ros-dir $R/campaign_v2/display_v3s_c1.4/ros_s1000_figdata \
  --scene-records $R/campaign_scene/tall1000s --display-cruise 1.4 \
  --xpu-trace xpu_a_cpsat_hard.csv --ros-trace ros_vanilla445.csv \
  --xpu-label "XPU-RT · CP-SAT" --ros-label "ROS 2 vanilla" \
  --camera-hz 45 --gantt-prefix $PWD/schedules/measured_gantt_v3 --gantt-rows xpu,ros \
  --out $R/refined/showdown_45hz_pinned_vs_rosdefault_s1000
```

`--xpu-label` must name the solver recorded in the board manifest; the verifier re-reads the manifest
and checks it. `ENERGY_CSV` is an environment variable, not a flag.

---

## 6. Verification (no hardware)

```bash
.venv/bin/python scripts/verify_showdown_figure.py --metrics \
    results/codesign_feedback/refined/showdown_45hz_pinned_vs_rosdefault_s1000_metrics.json
.venv/bin/python scripts/verify_archived_dumps.py
artifact/verify_no_hardware.sh
```

`--metrics` reports **0 FAIL over 48 checks**. `artifact/verify_no_hardware.sh` runs it with the
repo-wide checks in one command.

It re-derives, from the recorded inputs: every input file's sha256; that both flights came from the
same scene and each carries the episode seed the panel states; panel A's outcomes, gate counts, gains
and per-arm latencies against the registry, plus the per-scene tally from the scene census; the a-d
moment timestamps; panels B and C's k/n per control rate from the ablation CSVs; panel D's ratios by
re-importing `energy_ratios()`; panels E and G's means from the dumps; and, for each Gantt row, the
chain median, frames-late, hart placement, the lanes each network is drawn across, and the provenance
of every trace, sampler and manifest behind it, including that both arms ran the same staged YOLO IR
and that the executed schedule table equals the solved one.

It does **not** re-derive: the panel B and C statistical annotations (the points-difference text, the
Wilson bands and the p-values), the panel D bar labels and ordering, panel A's photographic backdrop,
the a-d imagery itself, panels F and H (which record no number), or the Gantt drawing -- only the
sidecar numbers behind it.

---

## 7. What a reviewer will ask, and the honest answer

**"Your baseline is backed up -- is that not a strawman?"** It is backed up: 242 ms, 760/766 frames
late, and its occupancy is lopsided because nothing pins the callbacks. Three answers are on record.
First, the same scene census holds a ROS 2 variant with control on **its own timer** which completes
3 of 12 rather than 0 -- so the deployment choice, not ROS 2 itself, is what fails here. Second,
`docs/Baselines/ros_arms_catalog.md` records 59 measured ROS 2 deployments on this board, including pinned ones
that reach a 10 ms control gap. Third, and this is the form to point at,
`warehouse_showdown_cam45_ros_static6` flies a **hand-pinned** baseline whose pool runs at 84-98 %
busy and whose control lands at 38.8 Hz, and reaches the same verdict.

**"Why 45 Hz?"** It is the camera rate the Tier A showdown's own schedule panel names, and it is the
rate at which this baseline's command rate sits near the control-rate floor rather than far below it
-- which is why the baseline survives into the course here and not at 30 Hz.

**"Two different episode seeds?"** Stated on the figure and checked by the verifier (§2). The scene is
the same; the episode draw is not. Newer searches fly one seed for both arms.

**"Is panel D measured?"** The moment is; the power is a momentum-theory rotor model over the same
logged wrench. Only the ratio between arms is meaningful.

**"Can I re-fly it and get this?"** No -- flights are not bit-reproducible, which is why the search
re-verifies a pair on the scene it flies rather than trusting a census row, and why the census rather
than the single flight carries the claim.
