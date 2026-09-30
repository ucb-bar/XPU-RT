# The 30 Hz showdown with the placement left to the solver

This page reproduces the two arms the warehouse figure compares at a 30 Hz camera. Both run the same
three generated networks on the same eight harts of the same SpaceMiT K1. What differs is who decides
where the work goes, and when control runs.

Companion pages: [`showdown_r30_reproduction.md`](showdown_r30_reproduction.md) for the earlier pair,
[`nav_sharding.md`](../K1/nav_sharding.md) for the navigation-width measurements this depends on,
[`partitioned_schedule.md`](../K1/partitioned_schedule.md) for the machine-restriction mechanism.

---

## 1. The two arms

| | ROS 2 · `cp3n4` | XPU-RT · `p30free` |
|---|---|---|
| placement | chosen by hand, as a deployment must | chosen by the solver |
| perception | 4-hart pool over the P cores (`taskset`) | 808 dispatches 4-wide on the P cluster |
| navigation | sharded 4 ways over the E cluster | mostly `CPU_P#0`, some E, some 4-wide |
| control | shares the E harts, in the goal callback | `CPU_E#3`, its own recurring slot |
| matrix engine | not used | **679 IME dispatches** |
| harts left idle | none | none |
| camera→control | **30.1 ms** | **26.75 ms** |
| command rate | **30 Hz** | **100 Hz** |
| board runs | `ros_traced/30_cp3n4_r{1,2,3}` | `xpurt_long/trace_p30freer{1,2,3}_other_run1.csv` |
| cadence trace | `ctrl_traces/ros_cp3n430.csv` | `ctrl_traces/xpu_p30free.csv` |

The scheduled arm is given **no placement instruction at all**: `data/toplevel/wh_chain30_free.json`
carries no `allowed_machines` and no `machine_width`. It is told only what the hardware can do -- the
matrix engine is legal on cluster 0 -- and what each width costs, and it chooses. That it puts the
detector on the P cluster is therefore a result rather than an instruction.

---

## 2. The baseline, on the board

```bash
for r in 1 2 3; do RATES="30" scripts/ros_traced_matrix.sh cp3n4 $r; done
.venv/bin/python scripts/pull_ros_traced.py 30_cp3n4_r1 30_cp3n4_r2 30_cp3n4_r3
```

`cp3n4` needs the node linked against the 4-way navigation build, so it selects
`ros_mb_chain_traced_pool_nav4` rather than the binary the other arms use; that binary is built once
and the arms already measured are not relinked. Building it, and the `nav2`/`nav4` objects it links,
is in [`nav_sharding.md`](../K1/nav_sharding.md).

Three replicates: control gap mean 33.40 ms (so 30 Hz), camera→goal 29.98 / 32.16 / 30.13 ms.
`pull_ros_traced.py` is a separate pass -- `ros_traced_matrix.sh` does not run it -- and it is given
the tags explicitly so it rewrites only those runs.

## 3. The scheduled arm: solve, then board

```bash
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_CPSAT_WORKERS=0 \
       XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain30_free.json \
    --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 1800 --use-profiled --board-calibration $CAL
REPS=3 scripts/board_partitioned30.sh \
    schedules/scheduled_wh_chain30_free_cpsat_profiled.json p30free gen/mb_shard_nav
```

`XPURT_CPSAT_PYTHON` must be set or the solve stops with "no interpreter with ortools found"; the
repository's own `.venv` carries ortools. The solve takes about 150 s unconstrained against about 20 s
when a cluster is named per network, because the option space is far larger.

`gen/mb_shard_nav` is the profile tree: it is `gen/mb_shard` with `fused_full`'s measured `topo_0_1`
and `topo_0_1_2_3` tables added. Without them a wider navigation is costed as one hart, so the solver
could not choose a width even where it pays.

Board: `makespan 490.08 ms`, `op_deadline_miss=0`, FEASIBLE against the codegen contract, three
replicates each verifying `max_abs_err=0` against the golden output, chain 26.75 ms in all three.

## 4. What the unconstrained solve establishes

| arm | placement | camera→control |
|---|---|---|
| `p30free` | solver decides, nothing pinned | **26.75 ms** |
| `p30w4imecp` | a cluster named per network | 26.75 ms |
| `p30efull` | both clusters taken whole by hand | 25.26 ms |
| `a30cpsat` | solver decides, engine not offered, one width profiled | 55.16 ms |

The first two agree, so the constraint is not what the chain rests on. The last is the same solver
without the two co-design inputs, which is what separates 26.75 from 55.16.

`p30efull`'s shorter chain is a phase effect and is recorded as one: its navigation stage is 0.78 ms
*longer*, and what moves is the wait from navigation finishing to the next control fire, 4.43 ms down
to 1.30, because control runs on a fixed 100 Hz tick. Giving control the whole cluster without
widening navigation reproduces 26.75 ms exactly, which is what separates the two explanations.

## 5. The cadence traces the flights replay

```bash
scripts/ctrl_trace_from_board.py results/codesign_feedback/ros_traced/30_cp3n4_r1/ctrl_gaps.csv \
    --out results/codesign_feedback/ctrl_traces/ros_cp3n430.csv --warmup-ms 3000
scripts/ctrl_trace_from_board.py \
    results/codesign_feedback/xpurt_long/trace_p30freer1_other_run1.csv \
    --out results/codesign_feedback/ctrl_traces/xpu_p30free.csv --warmup-ms 100
```

30.0 and 100.2 outputs/s respectively. Both arms are registered in `scripts/figure_constants.py` and
`scripts/measured_timing.py`; `measured_timing.py --verify` re-derives every recorded number from the
board runs with no drift.

## 6. Two censuses, because there are two defensible gain policies (need a GPU)

The controller's moment gain is a per-step impulse, so the same constant means different authority per
second at different command rates. There are two honest ways to set it, they answer different
questions, and the figure is built in both forms rather than one being chosen for the other.

```bash
scripts/campaign_free30.sh       # per-rate gain: 0.01667 at 30 Hz, 0.00500 at 100 Hz
scripts/campaign_free30_eq.sh    # one gain: 0.00500 for both arms
```

Each is two arms x four cruise speeds x twelve seeds, one camera period of goal hold. Read either only
through `scripts/flight_quarantine.py:flight_rows`.

**Per-rate gain** (`0.5 / eff_hz`) keeps each arm's per-step impulse right for its own rate, so no arm
is under-driven by an accident of arithmetic. The price is that it hands the two arms two different
controllers, and the comparison then moves two things at once.

**One gain for both arms** moves one thing: same controller, same number, only the command rate
differs. 0.00500 is the gain the scheduled arm already flew, so its 48 flights carry over unchanged
and only the baseline is flown for this census. It is also the baseline's best of the four gains
measured in `campaign_rosgain` (§8), so the single gain is the one that favours the baseline rather
than the scheduled arm.

Per-rate census (`campaign_free30/`), through the quarantine reader:

| arm | n | completed | mean gates | how the episodes end |
|---|---|---|---|---|
| XPU-RT `p30free`, gain 0.00500 | 48 | 3 | 1.90 | 45 crash, median 596 steps |
| ROS 2 `cp3n4`, gain 0.01667 | 36 | **0** | **0.00** | 27 out of step budget, 7 on the floor, 2 on a prop |

The baseline reaches the first gate in none of its flights at this gain, which is the same thing the
four-gain sweep in §8 reports for `vanilla4x2` and is the failure this form is drawn showing.

## 7. Which form draws which failure

The two censuses fail in different manners, so the two forms are drawn and verified on different rules,
and each render declares which it is:

| form | census | the baseline's drawn flight | render flag |
|---|---|---|---|
| per-rate gain | `campaign_free30/` | never reaches the first gate | `--baseline-fails-before-first-gate` |
| one gain | `campaign_free30_eq/` | crashes after one or two gates | (none; the default rule) |

`scripts/verify_showdown_figure.py` reads the declaration out of the sidecar and requires the drawn
flight to be that failure, rejecting the other. The declaration is made at render time rather than read
off the flight, so a render cannot satisfy the verifier by describing whatever it happened to draw.

The search scripts differ only in the acceptance they pass to `scripts/display_same_env.sh`:
`display_search_free30.sh` accepts `outcome=(crash|timeout) .*gates=0/4`,
`display_search_free30_eq.sh` keeps the `crash` after one or two gates the other forms use.

## 8. The flights each figure draws (need a GPU)

Both forms are flown by the same three stages, chained per form so only one runs at a time:

```bash
scripts/followon_free30.sh           # one gain: pair, then scene census, then energy
scripts/followon_free30_perrate.sh   # per-rate gain, once the first chain finishes
```

**The displayed pair.** The search flies one episode per seed with `layout_seed = seed`, so a census
cell is a different scene and a census outcome never carries over; the pair is accepted on what the
display flight itself does. Every attempt is logged beside the accepted one.

| form | cruise | seed | XPU-RT | ROS 2 | log |
|---|---|---|---|---|---|
| one gain | 1.6 | 1001 | success 4/4, 1003 steps, 100.2 Hz | crash after G1, 363 steps, 30.0 Hz | `campaign_free30_eq/display/search_c1.6.log` |
| per-rate | 1.4 | 1007 | success 4/4, 1133 steps, 100.2 Hz | **out of time** at G0, 1800 steps, 30.0 Hz | `campaign_free30/display/search_c1.4.log` |

The one-gain search spent 12 seeds at 1.4 and 12 at 1.2 before 1.6, where the scheduled arm completed
at the first two seeds tried; both logs are kept, so the number of attempts behind each pair is on
record rather than implied.

**Panel A's scene census**, twelve seeds per arm on the scene the pair flies:

| form | scene | XPU-RT completed | ROS 2 completed | mean gates |
|---|---|---|---|---|
| one gain | `campaign_scene/free30eq_l1001` | 0/12 | 0/12 | **2.33** vs 0.58 |
| per-rate | `campaign_scene/free30pr_l1007` | 0/12 | 0/12 | **1.75** vs 0.08 |

Neither arm completes the displayed scene twelve times over, so the count alone would report a tie that
the flights do not show. The panel therefore states the gates each arm reaches, which is what separates
them, and the verifier re-derives both numbers from the scene's own campaign table.

**Panel D's energy runs** re-fly six seeds per arm with the commanded wrench logged, cadence only -- no
latency, no hold -- so the panel isolates what the command rate alone does. The two arms fly different
gains in the per-rate form, and `run_energy_pair.sh` takes one gain, so that form runs the two arms as
two invocations into one CSV.

| form | CSV | moment | modelled power |
|---|---|---|---|
| one gain | `flight_energy_free30.csv` | **37x** | 21x |
| per-rate | `flight_energy_free30_perrate.csv` | **61x** | 119x |

Both bars in either figure come from the same logged wrench array; there is no wattmeter and no torque
sensor in this path, so only the ratio between arms is meaningful.

## 8a. Render and verify (no hardware)

```bash
R=results/codesign_feedback
COMMON=(--xpu-trace xpu_p30free.csv --ros-trace ros_cp3n430.csv
        --xpu-label "XPU-RT · solver-placed CP-SAT schedule"
        --ros-label "ROS 2 · 4-hart YOLO pool, 4-wide nav, all 8 cores"
        --camera-hz 30 --gantt-prefix $R/refined/free30d/measured_gantt_free30d --gantt-rows xpu,ros8)

ENERGY_CSV=$R/flight_energy_free30.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_free30_eq/display/search_c1.6/xpu_s1001_figdata \
  --ros-dir $R/campaign_free30_eq/display/search_c1.6/ros_s1001_figdata \
  --scene-records $R/campaign_scene/free30eq_l1001 --display-cruise 1.6 "${COMMON[@]}" \
  --out $R/refined/warehouse_showdown_cam30_solver_placed

ENERGY_CSV=$R/flight_energy_free30_perrate.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_free30/display/search_c1.4/xpu_s1007_figdata \
  --ros-dir $R/campaign_free30/display/search_c1.4/ros_s1007_figdata \
  --scene-records $R/campaign_scene/free30pr_l1007 --display-cruise 1.4 "${COMMON[@]}" \
  --baseline-fails-before-first-gate \
  --out $R/refined/warehouse_showdown_cam30_solver_placed_rate_gain
```

`--xpu-label` must name the solver recorded in the board manifest (`cpsat`); the verifier re-reads the
manifest and checks it. `ENERGY_CSV` is an environment variable, not a flag.

```bash
for f in warehouse_showdown_cam30_solver_placed warehouse_showdown_cam30_solver_placed_rate_gain; do
  .venv/bin/python scripts/verify_showdown_figure.py --metrics results/codesign_feedback/refined/${f}_metrics.json
done
```

Both report **0 FAIL**. Neither is in `verify_showdown_figure.REFINED_ALLOWLIST`, so `--all` checks them.

**What the wording of each form is keyed on.** A baseline that runs out of time struck nothing, so
panel A's title and legend, the a-d strip labels and panels E-G's titles and shading are chosen from
the baseline's own recorded outcome rather than assuming a collision; in the per-rate form the E-G axis
covers the baseline's whole 18 s flight and marks where the scheduled arm had already finished the
course. Panel I's note states the lanes each network's bars occupy and how many of those lanes the
trace itself named -- a sharded dispatch is traced on one hart and runs on four, and the note used to
report only the first while the bars drew the second.

---

## 9. The baseline's outcome does not depend on the gain it is flown at

The controller's moment gain is a per-step impulse, so a rate-independent constant under-drives a slow
arm; the law the gain-controlled grid uses is `moment_scale = 0.5 / eff_hz`, which at 30 Hz is 0.01667
rather than the 0.00550 the 100 Hz arm takes. A reader is entitled to ask whether the baseline fails
because of its command rate or because of the number it was flown at.

`scripts/campaign_rate30_gain.sh` with the baseline's cadence at four gains, cruise 1.4 and 1.8, twelve
seeds each (`results/codesign_feedback/campaign_rosgain/`):

| gain | completed | crash | timeout | mean gates | cleared >=1 gate |
|---|---|---|---|---|---|
| 0.00550 (the fast arm's) | **0/24** | 22 | 2 | 0.38 | 9/24 |
| 0.00800 | **0/24** | 10 | 14 | 0.17 | 4/24 |
| 0.01110 | **0/24** | 8 | 16 | 0.12 | 3/24 |
| 0.01667 (its own rate's) | **0/24** | 10 | 14 | 0.12 | 3/24 |

**Zero of ninety-six**, at every gain from the fast arm's constant to the one its own command rate
calls for. The gain-controlled grid says the same thing across rates rather than across gains: at each
rate's own gain, 0/60 at 20 Hz, 0/60 at 25 Hz, 0/48 at 33 Hz, 9/60 at 50 Hz, 16/60 at 100 Hz.

What the gain does change is the **manner** of failure. At the fast arm's constant the baseline is
under-driven, tracks sluggishly and meets an obstacle early -- 22 crashes, median 254 steps. At its own
rate's gain it is driven hard enough to avoid obstacles but cannot make progress, and most episodes run
to the step limit. Both are the same verdict; only one of them photographs as a crash.

This is why the figure is built in both forms rather than one. At **0.01667**, its own rate's gain, the
baseline is driven hard enough to avoid obstacles but cannot make progress, and the form drawn from that
census shows it never reaching the course. At **0.00500**, where it crashes in the course, it is also at
its best of the four gains measured here -- so the form that draws a crash draws it at the gain most
favourable to the baseline, not one chosen to produce the picture. Neither number changes the verdict:
0 of 96.

---

## 10. Panel I, the onboard schedule (no hardware)

```bash
X=results/codesign_feedback/xpurt_long; R=results/codesign_feedback/ros_traced/30_cp3n4_d_r1
S=schedules/scheduled_wh_chain30_free_cpsat_profiled_clamped.json
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$X/trace_p30freer1_other_run1.csv:$X/cpu_p30freer1_other_run1.csv:$X/manifest_p30freer1_other_run1.json:$S" \
  --arm "ros8:ros:$R/trace.csv:$R/cpu.csv:$R/manifest.json" \
  --window-ms 140 --out-prefix results/codesign_feedback/refined/free30d/measured_gantt_free30d
```

The baseline row is built from `30_cp3n4_d_r*` rather than from the runs the flights replay their
cadence from. Those are a second measurement of the same arm with the navigation node's pool recorded
in its manifest, which is what lets the panel place the navigation network; the cadence the flights
replay is unchanged between them, gap mean 33.40 ms on both and camera-to-goal 30.00 / 32.18 / 30.13
against 30.06 / 30.05 / 30.16.

Perception's four harts come from its own per-slice rows, measured where they ran. Navigation's do not
exist: its build shards inside the kernel at codegen time rather than dispatching through the runtime
pool, so `modelblaster_pool_trace_count` records nothing and a trace row carries only the callback
thread's hart. Those lanes are **credited** rather than measured -- the harts the manifest declares
(`nav_pool`, `nav_harts`), kept only where the sampler saw the lane busy -- and the callback's own hart
stays in `traced_target` so the two can be told apart. The threshold is 5 %, not perception's 50 %:
navigation carries one 3.6 ms kernel per 33 ms camera period, about a tenth duty even when every shard
runs.

Per-hart occupancy over the run, from the board's own sampler:

| | P#0 | P#1 | P#2 | P#3 | E#0 | E#1 | E#2 | E#3 |
|---|---|---|---|---|---|---|---|---|
| XPU-RT `p30free` | 82.9 | 38.5 | 36.4 | 36.3 | 7.5 | 5.8 | 4.0 | 47.2 |
| ROS 2 `cp3n4` | 77.0 | 67.5 | 66.4 | 66.3 | 12.7 | 9.8 | 10.2 | 9.2 |

Neither arm leaves a lane idle, and the window's bars put perception on `CPU_P#0-3`, navigation across
`CPU_E#0-3` and control on `CPU_E#2` for the baseline; the scheduled arm's widths come from the
schedule it executed.

---

## 11. What the commanded moment does across command rates, at each rate's own gain

Panel D draws two bars, one per arm, and a reader can reasonably ask whether the gap between them is a
property of the command rate or of the two particular deployments. `scripts/energy_rate_sweep_gain.sh`
answers it by replaying the baseline's cadence at six rates and the scheduled arm's at 100 Hz, six
seeds each, cruise 1.8, cadence only -- no latency, no goal hold, so nothing but the command rate and
its own gain differs between conditions.

```bash
scripts/energy_rate_sweep_gain.sh
```

→ `results/codesign_feedback/flight_energy_r30_rates_gain.csv`, 42 flights. Each condition flies
`moment_scale = 0.5 / eff_hz`, so no condition is under- or over-driven by a constant chosen elsewhere.

| command rate | gain flown | mean commanded \|M\| | ×  the 100 Hz arm | \|M\| per unit gain | maneuver fraction |
|---|---|---|---|---|---|
| 25 Hz | 0.01998 | 0.8501 | 49.4× | 42.5 | 0.996 |
| 30 Hz | 0.01667 | 0.8138 | 47.3× | 48.8 | 0.993 |
| 36 Hz | 0.01386 | 0.7796 | 45.3× | 56.2 | 0.994 |
| 40 Hz | 0.01244 | 0.7011 | 40.8× | 56.4 | 0.992 |
| 45 Hz | 0.01105 | 0.6337 | 36.8× | 57.3 | 0.991 |
| 60 Hz | 0.00794 | 0.0854 | 5.0× | 10.8 | 0.774 |
| **100 Hz** (scheduled arm) | 0.00499 | **0.0172** | 1× | 3.45 | **0.422** |

Within a condition the six flights agree closely -- 45 Hz spans 0.610 to 0.644, 25 Hz 0.813 to 0.872 --
so the ordering is not an artefact of which episodes happened to run long, and the durations, which
vary with where each flight ends, do not track the moment.

Two readings, and both belong in the record:

* **What the motors see** is the raw column: monotone in rate, and 37× to 49× the scheduled arm's
  everywhere below 60 Hz. This is the quantity the propulsive model turns into power.
* **How hard the controller is working** is the raw moment divided by the gain, which removes the part
  of the difference that is simply a larger per-step impulse. The gain spans 4.0× from 25 Hz to 100 Hz
  while the moment spans 49×, so most of the gap is not arithmetic: per unit of gain the under-rate
  conditions still command 12× to 17× what the 100 Hz arm does.

The maneuver fraction says the same thing in a third way. Below 60 Hz essentially all of the modelled
energy above hover goes into maneuvering, 0.99 at every rate; the 100 Hz arm spends 0.42.

The step between 45 Hz and 60 Hz is where both readings change character, which is the same place the
envelope panels put the boundary of the under-rate band.
