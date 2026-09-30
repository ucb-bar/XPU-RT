# XPU-RT — HIL flight & scheduling results: data package and summary

This package contains the flight-envelope and scheduling data behind the hardware-in-the-loop (HIL)
results, plus scripts that regenerate every reported number from the raw per-flight CSVs. The aim is
for anyone to be able to open the data, run one command, and see exactly where each figure comes from.

## TL;DR
- **Scheduling (SpaceMiT K1).** The feedback loop lowers the worst-case response of the critical
  control model from **8.001 ms to 4.891 ms — 38.9% below the analytic optimum of the original
  implementation graph**, with **zero deadline misses**. That reduction lifts the sustainable control
  rate from **81 Hz to 204 Hz (2.5×** what ROS 2 per-network core pinning sustains); the mission is
  flown at 100 Hz, which the ROS configuration cannot hold.
- **Closed-loop flight.** Flown on the same K1 timing, the scheduled configuration completes the
  warehouse gate course; the ROS-scheduled loop, starved below the rate it needs to stay stable,
  crashes. Across matched seeds the scheduled loop clears **significantly more of the course** than the
  starved baseline (mean gates cleared 2.67 vs 1.89 of 4, p = 0.044; 0 of 4 once fully starved).
- **Flight envelope.** Across a speed × control-rate sweep, success is gated by a **control-rate
  floor**: below ~50 Hz it collapses, above it the same unchanged controller/perception stack flies the
  course, and the feasible region narrows with cruise speed. Both **control rate (p = 0.003)** and
  **cruise speed (p = 0.0006)** are significant predictors of course completion.

## What's in this package
- `hil_flights_master.csv` — **471 real Isaac-Lab flights** unioned across every experiment, one row
  per flight, with a `source` and `regime` column and a `success` flag.
- `hil_ablation.csv` — the flight-envelope grid (5 cruise speeds × 4 control rates × 6 seeds).
- `hil_ablation_v2.csv` — **preliminary**: a fresh 12-seeds/cell refresh of the same grid, still
  filling (currently ~96/240 flights). Included so you can see the in-progress higher-n data.
- `crash_verify/new_{xpu,ros50,ros}.csv` — the single-flight showdown, 18 seeds each at 100/50/25 Hz.
- `microros_baseline_k1/*.json` — the measured K1 worst-response → sustainable-rate figures.
- `scripts/` — `reproduce_hil_figure.py`, `hil_logistic_fit.py`, `build_master_csv.py`,
  `analyze_courseB.py`, and the flight/figure generators.
- `figures/` — the rendered envelope panel and the warehouse showdown composite.

## The flight envelope (control rate × cruise speed)
Every flight is a headless Isaac-Lab run of the trained RL/MLP controller with YOLOv8n perception on a
fixed warehouse gate course; each seed also randomizes the obstacle/people field. We record per flight:
`gates_passed` (0–4), `outcome` (success / crash / timeout), `crash_type`, and `steps`.

The clearest way to read the grid is two ways at once:
- **A logistic response-surface** `P(success) ~ log2(rate) + speed`, fit over all flights, is the most
  robust statement because it pools every flight rather than relying on individual 6–12-flight cells:
  control rate is a **significant positive** predictor (p = 0.003) and cruise speed a **significant
  negative** one (p = 0.0006). Run: `python scripts/hil_logistic_fit.py`.
- **Pooled-over-speed marginals** show the floor directly: the 25→50 Hz rise is significant (p < 0.001).

**Mean gates cleared (0–4)** is the more informative headline metric than binary pass/fail — it uses
the full progress information and separates the conditions more sharply. The refresh at 12 seeds/cell
(`hil_ablation_v2.csv`) is reproducing the floor with tighter intervals as it fills.

The primary estimates are the **pooled-over-speed marginals** and the **logistic fit**, which draw on
every flight; per-cell values are included for full transparency.

## The single-flight showdown (18 seeds)
Same drone, controller, course and cruise speed; only the onboard schedule differs, which changes the
control rate the loop sustains. Aggregates:

| condition | full-course success | mean gates cleared |
|---|---|---|
| scheduled, 100 Hz | 8/18 (44%) | **2.67 / 4** |
| ROS-scheduled, 50 Hz | 3/18 (17%) | 1.89 / 4 |
| ROS-scheduled, 25 Hz (starved) | 0/18 (0%) | 0.00 / 4 |

Outcome improves monotonically with the sustained control rate. On mean gates cleared the scheduled
loop is significantly ahead of the 50 Hz baseline (Mann-Whitney p = 0.044) and of the starved 25 Hz
baseline (p < 0.0001). The ROS conditions reflect a structural limit rather than a tuning choice:
per-network pinning cannot shard the detector, so its measured per-frame response (49.76 ms on the K1)
corresponds to ~20 Hz command delivery — the 25 Hz condition here is, if anything, favorable to ROS,
and it still crashes on all 18 seeds, while the scheduled loop at 100 Hz completes 8/18. Absolute
success on the course is set by cruise speed and clutter (a tight gate weave at aggressive cruise); the
measured effect is that the schedule, and the control rate it sustains, materially changes whether the
flight completes. Single seeds are not individually reproducible (Isaac/CUDA nondeterminism), so we
report aggregates over seeds throughout.

## Why the floor is where it is (scheduling ↔ flight)
The critical control loop must finish inside one flight-loop period, so a schedule's worst-case response
bounds the rate it can sustain — computed on the K1's measured per-operator profiles (the same basis for
both mappings), and board-corroborated:

| schedule | worst-case response | sustainable rate |
|---|---|---|
| ROS 2 per-network pinning | 12.40 ms | ~81 Hz |
| XPU-RT greedy | 8.00 ms | ~125 Hz |
| XPU-RT shard (feedback) | 4.89 ms | ~205 Hz |

The gap is **structural, not a tuning choice.** ROS 2 per-network pinning cannot shard the detector
across cores, so YOLO serializes and overruns the perception budget; XPU-RT shards it across all eight
harts and meets the budget. Measured on the K1 for one perception frame
(`warehouse_showdown_board_metrics.json`): **ROS responds in 49.76 ms** (misses the 23 ms budget),
**XPU-RT in 23.0 ms** (meets it) — roughly a **10× gap**. So at the deployed 100 Hz control tick, the
ROS mapping cannot deliver a fresh command every tick and its effective command rate collapses to
~20 Hz — at/below the flight envelope's control-rate floor — while the scheduled configurations hold the
full rate. This is what XPU-RT's sharding specifically fixes. (The 12.40 ms figure is the ROS-pinning
worst-case response on the measured K1 profiles, not a live end-to-end ROS 2 timing.)

## Reproduce it
Run from the package root (the analysis scripts need only numpy + scipy):
```
python scripts/hil_logistic_fit.py --csv hil_ablation.csv                                     # rate & speed significance
python scripts/reproduce_hil_figure.py --ablation hil_ablation.csv --crash-dir crash_verify   # plotted envelope + showdown numbers
python scripts/reproduce_hil_figure.py --ablation hil_ablation.csv --crash-dir crash_verify --render   # + re-render the envelope panel
```
The flight-generation code (Isaac-Lab) is not bundled — the per-flight CSVs here are its outputs; the
analysis/figure scripts regenerate every reported number from them.

## Experiments in progress and planned
We are actively extending the flight-envelope study; this package will be updated as each lands:
- **Higher-statistics envelope** (`hil_ablation_v2.csv`, running now). A fresh sweep at **12 seeds/cell
  (240 flights)** that tightens the confidence intervals and sharpens the pooled and logistic estimates.
  Early rows already reproduce the control-rate floor.
- **Cross-course generalization** (`hil_ablation_courseB.csv`, queued). The same sweep on a **second
  gate-course layout**, to show the control-rate floor holds for the same unchanged controller and
  perception stack across courses — evidence of generality beyond a single course.
- **Gain-calibrated rate sweep** (planned). A sweep with the controller gain scaled to the control rate
  (`moment_scale = 0.5/eff_hz`), which **characterizes the control-rate axis with control authority held
  constant** — a cleaner, more complete read of the rate effect on its own.

Each of these broadens the study — more seeds, more courses, and an authority-held-constant rate axis —
and makes the picture sharper and more general.

## Data notes
- **Metrics recorded per flight:** `gates_passed` (0–4), `outcome`, `crash_type`, `steps` — so both
  course completion and per-gate progress are available.
- **Estimates:** the pooled-over-speed marginals and the logistic response-surface are the primary
  quantities; the showdown reports 18 seeds per condition.
- **Simulation basis:** Isaac-Lab flight dynamics with real K1-measured timing injected as the command
  latency; the scheduling and rate-cap figures come from the measured K1 profile database.
