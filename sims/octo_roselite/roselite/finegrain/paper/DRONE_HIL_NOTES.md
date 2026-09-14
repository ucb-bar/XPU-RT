# Drone HIL flight data — schema, grouping decision, sanity checks

Companion to `fig_drone_velocity.py` / `fig_drone_velocity.png`.

Source: `/scratch/dima/rose-infra/RoSE/experiments/copied_drone_3gate/hil_data_package.zip`
(3.7 MB, 22 files, sha256 `877f0a4a7575999d…`). Extracted read-only to a scratch dir; the
original zip was not modified. The figure script reads the extracted tree via `$HIL_PKG`.

---

## 1. Schema — what one row means

**One row = one flight.** Every flight is a headless Isaac-Lab run of the same drone stack
(RL/MLP controller `nav_fused_v12_cnn.pt` + YOLOv8n perception) on a fixed 4-gate warehouse
course; the seed also randomises the obstacle/people/crate field.

`hil_flights_master.csv` = **471 flights**, the union of every outcome-bearing CSV, with
`source`, `regime` and `success` added by `scripts/build_master_csv.py`.

| column | meaning (from the package's own column dictionary, not guessed) | units |
|---|---|---|
| `seed` | RNG seed; also drives the randomised obstacle/people field | — |
| `cruise_speed` | **commanded** forward speed setpoint | m/s |
| `sim_dt`, `decimation`, `control_dt_ms` | physics step, control decimation, control period | s / — / ms |
| `sched_latency_ms` | worst-case command latency; motor command held (ZOH) between refreshes | ms |
| `hold_steps` | control steps per refresh = `ceil(latency / control_dt)` | — |
| `eff_cmd_hz` | effective command rate = `1000 / (control_dt_ms × hold_steps)` | Hz |
| `moment_scale` | controller action→moment gain | — |
| `gates_passed` / `K` | gates cleared / total gates (always 4) | — |
| `steps` | sim steps survived | — |
| `outcome` | `success` \| `crash` \| `timeout` | — |
| `crash_type` | e.g. `clip`, `ground`; blank when not a crash | — |
| `walk_speed` | **pedestrian** walking speed (NOT the drone) | m/s |
| `percep_hold_ms`, `percep_refresh`, `percep_latency_ms`, `percep_delay_steps` | perception-freshness knobs | ms / — |
| `gust`, `motor_tau`, `pipeline_zoh` | disturbance / actuator / pipeline knobs (mostly 0) | — |
| `source`, `regime` | provenance + experiment tag added by `build_master_csv.py` | — |
| `success` | `1` iff `outcome == "success"` (`build_master_csv.py`, verified in code) | 0/1 |

**Success is binary per flight and means the full course**: `success = int(outcome == "success")`,
and `outcome == "success"` requires all 4 gates. Not "gates passed", not "no crash" — a
`timeout` is a failure even if 3 of 4 gates were cleared.

### Conditions / arms swept (the `regime` column exists specifically to stop pooling)

| regime | n | speeds (m/s) | rates (Hz) | gain | notes |
|---|---|---|---|---|---|
| envelope (figure source) | 120 | 1.0–1.8 (5) | 25/33/50/100 | fixed 0.0055 | **fully crossed 5×4×6** |
| perc-freshness/safety | 116 | 1.0–1.5 (5) | 100 only | 0.0055 | separate perception study, crowd 0.1–3.0 |
| calibrated-gain grid | 96 | 1.0 and 1.4 only | 25–200 | `0.5/eff_hz` | **`sim_dt = 0.001`, 10× finer physics** |
| misc/verification | 85 | 1.0/1.2/1.4 | 50/100 | 0.005–0.01 | walking-crowd grids + singletons |
| showdown | 54 | **1.4 only** | 25/50/100 | calibrated | XPU 4.89 / ROS50 12.40 / ROS25 35.58 ms ZOH |

---

## 2. Grouping decision, and why

Success is binary per flight, so a success *rate* only exists over a group. The grouping key is
**(experimental arm) × (commanded cruise speed)**.

**Why the arm has to be in the key.** The package says so explicitly and repeatedly —
`README_hil_data.md`: *"Do not pool the two sources or read them cell-for-cell"*;
`build_master_csv.py`: the `regime` column exists so experiments are *"never silently pooled"*.
The arms differ in controller gain, perception hold, crowd density, latency model, **and physics
step**. Pooling them would manufacture a trend out of confounds.

**Panel A — the envelope arm (`hil_ablation.csv`).** This is the only arm designed to answer the
question: a fully crossed 5 speeds × 4 rates × 6 seeds. Coloured series are the per-rate cells
(n=6). Because the design is *balanced*, the pooled-over-rate marginal (n=24 per speed) is a
legitimate estimate and is drawn in grey. Dashed lines are the authors' own logistic response
surface `P(success) ~ log2(rate) + speed` (`scripts/hil_logistic_fit.py`), refit here.

**Why not bin an achieved velocity instead:** there is no achieved-velocity column (see §3.2).

**Panel B — replication check.** Control rate pinned at 100 Hz so rate cannot confound speed, and
each independent arm drawn separately. Arms with only two speeds get a dotted link — two points
are not a trend.

**Excluded from the velocity plot:** the showdown regime (all 54 flights at a single cruise speed
of 1.4 m/s — it cannot contribute a velocity trend at all) and `hil_ablation_v2.csv` (a separate
re-run, §3.1).

### Agreement with the authors' logistic fit

Refitting `hil_logistic_fit.py` reproduces its published coefficients exactly:
`log2(rate)` +0.715/SD, OR 2.04×, **p=0.0030**; `cruise_speed` −0.923/SD, OR 0.40×, **p=0.0006**
(n=120). The binned points agree with the fitted surface in the pooled sense — the marginal falls
from 12/24 at 1.0 m/s to 2/24 at 1.6 and 1.8 — **but the per-cell points do not track the smooth
surface**: the marginal is non-monotonic (0.50 → 0.21 → 0.29 → 0.08 → 0.08), the 1.2 m/s cell sits
below the 1.4 m/s cell, and every Wilson interval from 1.2 to 1.8 m/s overlaps every other. The
logistic model imposes monotonicity that the binned data does not actually show over 1.0–1.4 m/s.

---

## 3. Sanity checks (with numbers)

### 3.1 Flight-count reconciliation and de-duplication

| file | rows | in master? |
|---|---|---|
| `hil_flights_master.csv` | **471** | — |
| `hil_ablation.csv` | 120 | yes (`source=hil_ablation`) |
| `hil_ablation_v1_120flights_fixedgain.csv` | 120 | **no — excluded** |
| `hil_ablation_v2.csv` | 96 | **no — excluded** |
| `crash_verify/new_{xpu,ros50,ros}.csv` | 18 + 18 + 18 = 54 | yes |

- **`hil_ablation.csv` and the "v1 archive" are byte-identical** (both md5 `aba535eb…`). The README
  presents v1 as the archived original and `hil_ablation.csv` as the current source, but today they
  are the same 120 flights. Naively unioning the CSV files would have **double-counted 120 flights**.
  `build_master_csv.py` already excludes it; so does this figure. **0 rows dropped** as a result.
- **`hil_ablation_v2.csv` is a genuinely separate re-run, not extra seeds.** It shares 48
  `(seed, speed, rate)` keys with v1, and **25 of those 48 have a different outcome** — consistent
  with the package's own Isaac/CUDA nondeterminism note. It must not be pooled with v1 or appended.
- **Duplicates inside the master: 0.** On the key
  `(source, seed, cruise_speed, eff_cmd_hz, moment_scale, percep_hold_ms, percep_refresh, walk_speed)`
  there are no repeats. (A naive `(source, seed, speed, rate)` key shows 46 apparent repeats, but
  those are the walking-crowd grids re-using seeds across perception sub-conditions — distinct
  flights, correctly kept.) **No rows were dropped.**
- **One flagged oddity, not deduplicated:** 2 flights (`seed` 1001 and 1003, 1.4 m/s, 100 Hz, gain
  0.005) appear in both `crash_verify_new_xpu` and `hil_pilot` with *identical* `steps` (1018 and
  1055) and outcome, but different `sim_dt` (0.01 vs 0.001) and `sched_latency_ms` (4.89 vs 0.0).
  Two different configs producing byte-identical step counts twice is unlikely; this may be a
  re-tagged log. They sit in different arms and are 2 of 471 rows, so they cannot affect the figure.

**Doc discrepancies found:**
- `README_hil_data.md` says the master holds **435 flights** and the showdown is **6 seeds each**.
  Both are stale: the showdown is now 18 seeds each (54 total), 435 + 36 = **471**, which is what
  `RESULTS_SUMMARY.md` and the actual file say. Its "3/6 vs 0/6" showdown numbers are likewise the
  old 6-seed figures; the current data gives 8/18, 3/18, 0/18 (matching `RESULTS_SUMMARY.md`).
- `hil_ablation_courseB.csv` is referenced by the README and required by `scripts/analyze_courseB.py`,
  but **is not in the package** (that script cannot run).
- **Only 174 of 471 flights are traceable to a shipped raw CSV** (120 envelope + 54 showdown). The
  other 297 rows (`hil_dense`, `hil_pilot`, all `perc_crash_*`, the walkgrid/ratecliff sets) exist
  only inside the master; their source files were not bundled.

### 3.2 Is velocity commanded or achieved?

**Commanded.** `cruise_speed` is documented as *"commanded forward speed (m/s)"* and **no
achieved-velocity column exists in any file** in the package (all 6 CSV headers were enumerated;
the only other `*speed*` column is `walk_speed`, which is the pedestrians'). So there is no choice
to make — but the commanded value is defensible as a velocity axis, because it is **verifiably
flown**:

Implied path length `v × steps × sim_dt` on successful flights —

- envelope arm, 28 successes: median **14.55 m**, full range **13.99–16.04 m**, per-speed medians
  14.69 / 14.50 / 14.49 / 14.68 / 15.30 m at 1.0 / 1.2 / 1.4 / 1.6 / 1.8 m/s.
- all 471 flights' 141 successes: median 14.63 m, IQR 14.20–15.15 m.

Completion time therefore scales as 1/v on a fixed ~14.6 m course: the drone really does fly the
commanded speed. (The tail out to 22–24 m is the perception-freshness arm, where the drone detours
around walking people, so the *flown path* is genuinely longer than the course — not a data error.)

A per-flight achieved speed could only be derived from `steps` on **successful** flights, which
would condition the velocity axis on the outcome — circular. Rejected.

### 3.3 Missing / ambiguous data

- **Blank or NaN `cruise_speed`: 0** of 471. Nothing dropped.
- **Unresolvable `outcome`: 0** of 471. Every row is exactly one of `success` (141) / `crash` (275)
  / `timeout` (55).
- **Timeouts (55) are the only arguably ambiguous outcome.** They are counted as **failures**,
  following the authors' own `success = int(outcome == "success")`. This is safe here: **no timeout
  has `gates_passed == 4`** (max is 3), so not one is a hidden completion. Their distribution is
  gates 0/1/2/3 = 27/5/7/16.
- Blank cells in `percep_*`, `walk_speed`, `gust`, `motor_tau`, `pipeline_zoh` (218 rows) and
  `crash_type` (399 rows) mean "that run did not log that knob" — they are not used as predictors.
- **Nothing was silently dropped anywhere in this analysis.**

---

## 4. Honest verdict

**Over the envelope arm's full 1.0–1.8 m/s range, success does fall with commanded cruise speed,
and it is statistically significant** (per-SD odds ratio 0.40×, p=0.0006, n=120) — this reproduces
the authors' published claim.

**But the effect is not robust, and the figure says so:**

| test | n | speed effect | p |
|---|---|---|---|
| envelope, 1.0–1.8 m/s (authors' primary) | 120 | OR 0.40× | **0.0006** |
| envelope, restricted to 1.0–1.4 m/s | 72 | OR 0.62× | 0.10 (n.s.) |
| perception-freshness arm @100 Hz | 116 | OR 0.77× | 0.17 (n.s.) |
| walking-crowd arm @100 Hz | 58 | OR 1.05× | 0.85 (n.s.) |
| authors' own 12-seed refresh (`v2`) | 96 | OR 0.85× | 0.48 (n.s.) |
| calibrated-gain grid | 96 | separates completely (0/8 @1.0 vs 4/6 @1.4) | degenerate |

The significant result is **carried entirely by the 1.6 and 1.8 m/s cells, which only the envelope
arm ever flew** (2/24 and 2/24). Over the 1.0–1.4 m/s range that every arm shares, no arm shows a
speed effect, and the calibrated-gain grid trends the *opposite* way (0/8 at 1.0 m/s vs 4/6 at
1.4 m/s — though it also runs a 10× finer physics step, so it is not comparable).

**The defensible statement is a completion ceiling somewhere above ~1.4 m/s, not a smooth
velocity–success relationship.** Below 1.4 m/s, velocity and success show no relationship that
this dataset can resolve at n=24 per point.

**What the data supports much better than the velocity axis** is the *control-rate* axis, which is
the package's actual headline: rate is the stronger and better-replicated predictor (OR 2.04×/SD,
p=0.0030 in v1 and OR 1.97×/SD, p=0.0036 in the independent v2 refresh — the one coefficient that
*does* reproduce across re-runs), and the 18-seed showdown is monotone in rate at a single speed
(0/18 at 25 Hz → 3/18 at 50 Hz → 8/18 at 100 Hz).

---

## 5. Reproduce

```bash
unzip hil_data_package.zip -d <somewhere>
HIL_PKG=<somewhere>/hil_data_package python3 fig_drone_velocity.py
```
Prints the full k/n grid, the logistic coefficients, the per-arm 100 Hz table and every sanity
check above, and writes `fig_drone_velocity.png` (dpi 150).
