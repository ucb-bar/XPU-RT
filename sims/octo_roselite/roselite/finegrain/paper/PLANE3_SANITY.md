# Plane sweep (44 arms), final validation

**FINAL: 1760 of 1760 cells, 42,240 episodes, exactly 10 seeds per (task, arm) pair.**
(Snapshot 1 was 563 cells / 13,512 episodes / 2-5 seeds; deltas are called out below.)
Corrected drive-torque energy channel (`t2_drive_arm_sus`).

> **135 cells were recovered during this check.** The seed-300 pass ran before
> `job_plane3.sh` gained its `reduce_torque3.py` step, so 133 seed-300 cells and 2
> seed-301 cells held a complete `summary.json` and the full per-tick `.npy` set but no
> `energy2.json` — simulated, never reduced, and invisible to every count that keys on
> `energy2.json` (including the progress monitor). `g5grid/recover_plane3.sh run` reduced
> all 135 in place with 0 failures; no re-simulation was needed. Seeds 300-306 are now
> complete at 176/176. Without this they would have been lost when the instances stop.

```
/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/g5grid/fetch_plane3.sh
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/paper
python3 plane3_audit.py                                   # the numbers below
/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/g5grid/recover_plane3.sh count                          # simulated-but-unreduced cells
python3 fig_plane3_progress.py && python3 fig_plane3_check.py
```

## Integrity

176/176 (task, arm) pairs, 1760 distinct cell names in 1760 files (no duplicates),
every cell exactly 24 episodes. One cell scores 0/24: `spoon/g283_260` seed 305 — the
worst arm in the plane (283 ms period, ~11% success), whose sibling seeds score 1-5/24
and whose energy distribution is unremarkable. Expected at that rate, not a fault.

## Headline numbers

| task | success rho(period) | energy rho(period) | rho(success, energy) | replication r | resolution |
|---|---|---|---|---|---|
| egg | **−0.830** | +0.765 | **−0.945** | **+0.881** | 2.36x |
| spoon | **−0.843** | +0.757 | **−0.847** | **+0.870** | 2.57x |
| coke | −0.328 | +0.534 | −0.372 | +0.342 | 1.30x |
| drawer | +0.461 | −0.048 | +0.117 | +0.385 | 0.93x |

*resolution* = across-arm spread in success / mean per-arm SEM. Above ~1.3 the surface is
structure; at ~1.0 it is noise. Replication is against the earlier, independent 20-seed
sweep (seeds 100-129 vs 300-309). Replication offsets: egg −2.3 pts (0.7 noise bands),
spoon +0.7, coke +1.0, drawer +0.9 — all within tolerance.

## What changed since snapshot 1

Everything moved the right way.

- Trends **strengthened**: egg success rho −0.707 → −0.805, spoon −0.712 → −0.804, and
  egg's success/energy coupling −0.842 → **−0.950**.
- Replication **improved**: egg +0.775 → +0.860, spoon +0.739 → +0.874, drawer +0.138 → +0.321.
  coke drifted +0.321 → +0.275, consistent with a genuinely flat surface.
- Resolution improved on every task: egg 1.62 → 2.24, spoon 1.80 → 2.45, coke 1.20 → 1.30,
  drawer 0.99 → 1.02.
- **coke's energy estimator flip is resolved.** It was +0.296 (median) vs −0.073 (mean) and
  unquotable; it is now +0.439 vs +0.383, same sign, and can be quoted.
- **Energy outliers 2 → 0.** `drawer/g130_320` cleared with more seeds; `egg/g130_305`
  cleared once its recovered seed-300 replicate was restored. No panel denominator is an
  outlier any more.
- The drawer zero-torque mode is **stable to a tenth of a point**: 21.6% → 21.8%. It is a
  reproducible property of the task, not a sampling artefact.

## Findings

**1. Period, not latency, is the axis.** Spearman(latency, period) = −0.62 across the 44
operating points — they are a scheduler Pareto front, so an arm buys a shorter release
period by paying observation age. Regressing on latency alone reads backwards.

**2. Success and energy are co-optimised on widowx, not traded.** rho(success, energy) =
−0.950 egg, −0.816 spoon, −0.411 coke. Shorter period wins both at once.

**3. The eggplant histogram is the energy story in one panel.** Failures pile at ~1e6,
successes at 1e4-1e5: the expensive episodes are the ones that flail without finishing.
This is what the gravity-proxy metric could not see.

**4. close_drawer's energy channel is bimodal — its panel is not a "flat energy" result.**
21.8% of drawer episodes integrate below 100 N²m²s against a task median of 1.7e4: the arm
never moved for the full 1017 ticks, and **not one of them ever succeeded**. Other tasks
show 0.2-1.7%.

## Caveats still outstanding

- **coke and drawer success surfaces are not resolved** (1.29x and 1.03x the per-arm SEM).
  Consistent with the earlier finding that both are flat to within a few noise bands; the
  remaining ~440 cells will tighten but are unlikely to reveal structure.
- **drawer's energy trend sign-flips between estimators** (+0.066 median vs −0.241
  mean). Do not quote it. This is the bimodality of finding 4 showing up as instability,
  not a new effect — the coefficient is near zero under both.
- **Consider switching the energy estimator to mean-of-episodes.** Per-episode integrals
  span 10²-10⁵x within one cell, and the seed-to-seed CV of the cell statistic is roughly
  halved by the mean versus the median inherited from `fig_metrics3.py`: egg 0.150 vs
  0.360, spoon 0.191 vs 0.518, coke 0.169 vs 0.268 (drawer ties, 0.380 vs 0.354). Decide
  once on the finished sweep and apply to both figure sets.
