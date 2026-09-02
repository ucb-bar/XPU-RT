# QRB5165 schedule sweep — RESULTS

Written after the run. `SETUP.md` in this directory is the contract, written
before it; nothing here revises it. Every number below is measured, or is a
prediction a measurement is compared against. Where the two disagree, the
text names which of SETUP.md's documented confounds accounts for it.

Medians are over 3 reps; the spread is quoted with them. No single-rep
number appears as a result.

---

## 0. What ran, and what did not

    Sweep A   arms baseline, fused    seeds 0-7   --max-ops  800    16 points
    Sweep B   arm  fused_vint         seeds 0-1   --max-ops 2000     2 points

    18 generated -> 12 built and run (3 reps each, 36 runs) -> 6 REJECTED.

The six rejections are seeds 4, 5 and 7 in **both** matched arms, all on
predicate 4 (`no non-periodic work: nothing to pack against the periodic
load`) — those seeds draw zero yolov8n copies from the bank's `count {0,5}`
band.

**Correction (checked against the FPGA sweep's own artifacts).** An earlier
draft of this section claimed the FPGA sweep rejected the same three seeds,
and offered that as evidence the RNG stream ported faithfully. It did not:
`fpga_20260829-195805/results.json` contains all 16 points, seeds 0-7 in
both arms, so seeds 4/5/7 produced usable workloads there and were rejected
only here. The port therefore does NOT reproduce the FPGA's draw stream, and
no cross-target claim rests on it.

What does still hold is the property this sweep actually needs: within this
sweep, `baseline_seedN` and `fused_seedN` are drawn from the same stream and
differ only in the mid-size periodic slot — which is why the rejections are
symmetric across arms. The matched-pair comparison in Q1 is sound; the
"identical to the FPGA taskset" reading is not.

All 36 runs completed with `N/N entries executed`, zero skipped entries,
zero provenance mismatches, and a board lock wait of 0.28–0.80 s per rep
(never anywhere near a rep's own runtime, so no point needed re-running on
the contention rule).

---

## 1. Does fused beat dronet across many workloads?

**Weakly yes on direction, and the makespan is the wrong place to look for
it.** Over the five matched pairs, substituting `fused_split` for `dronet`
in the mid-size periodic slot is better in 3, worse in 1 and a tie in 1 — on
both the predicted and the measured makespan, and the two agree on which
seeds and by how much:

| seed | predicted makespan b -> f | measured (median of 3) b -> f | MOSEK status |
|---|---|---|---|
| 0 | 67.57 -> 66.34 ms (**-1.8%**) | 82.98 -> 83.23 ms (+0.3%) | optimal / optimal |
| 1 | 79.57 -> 84.57 ms (+6.3%) | 93.09 -> 98.87 ms (+6.2%) | inaccurate / inaccurate |
| 2 | 73.34 -> 73.34 ms (0.0%) | 90.48 -> 86.56 ms (**-4.3%**) | optimal / optimal |
| 3 | 48.40 -> 44.02 ms (**-9.0%**) | 52.68 -> 48.37 ms (**-8.2%**) | optimal / optimal |
| 6 | 91.69 -> 84.30 ms (**-8.1%**) | 109.31 -> 98.64 ms (**-9.8%**) | inaccurate / inaccurate |

3-rep spreads: 0.16–7.82 ms, median 4.44 ms (4.4% of the makespan). Held
against that noise floor the evidence is thinner than the table looks: only
seed 6 (−10.67 ms, spreads 4.64/5.41) is clearly outside it. Seeds 1
(+5.78 ms, spreads 2.69/4.24) and 3 (−4.31 ms, spreads 3.94/0.16) are
marginal, and seeds 0 (+0.25 ms) and 2 (−3.92 ms, spreads 7.82/6.59) are
inside it. **One clear win, two marginal (one in each direction), two
ties** is the honest reading of five points.

The FPGA sweep found fused better in 4 of its 5 surviving pairs (−5.6%,
−3.1%, +3.6%, −1.7%, −5.7%, predicted, greedy solver). Here it is better in
3 of 5, at −1.8% to −9.0% predicted and −4.3% to −9.8% measured. **The
direction reproduces and the magnitude is the same order** — a modest
confirmation across two very different targets, on a small number of points
with a 4.4% noise floor. It is not a result either sweep should carry
alone.

The reason it is weak is structural and is visible in the trace:

> **In all 12 points, the last entry to finish is a `yolov8n_head`.**

The makespan is bounded end to end by the detection chain, on whichever of
the DSP/CPU lanes the head landed. What the mid-size periodic model changes
is only how much that chain gets fragmented, so the substitution can only
ever move the makespan by second-order amounts.

Where the substitution *does* show up is the same place flow_c's README
reports it — the CPU lane:

| seed | CPU-lane busy ms (b / f) | mlp_control exec p50 (b / f) | mlp_control worst deadline overrun (b / f) |
|---|---|---|---|
| 0 | 52.60 / 54.43 | 0.082 / 0.106 ms | +7.20 / +9.57 ms |
| 1 | 69.89 / 62.62 | 0.067 / 0.068 ms | +11.05 / **+6.46** ms |
| 2 | 66.88 / **53.00** | 0.075 / 0.066 ms | +13.19 / **+8.24** ms |
| 3 | 0.42 / 6.10   | 0.065 / 0.060 ms | +0.03 / −0.00 ms |
| 6 | 57.99 / 73.23 | 0.064 / 0.079 ms | **−3.74** / +8.65 ms |

fused lowers CPU-lane occupancy in 2 of 5 and raises it in 3 — much less
one-sided than the single hand-built 4-way comparison in flow_c's README
(6.39 -> 1.48 ms of CPU work per instance). The difference is that in this
sweep the *solver*, not a manifest, decides where the fused tiles go, and
with the corrected `mlp_control@cpu` cell and a mostly-idle HTA it does not
always evacuate the CPU. `fused_tail` lands on the CPU lane in 17 of the 19
instances scheduled across the fused arm, which is where the extra CPU time
in seeds 0 and 6 comes from. Control-loop punctuality moves with the CPU
load and not with the arm: fused improves the worst overrun where it lowers
CPU occupancy (seeds 1, 2) and worsens it where it raises it (seeds 0, 6).

**Caveat that limits this answer:** seeds 1 and 6 were solved to MOSEK's
`optimal_inaccurate` — the 300 s limit cut the branch-and-bound off and the
incumbent was returned. Those two rows compare two incumbents, not two
optima. Seeds 0, 2 and 3 are proven optimal in both arms.

---

## 2. How does a heavy non-periodic vision model reshape a schedule?

`fused_seedN` and `fused_vint_seedN` are matched exactly: identical periodic
taskset (same periods, windows, phases) and the identical yolov8n draw. The
*only* difference is ViNT, plus the horizon extension ViNT forces, which
raises every periodic instance count.

| | seed 0 | seed 1 |
|---|---|---|
| horizon | 42.0 -> 78.9 ms | 47.7 -> 88.4 ms |
| operations | 23 -> 40 | 35 -> 57 |
| predicted makespan | 66.34 -> 94.34 ms | 84.57 -> 116.63 ms |
| measured makespan (median of 3) | 83.23 -> 103.55 ms | 98.87 -> 127.12 ms |
| CPU-lane busy | 54.43 -> **32.76** ms | 62.62 -> **32.17** ms |
| DSP-lane busy | 81.72 -> 101.01 ms | 89.11 -> 117.43 ms |
| HTA-lane busy | 53.47 -> 67.18 ms | 59.18 -> 81.46 ms |

The reshaping is real and it is precise:

> **Adding ViNT evacuates the CPU lane of detection work.** In 8 of the 10
> non-ViNT points exactly one `yolov8n_head` sits on the CPU lane
> (`3 dsp + 1 cpu`, `4 dsp + 1 cpu`, ...); the two exceptions are the seed-3
> pair, which draw only two yolov8n copies and fit both heads on the DSP. In
> **both** ViNT points — which carry four and five copies, more than any
> non-ViNT point that did put a head on the CPU — the count is zero:
> `4 dsp + 0 cpu` and `5 dsp + 0 cpu`.

ViNT's decoder composes on *nothing but the CPU* (`compose_failures`: DSP
rejects it with `Param[0] has incorrect Value 1`, HTA has no transformer op
set), so the solver reserves the CPU lane for it and pushes the detection
heads — which have a DSP alternative — onto the DSP. That is a placement
change caused by a single tile's capability constraint, not by its size.
CPU-lane busy time falls by 40–48% while DSP-lane busy time rises by 24–32%.

So a heavy non-periodic vision model reshapes *placement* here. It does not
reshape the *critical path*, which is the subject of question 3.

---

## 3. Does that answer survive when the accelerator is fast?

**The FPGA result does not reproduce here.** SETUP.md predicted this was the
most likely outcome, and the measurement agrees.

On the FPGA, ViNT is ~2.7 s standalone against ~30 ms for anything else in
the bank — roughly 100x — so it dominates any schedule it appears in by
construction. On this board it is not even the largest single model:

    ViNT   critical path  14.213 ms (encoders @ DSP) + 37.826 ms (decoder @ CPU) = 52.0 ms
    yolov8n serial on DSP 13.268 + 15.377                                        = 28.6 ms
    fused_split serial on DSP                                                    =  5.2 ms
    dronet best (DSP)                                                            =  0.65 ms

ViNT is 1.8x yolov8n, not 100x. And the arms carry **four to five** yolov8n
copies against **one** ViNT, so detection is 114–143 ms of work against
ViNT's 52 ms. The consequences, measured:

* **ViNT never bounds the makespan.** In both ViNT points the last entry to
  finish is a `yolov8n_head@dsp`, exactly as in the ten points without ViNT.
* **ViNT is a minority of the machine's busy time.** 37.55 ms and 42.68 ms
  of measured execute time (across the CPU and DSP lanes) inside 103.55 ms
  and 127.12 ms of wall clock — 36% and 34% of the makespan, spread over two
  lanes that are each also doing other work.
* **The schedule absorbs it.** Adding ViNT costs +20.3 ms and +28.3 ms of
  measured makespan on top of a 52 ms model, i.e. the solver hides roughly
  half of ViNT behind work that was already there.
* **Prediction gets *better*, not worse, when ViNT is added** (ratio
  1.25 -> 1.10 and 1.17 -> 1.09), because ViNT's two large tiles displace
  small CPU tiles whose per-dispatch overhead the cost model does not carry.

The honest statement is therefore: **on this target the "heavy vision model
reshapes everything" result is false as stated.** What survives is a
strictly weaker and different claim — a non-periodic model that composes on
only one backend reshapes *placement* on that backend (section 2), which is
a capability effect, not a magnitude effect. Rerunning this sweep with a
ViNT count of 4-5 instead of 1, or on a target where ViNT is 100x its
neighbours, would be needed to test the FPGA's claim on its own terms.

One number in SETUP.md's framing of this question has moved since it was
written — see section 6.

---

## 4. Predicted vs actual

Median makespan ratio over the 12 points: **1.17x** (range 1.09–1.25x),
with a median 3-rep spread of 4.44 ms (4.4%).

That is materially worse than the 1.00x flow_c's README reports for its
hand-built 4-way schedule under the same `--tuned` conditions, and the cause
is visible per lane:

| lane | entries | predicted | actual | ratio |
|---|---|---|---|---|
| hta | 240 | 1763.9 ms | 1926.5 ms | **1.092** |
| dsp | 342 | 2569.0 ms | 2992.3 ms | **1.165** |
| cpu | 492 | 1301.2 ms | 1709.9 ms | **1.314** |

Per tile, pooled over every run point (median of the per-point medians;
`obs` counts (point, network-copy) groups, each itself a median over 3 reps):

| tile @ lane | obs | predicted | actual p50 | ratio | ratio range |
|---|---|---|---|---|---|
| yolov8n_backbone @ hta | 38 | 13.909 | 13.951 | **1.003** | 0.88 – 2.04 |
| yolov8n_backbone @ dsp | 11 | 13.267 | 13.191 | **0.994** | 0.89 – 1.18 |
| yolov8n_head @ dsp | 41 | 15.377 | 17.468 | 1.136 | 0.99 – 1.76 |
| yolov8n_head @ cpu | 8 | 39.464 | 52.166 | 1.322 | 1.24 – 1.45 |
| vint_encoders @ dsp | 2 | 14.213 | 16.084 | 1.132 | 1.12 – 1.14 |
| vint_decoder @ cpu | 2 | 37.826 | 22.592 | 0.597 | 0.59 – 0.60 |
| fused_tail @ dsp | 4 | 4.303 | 3.406 | 0.792 | 0.76 – 0.81 |
| dronet_full @ hta | 7 | 2.030 | 1.814 | 0.894 | 0.68 – 7.11 |
| fused_vision_conv @ hta | 7 | 0.931 | 1.122 | 1.205 | 1.12 – 1.28 |
| fused_depth_conv @ hta | 4 | 1.568 | 0.880 | 0.561 | 0.43 – 0.66 |
| mlp_control_full @ dsp | 8 | 0.589 | 0.669 | 1.135 | 0.65 – 2.68 |
| dronet_full @ dsp | 7 | 0.645 | 1.215 | 1.884 | 1.29 – 3.12 |
| fused_vision_conv @ dsp | 6 | 0.459 | 0.855 | 1.862 | 1.06 – 3.56 |
| fused_depth_conv @ dsp | 4 | 0.392 | 0.742 | 1.893 | 1.17 – 3.47 |
| fused_tail @ cpu | 7 | 0.354 | 1.374 | 3.881 | 2.63 – 8.45 |
| mlp_control_full @ cpu | 12 | 0.107 | 0.069 | 0.645 | 0.53 – 0.90 |
| dronet_full @ cpu | 2 | 6.998 | 10.154 | 1.451 | 1.20 – 1.70 |
| fused_depth_conv @ cpu | 7 | 0.014 | 0.137 | 9.786 | 5.07 – 19.21 |

**Every ratio far from 1.0 is one of the three confounds SETUP.md names.**
Taking them in order of how far they are off:

* **`fused_depth_conv@cpu` 9.79x, `fused_tail@cpu` 3.88x, `yolov8n_head@cpu`
  1.32x, `dronet_full@cpu` 1.45x — CPU load-dependence.** This is the
  confound SETUP.md put first and flow_c's README documents as unfixable by
  measurement: the QNN CPU op package builds its thread pool at bringup on
  the main thread with full-machine affinity, so a CPU tile's cost is a
  function of what runs beside it. `fused_tail` is the canonical case — the
  README records 0.35 ms alone and 3.69 ms in situ; this sweep gets 1.374 ms
  in situ across 7 points, i.e. the same effect at a lower concurrency.
  `fused_depth_conv@cpu`'s 9.79x is the same cause amplified by a second:
  the cell is **14 µs**, below the noise floor of anything that shares a
  core, so a 120 µs absolute error reads as 10x.

* **`dronet_full@dsp` 1.88x, `fused_vision_conv@dsp` 1.86x,
  `fused_depth_conv@dsp` 1.89x — sub-millisecond DSP tiles pay a fixed
  FastRPC round trip the cell amortises away.** The measurements file puts
  DSP per-dispatch overhead at 0.367 ms (synthetic 1-MAC probe). Adding it
  to each cell predicts 1.012 / 0.826 / 0.759 ms against 1.215 / 0.855 /
  0.742 ms measured — i.e. the overhead accounts for essentially all of the
  gap. The cells are loop means over 40–50 iterations, which amortises a
  cost the runtime pays once per instance. This is a **cost-model gap, not a
  measurement error**, and the size dependence proves it: every DSP tile
  over 10 ms — yolov8n's backbone (0.994x) and head (1.136x), ViNT's
  encoders (1.132x) — sits within 14% of its cell, because 0.37 ms is a
  rounding error at 14 ms and a doubling at 0.4–0.65 ms.

* **`fused_depth_conv@hta` 0.56x, `dronet_full@hta` 0.89x with a 0.68–7.11
  range — HTA noise on sub-2 ms graphs.** SETUP.md states these cells are
  "order-of-magnitude only" (measured ±4% best case but 0.26–2.96 ms p99 on
  the dispatch probe), and the range here is exactly that. The two HTA tiles
  above 10 ms (`yolov8n_backbone@hta`, 38 point-observations) land at
  **1.003x**, which is the best single number in this sweep.

* **`vint_decoder@cpu` 0.597x — the cell was measured in the wrong state,
  and this sweep is the third data point on why that does not converge.**
  The frozen cost model carries 37.826 ms for this cell, a `feedback`
  promotion from the `flowc_vint_nav_qrb5165_nav` run. In *this* schedule it
  measures 22.59 ms (0.59x), because this schedule gives the decoder a
  different share of the machine. flow_c's README records the same tile at
  12.7 ms standalone -> 37.8 ms in situ -> 12.8 ms after promotion. A scalar
  cell cannot express a cost that depends on concurrent load; nothing in
  this sweep contradicts that, and this sweep did **not** run `feedback`.

* **`mlp_control_full@cpu` 0.645x — the corrected cell is conservative in
  situ, which is the right direction.** The coordinator's correction landed
  before any point was solved: the CPU tile is now the fp32 context at
  107.3 µs, not the int8 one at 28.5 µs, because QnnCpu's int8 path returns
  a constant for this network regardless of input. Measured in situ it is
  69 µs. So the corrected cell over-predicts by ~1.55x — a schedule solved
  against it leaves the control loop more headroom than it needs, which is
  the safe error. Every schedule in this sweep dispatches
  `ctx_mlp_control_fp32__Cpu.bin` on its CPU placements; this is verified
  per entry, per rep, in `provenance.json`.

### Control-loop punctuality

`mlp_control` executes in 0.060–0.127 ms p50 everywhere — the tile itself is
never the problem — but its *start* is frequently late, and what it shares
its lane with is why:

| point | largest CPU tile | worst overrun past deadline | misses |
|---|---|---|---|
| baseline_seed3 | 0.09 ms | +0.03 ms | 3/6 (all ≤25 µs) |
| fused_seed3 | 3.69 ms | −0.00 ms | 0/6 |
| baseline_seed6 | 57.05 ms | **−3.74 ms** | **0/10** |
| the other nine points | 22.3 – 53.7 ms | +5.13 to +19.20 ms | 1–6 of 6–18 |

The two points whose CPU lane holds nothing bigger than 3.7 ms are punctual
to the microsecond. In nine of the remaining ten a 22–54 ms CPU tile (a
`yolov8n_head@cpu`, or ViNT's decoder) sits in front of the control loop and
the gate cannot recover, costing +5 to +19 ms.

`baseline_seed6` is the instructive exception and it is not noise: it carries
a 57 ms `yolov8n_head@cpu`, all ten `mlp_control` instances are also on the
CPU lane, and it still misses nothing — MOSEK found an arrangement that runs
the whole 50 ms control sequence *before* the big tile starts. So the
constraint is expressible and sometimes satisfiable; the solver simply is
not asked to satisfy it, because **the cost model has no term for what a CPU
tile does to its lane-mates**. Where the geometry happens to work out the
control loop survives; where it does not, nothing in the objective notices.

---

## 5. Solver

| | |
|---|---|
| MILP (MOSEK), proven `optimal` | 6 points — seeds 0, 2, 3 in both arms |
| MILP (MOSEK), `optimal_inaccurate` at the 300 s limit | 6 points — seeds 1, 6 in both arms, and both ViNT points |
| greedy_periodic fallback | **0 points** |
| solve time | 2.3 s – 310.8 s (median 250.5 s) |

Op counts ran 16–57, well inside the range SETUP.md expected MOSEK to
handle, and it returned inside its limit every time — so the fallback was
never triggered. Solve time rises steeply with the op count (16 ops in 2.3 s,
20 in 11.3 s, 26 in 62.6 s, 32 in 195.9 s) but the limit is not a clean
function of it: 25 ops (`baseline_seed1`) hit the 300 s wall while 32 ops
(`fused_seed2`) proved optimality in 196 s. Structure, not size, decides.

**Deviation from SETUP.md, recorded:** SETUP.md planned `greedy_periodic`
outright for Sweep B, on the grounds that MOSEK "did not converge at 5576
ops". Sweep B here is 40 and 57 ops, not 5576 — this flow's coarse dispatch
graphs are one op per tile — so MILP was tried and returned inside its
limit, and the operative rule (fall back only when MILP does not return) was
followed. Both ViNT points are MILP incumbents, and are labelled as such in
`results.json` (`solver_status`).

---

## 6. The cost model this sweep solved against

Frozen once, before any solve, as `cost_model.json`
(sha256 `b4f1a684…29311d81`), and used by every one of the 12 points, so all
points are internally comparable. Three cells differ from the values
SETUP.md quotes, all of them changes that landed between SETUP.md being
written and this sweep starting:

| cell @ backend | SETUP.md | frozen | why |
|---|---|---|---|
| `mlp_control/mlp_control_full@cpu` | 28.5 µs | **107.3 µs** (+276%) | correctness fix: QnnCpu's int8 path returns a constant for this network; the CPU tile is now the fp32 context |
| `mlp_control/mlp_control_full@dsp` | 404 µs | **589.0 µs** (+46%) | a `feedback` promotion from an earlier ViNT run |
| `vint/vint_decoder@cpu` | 12694 µs | **37826.0 µs** (+198%) | a `feedback` promotion from the same run |

Everything else — dronet, yolov8n, all three fused_split tiles, ViNT's
encoders — matches SETUP.md to the digit.

The third row matters for question 3's framing. SETUP.md computes ViNT's
"whole split critical path" as ~25 ms from the 12.694 ms decoder cell; with
the promoted 37.826 ms cell it is 52.0 ms. **Both readings support the same
conclusion** — 26.9 ms would make ViNT *smaller* than yolov8n's 28.6 ms and
52.0 ms makes it 1.8x — and the measured durations in this
sweep put ViNT's actual in-situ critical path at 16.08 + 22.59 = **38.7 ms**
against yolov8n's measured 13.19 + 17.47 = **30.7 ms**, i.e. **1.26x**. On no
reading — SETUP.md's cells, the frozen cells, or the measurement — is it 100x
anything.

The live `measurements/qrb5165_v66.json` changed during this sweep — it was
sha256 `b4f1a684…` when frozen and `ac1131bc…` when the last rep finished.
Another tenant appended GPU cells for the ViNT tiles, three
`vint_obs_batch` cells, four compose failures and four notes, and
re-promoted `vint_par/vint_decoder` from 14.4 to 28.5 ms. **None of the
changed or added cells is one this sweep uses**: no `gpu` lane exists in this registry's slot map, and
`vint_par` is a different binding manifest. Verified by diffing the frozen
snapshot against the live file after the last run. The frozen sha and the
per-point verification are in `results.json` under `provenance`.

---

## 7. Provenance

SETUP.md lists five gates. Four hold; one could not be read, and what
replaces it is stronger. `verify_provenance.py` reports **zero mismatches
across every point, every rep and every column**.

1. **`dispatch_table.h` sha256 recorded with the point** — yes, in
   `results.json`, alongside `runtime_main.cpp`'s.
2. **`[summary] N/N entries executed` matches the dispatch count** — yes,
   36/36 runs, no skipped entry anywhere.
3. **Every `[bringup]` line's context filename matches the manifest** —
   **NOT AVAILABLE, and not because of anything this sweep did.**
   `qnn_models/runtime/deploy_and_run.sh` takes the board lock with
   `exec {lockfd}> /tmp/qnn_board.lock 2>/dev/null`, and an `exec` with no
   command applies its redirections to the shell for the rest of the script
   — so every byte the runtime writes to stderr on the board, `[bringup]`
   and `[main]` included, is discarded before ssh can carry it home. No
   `run.log` produced through that script has a `[bringup]` line. What
   replaces it: predicate 6 checked every context the table names was on the
   board *before* each run; the runtime *skips* entries whose context is
   missing, so `N/N` is itself proof none was; and gate 4 reads the context
   actually used per entry rather than per context.
4. **The trace's per-entry `ctx` column matches the intended placement** —
   yes. For all 36 runs, every trace row's `network`, `name`, `core_kind`,
   `ctx`, `graph`, `predicted_start_ms` and `predicted_duration_ms` are
   identical to the emitted `dispatch_table.h` row of the same `entry_id`,
   and the table's contexts are identical to the binding manifests'. That is
   a row-for-row identity between manifest, emitted table and executed
   trace, over all **1,074** executed trace rows in the 36 runs: 0
   mismatches.
5. **Board build gated on rc=0 with a fresh binary** — `deploy_and_run.sh`
   compiles under `set -euo pipefail` and aborts the run on a non-zero g++;
   sources are re-scp'd and rebuilt for every rep, and the table sha plus
   gate 4 close the stale-binary path that gate 3 would have covered.

Two further checks, from SETUP.md's own list of target-specific predicates:

* **Predicate 6** (context staged before the run): passed for all 12 points,
  6–13 contexts each. `ctx_mlp_control_fp32__Cpu.bin` was the one context
  not yet linked into `/root/qnn_runtime_ctx`; `flow_c.py stage` linked it
  from `/root/repro_perlane`.
* **Predicate 7** (no capability-excluded sentinel in the chosen placement):
  passed for all 12 points — no scheduled `(tile, lane)` pair resolves to a
  cell the cost model derived rather than measured.

---

## 8. Conditions, and what they cost

* **`--tuned` on every rep**: performance governor on all 8 cores, one
  warm-up walk (`FLOWC_ITERATIONS=2`, the reported trace is walk 2),
  governor restored to `schedutil` afterwards. The board was left on
  `schedutil` and with no sweep processes or directories at the end.
* **Board sharing**: two other agents were on 10.44.120.201 throughout.
  Every rep records its lock wait; all 36 were 0.28–0.80 s, i.e. the board
  was never held against this sweep for anything close to a rep's runtime,
  and no point met SETUP.md's re-run condition. Points ran strictly
  serially. The board never became unreachable, so no relay power cycle was
  needed.
* **One unavoidable cross-tenant effect**: `flow_c.py run --tuned` restores
  `schedutil` unconditionally after each rep. If another tenant had set
  `performance` for their own run, this sweep took it away from them 36
  times. That is flow_c's behaviour, not something the driver added, but it
  is a real interaction and worth fixing upstream (save and restore the
  governor that was there, rather than assuming `schedutil`).
* **One board interaction outside the lock**: `flow_c.py stage` opens its
  own ssh and the driver cannot wrap it (called 3 times, once per arm). It
  creates symlinks and runs no compute, so it cannot perturb a measurement.
  Everything that executes on the board — the predicate-6 probes, the lock
  probes, the runs — went through
  `ssh ... "flock -w 900 /tmp/qnn_board.lock -c '...'"` inside
  `timeout -s KILL`, or through `deploy_and_run.sh`'s own flock.

---

## 9. What would have to change to make this sweep stronger

1. **A per-dispatch overhead term in the cost model.** Adding the measured
   0.367 ms DSP / 0.540 ms HTA / 0.0026 ms CPU dispatch cost to each cell
   would take the three sub-millisecond DSP tiles from ~1.88x to ~1.0x and
   remove most of the 1.165x DSP-lane error, without a single new
   measurement — the numbers are already in `measurements/qrb5165_v66.json`.
2. **A contention term for the CPU lane** (or actually binding the QNN CPU
   backend's thread pool). Until then the solver will keep putting
   50 ms CPU tiles in front of a 500 Hz control loop, and `feedback` will
   keep oscillating instead of converging.
3. **More seeds.** 5 matched pairs, of which 2 are MILP incumbents, is thin
   for a 4–10% effect against a 4.4% rep-to-rep spread. The generator and
   driver here run unattended; the binding constraint is MILP time, so
   either a longer budget or `greedy_periodic` across the board (which would
   also make the arms comparable on equal solver footing) buys more points.
4. **A ViNT-heavy arm.** To test the FPGA's claim on its own terms, ViNT has
   to be the dominant load — `count {4,5}` for ViNT against `{0,1}` for
   yolov8n — rather than one instance against four or five detections.

---

## 10. Files

    SETUP.md                       the contract, written before the run
    banks/                         hardware + model bank for the ported generator
    gen_random_workload.py         ported from RoSE verbatim except --banks-dir
    sweep_unbounded_nonperiodic.py ported; arms, 3-lane hardware, predicate 5,
                                   and the per-point Flow C spec
    drive.py                       artifacts/solve/runtime/stage/run/results
    verify_provenance.py           the gates in section 7
    analyse.py, analyse2.py        the tables in sections 1-4
    cost_model.json                the frozen cells every point solved against
    bank_spec.json                 the Flow C spec used to re-emit the artifacts
    workloads/<point>.json         xpu-rt taskset per point
    workloads/<point>.flowc.json   Flow C spec per point (one entry per copy)
    workloads/rejected/            the 6 rejected points + their reason
    schedules/scheduled_*.json     one per point, solver recorded in results.json
    runtimes/<point>/              dispatch_table.h + runtime_main.cpp, sha256'd
    runs/<point>/rep{1,2,3}/       run.log (trace block) + trace.csv + driver.log
    logs/                          generation, artifacts, solve, runtime, stage
    plots/<point>_profiled.png     the scheduler's predicted timeline
    plots/predicted_vs_actual/     predicted-vs-actual gantt, full and zoomed
    generated.json                 generation + validation record for all 18
    provenance.json                per-point, per-rep gate results
    state.json                     the driver's own per-stage record (raw)
    logs/analysis_tables.txt       the full output of analyse.py + analyse2.py
    results.json                   the sweep

This sweep wrote nothing outside this directory except the artifacts the
flow itself emits and consumes: the coarse dispatch graphs under
`gen/qnn_vmfb/<net>/qrb5165_flowc/` and the profile CSVs under
`gen/profile/<HW>/qrb5165_flowc/`, re-emitted from the frozen cost model by
`flow_c.py artifacts` and byte-identical to what is already committed. The
`data/toplevel/networks_<point>.json` files `flow_c.py artifacts` writes as
a side effect were unused (the solver reads this directory's `workloads/`)
and were removed. `qnn_models/flow_c/measurements/qrb5165_v66.json` was read
and never written. On the board, the 12 `/root/flowc_sweepc_*` build
directories were removed and the governor was left on `schedutil`.

---

## Addendum: the six rejected periodic-only points

The six points rejected on `no non-periodic work` were subsequently built and
run. Full write-up and provenance: `addendum_periodic_only/README.md`;
tables from `analyse_addendum.py`. The sweep's own pre-registered record is
untouched -- the six still carry `status: REJECTED` in `generated.json` and
do not appear in `results.json`. The addendum reuses this sweep's frozen
`cost_model.json` and lands in its own tree via `drive.py`'s `SWEEP_OUT`.

**Why they were worth running.** Q1 came back weak here for a structural
reason: in all 12 accepted points a `yolov8n_head` finishes last, so the
mid-size periodic slot cannot move the makespan by more than second order.
These six have no yolov8n, so periodic work owns the makespan.

**Q1 reverses.** 18 runs, 6 points x 3 reps, all clean:

    seed  pred b  pred f    d%  |   act b   act f     d%  | noise b/f %  verdict
    4      20.11   20.11   0.0  |   20.04   20.46    2.1  |  0.0/0.1     dronet wins
    5      48.11   48.11   0.0  |   48.04   48.05    0.0  |  0.0/2.0     within noise
    7      16.64   16.81   1.0  |   18.19   18.97    4.3  |  0.4/1.1     dronet wins

Two of three pairs favour **dronet** by margins well outside rep noise. That
is the opposite direction from the main sweep (fused better in 3 of 5,
weakly) and from the FPGA (4 of 5).

The mechanism is that `fused_split` is *more work*, not slower work:
`fused_seed4` executes 13.903 ms of busy time against its baseline's 4.887 ms
inside the same 20 ms release window. So Q1 is really "does the extra fused
work fit in the schedule's slack", and the answer depends on what sets the
makespan -- release-gated (hides it, 0.0-2.1%), compute-bound (exposes it,
4.3%), or masked behind a yolov8n_head as in all 12 accepted points.

**Combined reading across all three regimes: fused_full's FPGA advantage does
not reproduce on this target, and where it is visible at all it is a cost.**

**A clean decomposition of the 1.17x makespan ratio.** In the periodic-only
regime the median ratio is **1.010x**. Three points land at exactly 1.000 --
those whose makespan is release-gated, where the runtime hits its wall-clock
release times to the microsecond. The error appears only in the compute-bound
points: 1.090x and 1.130x, the same magnitude as the main sweep's per-lane
ratios (hta 1.092, dsp 1.165, cpu 1.314). So the 1.17x is not a uniform
modelling bias -- it is concentrated entirely in compute-bound portions of the
schedule, and the release-gated portion contributes no error at all.

**A scheduler bug this exposed.** The rejection predicate guarded a real
limitation. Three places in the XPU-RT stack degrade when a workload has zero
non-periodic operations, and one is a bug, now fixed:

  * `xpu-rt/scheduler.py:510` -- `C_max` is unconstrained from below, so the
    MILP objective is trivial and MOSEK raises `SolverError`. Worked around
    per-point by setting `restrict_makespan_to_nonperiodic: false`, recorded
    in each addendum workload's `_comment`. (Predicted makespans in the
    addendum are therefore not comparable to this sweep's; the Q1 comparison
    is internal to the addendum and unaffected.)
  * `xpu-rt/postprocessing.py:312` -- trim no-ops. Benign, already handled.
  * `xpu-rt/workload_factory.py:415` -- **fixed.** It collapsed every periodic
    network to 1 instance while ignoring an explicit `num_instances`, so a
    12-operation workload was silently scheduled as 3. The override at
    `workload_factory.py:460` exists precisely so a toplevel JSON can pin
    instance counts; the early return jumped over it. The fix executes only
    when there are no non-periodic ops, which no accepted point in this sweep
    has. Regression check: `baseline_seed0` re-solves to 20 operations,
    `optimal`, **67.571 ms** -- identical to its pre-registered value.
