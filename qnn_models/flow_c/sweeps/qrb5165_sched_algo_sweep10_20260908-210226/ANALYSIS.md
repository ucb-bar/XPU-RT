# sched_algo_sweep10 on real silicon — QRB5165 / Flow C — RESULTS

Written after the run. `SETUP.md` in this directory is the contract, written
before it; **nothing here revises it**. Every number below is measured, or is a
prediction a measurement is compared against. Where the two disagree, the text
names which of SETUP.md's documented confounds accounts for it, or says it does
not know.

Medians are over 3 reps and the spread is quoted with them. **No single-rep
number appears as a result.**

---

## 0. What ran, and what did not

```
Phase 1   16 networks built whole, 5 (backend, precision) composes each   80
          + 12 rewritten variants, same 5 composes each                   60
          cells: 141 (tile, lane) pairs x 5 independent passes           705
Phase 2   11 families x 4 lane configs                                    44
          generated and kept                                             42
Phase 3   42 cells x 11 executed solvers (+ best-of-fast synthesised)    462
Phase 4   199 planned points -> 97 unique schedules by content hash
          291 board runs (97 x 3 reps), 102 points attributed by dedupe
Phase 5   12 cells with ALL twelve solvers measured; 42 cells with the
          feasible-first winner and greedy measured
```

**Everything planned in SETUP.md ran.** Two things it named as possible are
worth stating explicitly because they did not turn out to be limits:

* **CP-SAT was not blocked.** The brief warned that no interpreter on this host
  had `ortools` and that the four CP-SAT entries might have to be dropped.
  A venv was created (`ortools 9.15.6755`), so **all twelve solver entries ran**
  and none of the reference's conclusions is untestable for want of a solver.
* **`yolov8_nano_sh` did not time out.** Its `extract_graph` pass took 1998 s
  against a 3600 s ceiling, so `bimodal` — the only family that uses it —
  survived.

What did NOT run, and why:

* **The shard arm.** `wl_sweep_shard` shards operators across two harts, which
  shows up as more dispatches. This target has no such primitive: the work
  inside a QNN dispatch belongs to HVX, the tensor accelerator or the CPU op
  package, and the host cannot subdivide it. So the reference's shard-specific
  findings — cold `cpsat` falling from +10.18% to +1.87% at 801 ops, and
  `cpsat:warmbest` holding +8.42% there — **cannot be re-tested here at all.**
  That is a real loss of half the reference's coverage.
* **Two cells, `vint_intro_hd` and `vint_multi_hd`,** rejected on predicate 5:
  ViNT's decoder has neither an HTA nor a DSP cell, so in a two-accelerator
  config the solver would only ever see exclusion costs for it.
* **Cells beyond the 12 all-solver ones** carry only the winner and greedy.
  30 of the 42 cells therefore contribute to the coverage claim but not to the
  ranking claim.

Board discipline held: **597 rep records, all `N/N entries executed`**, zero
skipped entries, zero predicate-6 or predicate-7 violations, and a board lock
wait of 0.26 s median / 0.74 s max — never remotely near a rep's own runtime,
so no point needed re-running on the contention rule.

---

## 1. Does the reference's ranking survive measurement?

**The grouping survives. The ordering inside the top group does not, and this
board cannot resolve it either way.**

| solver | measured mean vs greedy | median | worst | cells | predicted here | reference (FireSim, predicted) |
|---|---|---|---|---|---|---|
| **cpsat:warmbest** | **+41.69%** | +45.78% | +3.09% | 12 | +38.78% | **+9.75%** (rank 1) |
| **sa** | **+41.13%** | +41.42% | +0.57% | 12 | +38.78% | +8.75% (rank 3) |
| **best-of-fast** | +36.99% | +41.02% | +0.00% | 13 | +34.98% | +7.63% (rank 5) |
| cpsat:warm | +36.33% | +45.78% | **−61.28%** | 12 | +38.78% | +8.30% (rank 4) |
| cpsat | +35.95% | +42.12% | **−61.28%** | 12 | +38.78% | +6.03% (rank 6) |
| pso | +30.26% | +40.54% | −42.85% | 15 | +32.21% | **+9.10%** (rank 2) |
| greedy_reserved | +17.71% | 0.00% | +0.00% | 12 | +14.74% | −0.10% |
| greedy_periodic | +16.82% | 0.00% | +0.00% | 14 | +13.42% | +1.04% |
| heft_edf | +10.58% | +31.92% | **−301.42%** | 14 | +30.64% | +2.37% |
| heft | +3.75% | +21.20% | **−301.42%** | 14 | +23.24% | −1.66% |
| **decomposed** | **+0.01%** | 0.00% | −2.94% | 25 | 0.00% | **−4.92%** |
| greedy | 0 | 0 | 0 | 42 | 0 | 0 |

What reproduces:

1. **`cpsat:warmbest` is first in both.** It is also the only entry in the top
   group whose worst case is positive (+3.09%), and the reason is exactly the
   one the reference gives for inventing it: `cpsat` and `cpsat:warm` both
   carry a **−61.28%** worst case here, and `cpsat:warm`'s hint *is* `heft_edf`,
   whose own worst case is −301%. Hinting from the best feasible heuristic
   instead removes that failure mode. **That specific mechanism is confirmed on
   hardware.**
2. **The top six are the same six**, in both studies: `cpsat:warmbest`, `sa`,
   `cpsat:warm`, `cpsat`, `best-of-fast`, `pso`. Their internal order differs.
3. **`decomposed` is at the bottom, at or below greedy, in both.** The
   reference's most interesting negative result — that the four-solver sweep's
   "decomposed wins, +2.18%" became −4.92% once six more solvers were in — is
   directionally confirmed by measurement.
4. **`heft` and `heft_edf` carry the catastrophic tail in both.** The
   reference's cautionary case is not just preserved, it is worse: −301% on
   `scale_ladder_quad`, against the reference's −119%.

What does not:

5. **`pso` falls from rank 2 to rank 6.** Its measured mean (+30.26%) is the
   lowest of the top six, and its worst case (−42.85%) is negative where the
   reference reports pso as greedy-dominating by construction (+0.00% worst
   over 78 workload-arms). The mechanism is not mysterious — `pso` is seeded
   from the heuristics and only accepts improvements *in the cost model*, so
   its floor is a floor on the prediction, not on the measurement. On this
   board the cost model is not exact, so the structural guarantee does not
   transfer. **A guarantee that holds in a cost model does not hold on
   silicon.**
6. **`greedy_reserved` and `greedy_periodic` rank far higher here** (+17.7%,
   +16.8%) than in the reference (−0.10%, +1.04%). In the reference their means
   were computed over only the ~56 of 78 arms where they stayed feasible; here
   they stay feasible on essentially everything, because the periods were
   re-derived from this board's measured cells rather than transported.
7. **The magnitudes are 4x the reference's.** The top group is +30 to +42% here
   against +6 to +10% there. That is a property of the target, not of the
   solvers: this board's four lanes span 0.06 ms to 84 ms on the same tile,
   where FireSim's two harts differ by well under 2x. **Do not read the
   magnitudes across targets** — SETUP.md forbids it and this is why.

### The finding that matters most: most of this ranking is unresolvable here

Held against the measured rep spread of the two points being compared:

> **476 of 792 solver pairs (60%) are separated by less than the noise, and
> within the reference's top six it is 167 of 180 — 93%.**

So the honest statement is: **the reference ranks six solvers that this
hardware cannot tell apart.** What measurement *does* resolve is the grouping —
top six vs {greedy_reserved, greedy_periodic} vs {heft, heft_edf} vs
{decomposed, greedy} — and the tails, which is where the deployment advice
actually lives.

Kendall tau between the predicted and measured orderings over the 12
all-solver cells: **median +0.742**, min +0.364, max +1.000. The cost model
gets the broad order right and the fine order wrong.

---

## 2. The noise floor, stated before it is used

| quantity | median | p90 | max |
|---|---|---|---|
| rep spread, non-periodic makespan | **9.18%** | 37.9% | 159.6% |
| rep spread, wall clock | 7.62% | — | 159.6% |
| rep spread excluding `scale_ladder` | 6.94% | 29.2% | 61.8% |

The precedent QRB5165 sweep measured ~4.4%. This one is roughly twice that, and
the reason is in the matrix: this sweep deliberately includes sub-millisecond
accelerator tiles, and those are where the board is least repeatable.

**SETUP.md predicted this and predicted its shape.** It said the accelerator
cells are bimodal, that the median cell would not carry the collapsed cost, and
therefore that "measured > predicted should have a heavy right tail rather than
being symmetric noise". That is what happened:

| | measured / predicted (non-periodic) |
|---|---|
| all 199 points | median **1.025x**, min 0.796, max **4.874x** |
| excluding `scale_ladder` | median **1.030x**, min 0.796, max **1.668x** |

Per family:

| family | ratio median | ratio max | rep spread median | rep spread max |
|---|---|---|---|---|
| `depth_nav` | 0.998 | 1.050 | 0.03% | 6.5% |
| `tight_loop` | 1.039 | 1.051 | 0.23% | 1.5% |
| `bimodal` | 1.013 | 1.269 | 2.34% | 16.2% |
| `vint_multi` | 1.010 | 1.040 | 14.07% | 22.4% |
| `vint_intro` | 0.995 | 1.050 | 7.43% | 26.5% |
| `saturation` | 1.000 | 1.242 | 22.71% | 59.9% |
| `depth_contended` | 1.103 | 1.593 | 6.94% | 35.6% |
| `depth_chain` | 1.087 | 1.165 | 12.55% | 24.2% |
| `control_mix` | 1.514 | 1.668 | 7.12% | 37.9% |
| `perception_heavy` | 0.956 | 1.647 | 29.21% | 61.8% |
| **`scale_ladder`** | **1.639** | **4.874** | 15.19% | **159.6%** |

`scale_ladder` is the whole tail. It is six `dronet` rungs, all non-periodic,
every offered cell between **0.592 ms** (`dronet_sb@dsp`) and **2.788 ms**
(`dronet_se@hta`) — i.e. entirely inside the regime where the DSP and HTA
power-collapse between invocations. Its worst point,
`scale_ladder_quad__heft`, predicts 2.61 ms and measures 12.74 ms with reps of
[12.741, 14.618, 6.215]: **the rep-to-rep spread is 2.3x the whole prediction.**
The cost model recorded the same bimodality when the cells were taken —
`dronet_sg@dsp` measured [2858, 1027, 882, 820, 841] µs over five passes — which
is why the cells are medians over five passes rather than single measurements.

### An unplanned independent check: the same schedule, measured twice

The content-hash dedupe is conservative — it can keep two points that would
execute identically, never the reverse. It did: **9 dispatch tables came out
byte-identical from two different (cell, solver) points** and were therefore
measured in two separate board sessions, with separate builds and separate
stagings.

| dispatch table | point A | A | point B | B | diff |
|---|---|---|---|---|---|
| `447a182c` | `bimodal_dc__greedy` | 6.932 | `bimodal_quad__greedy` | 6.920 | 0.17% |
| `659492df` | `bimodal_cg__cpsat-warm` | 20.947 | `bimodal_cg__heft` | 21.100 | 0.73% |
| `530bfe13` | `control_mix_quad__cpsat-warm` | 3.459 | `control_mix_quad__greedy_periodic` | 3.492 | 0.95% |
| `8dd82d36` | `control_mix_dc__cpsat` | 3.546 | `control_mix_dc__greedy_periodic` | 3.501 | 1.27% |
| `13c456ce` | `perception_heavy_dc__greedy` | 6.430 | `perception_heavy_quad__greedy` | 6.564 | 2.04% |
| `551ce3f5` | `saturation_quad__cpsat-warm` | 4.527 | `saturation_quad__heft` | 4.274 | 5.59% |
| `d1dc0b85` | `vint_intro_dc__cpsat` | 43.664 | `vint_intro_dc__heft` | 48.078 | 9.18% |
| `538305d9` | `saturation_dc__cpsat-warm` | 6.379 | `saturation_dc__heft_edf` | 5.707 | 10.53% |
| `6d32520a` | `vint_multi_dc__cpsat-warm` | 44.923 | `vint_multi_dc__heft` | 52.304 | 14.11% |

**Between-session reproducibility of an identical dispatch table: median
2.04%, max 14.11%.** That is *tighter* than the within-session rep spread
(9.18% median), which says the rep-to-rep variation is dominated by what the
board does between invocations rather than by drift between sessions — the
same power-collapse mechanism, not a systematic bias.

The two worst rows are `vint_multi_dc` and `vint_intro_dc`, both of which put
ViNT's decoder on the CPU lane. That is SETUP.md's first documented confound —
CPU cells are load-dependent and the cost model cannot express it — showing up
in the reproducibility rather than in the mean.

**Conclusion for the cost model: it is accurate.** Excluding the one family
that is entirely inside the collapse regime, the median cell predicts the
measured non-periodic makespan to **3.0%**, over 121 (cell, solver) points and
four lane configurations. That is a better result than the study needed.

---

## 3. `decomposed`: the reference's reversal, re-examined

The reference's headline negative is that `decomposed` is **−4.92%** against
greedy and 9th of 12. Measured here it is **+0.01%** — that is, it is greedy.

Of the 25 measured `decomposed` points, **22 tie greedy exactly** and its worst
case is −2.94%. The reference reports the same tying behaviour (a bit-identical
objective to greedy on 43 of 80 workload-arms) but has its mean dragged to
−4.92% by a single family, `depth_contended`, where it is 46.6% worse.

**That family does not reproduce the effect here.** On this board
`depth_contended` is a 5-operation problem — fastdepth x2, dronet_sf x2,
yolov8_nano_sc x1, one tile each — where the reference's is 126+ operations of
sharded kernels. `decomposed` is a greedy variant; with five operations there
is nothing for it to decompose badly.

So: **the sign reproduces and the magnitude does not, and the reason is that
the dispatch space on this target is 5–33 tiles where the reference's is
126–801 operations.** That is a structural difference between the targets, not
a disagreement about the solver. It is the same reason the reference's
`depth_contended` outlier has no analogue here.

---

## 4. Where predicted and measured disagree, and why

Three disagreements are large enough to name, and each has a documented cause.

**(a) `control_mix`, ratio 1.514.** The worst systematic over-run outside
`scale_ladder`. `control_mix` is `mlp_control_sd` x8 + `dronet_se` x4 +
`yolov8_nano_sc`, and in the `cg` config its only lanes are CPU and GPU. The
CPU lane carries eight `mlp_control` instances at a 0.115 ms cell — SETUP.md's
first confound is exactly that CPU cells are load-dependent and the cost model
cannot express it, with the shipped README documenting a 5.6x in-situ inflation
for a contended CPU tile. This is that confound, and it is the largest one in
the sweep.

**(b) `scale_ladder`, ratio up to 4.874.** Accelerator power-collapse on
sub-millisecond tiles, as above. Predicted by SETUP.md, and visible in the cost
model's own pass-to-pass spreads.

**(c) `perception_heavy`, ratio median 0.956 — measured FASTER than
predicted.** Its rep spread is 29% median / 62% max, so this is inside its own
noise and no claim is made about it. Naming it anyway because a ratio below 1
is the shape a reader should be suspicious of.

Everything else sits between 0.995 and 1.103 at the median.

---

## 5. Phase 1 was the part that decided whether this experiment was worth running

This is not the question the sweep was commissioned to answer, but it is the
finding with the widest consequence, so it goes here rather than in a footnote.

Phase 1 built all 16 networks whole and found **HTA rejected 15 of 16**, on
three blockers (`Batchnorm`, `Elu`, `StridedSlice`). Recording those and
proceeding would have produced a complete, internally consistent, and
**materially wrong** experiment: 12 of 18 tiles would have reached the solver
with one fewer placement than the silicon supports, on a board whose whole
interest is lane heterogeneity.

Removing them instead — three numerics-preserving graph rewrites, none of which
existed in the tree — changed the cost model this much:

| | before rewrites | after |
|---|---|---|
| HTA cells | 1 of 18 | **12 of 18** |
| DSP cell, `yolov8_nano_sh` | 11.467 ms | **8.691 ms** (−24%) |
| DSP cell, `dronet_sg` | 1.077 ms | **0.882 ms** (−18%) |
| CPU cell, `yolov8_nano_sh` | 60.295 ms (fp32) | **16.627 ms** (int8, −72%) |
| lane configs the axis can offer | 3 (no HTA) | **4, including a true `quad`** |

Two of those rows are not about HTA at all. Removing `Batchnorm` and
`StridedSlice` made the **DSP** 3–24% faster on the same networks — the lane
that already ran them. And the per-tile precision decision, which the slicing
study insists on and which this sweep re-measured on its own networks, moved
`yolov8_nano_sh`'s CPU cell by 3.6x. A blanket per-flow precision choice would
have mispriced the largest tile in the sweep, in the lane the scheduler is most
likely to overflow into.

The full compose-failure catalogue, the per-lane before/after that justified
each adoption, the rewrites that were tried and lost, and the granularity
decision per rung are in **`results/REWRITE_LEDGER.md`**, generated from the
committed records.

Two negative results in there are worth surfacing:

* **The `Elu` -> `Relu` probe was measured and refused.** It is the only way
  `mlp_control` reaches HTA and it is numerics-**changing**. It measures
  2.186 ms on HTA against 0.176 ms on the CPU — 12x worse. The decision to
  accept CPU/DSP/GPU placement for the control loop is therefore measured, not
  argued, and nothing from that variant entered the cost model.
* **Every network keeps one tile.** Against the slicing study's break-even rule
  (`0.37 ms + 5.4 ns x boundary_bytes` per extra DSP dispatch, boundary bytes
  counted over every crossing tensor and read from each rung's own ONNX), no
  cut pays — on `dronet` and `mlp_control` because the added dispatch alone
  exceeds the whole network, and on `yolov8_nano` and `fastdepth` because the
  rewrites already made every lane reachable whole, so a cut buys no placement
  it does not already have. `vint` keeps its shipped two-tile split, where the
  cut genuinely does buy placement.

---

## 6. A defect this pipeline caught, recorded because it nearly did not

The frozen cost model initially carried `vint/vint_encoders@gpu`, a cell the
shipped binding manifest has no context for. The scheduler has no "forbidden"
flag — it placed the encoders on the GPU in two cells — and `flowc/schedule.py`
then refused to emit a runtime for them. **The failure was loud and early only
because predicate 6 and 7 exist**; without that check the two cells would have
been silently dropped from Phase 4 and the coverage claim would have been
wrong.

`build_cost_model.py` now drops any cell whose manifest declares no context,
records what it dropped, and the two cells were regenerated and re-solved from
the corrected model. Every number in this document is from the corrected model. `cost_model.json`
records the sha256 of its own inputs, `verify_provenance.py` checks them, and
`results/phase4_results.json` records the emitted `dispatch_table.h` sha256 per
point so the executed schedule is tied to the predicted one.

---

## 7. Honest caveats

1. **Half the reference's coverage is missing.** One arm, not two. The shard
   arm does not port, for a structural reason, and every shard-specific finding
   in the reference is untested here.
2. **The dispatch space is 4–33 tiles, against the reference's 126–801
   operations.** This is the single biggest difference between the two studies
   and it explains both the `decomposed` magnitude gap and why so many cells
   have no scheduling freedom at all. **19 of 42 cells return a single
   distinct feasible objective** — every solver ties. 12 of those 19 are the
   cells in which every operation is periodic, where the objective degenerates
   to "last release + critical path" and there is nothing to trade against;
   the other 7 tie for the ordinary reason that the problem is small enough
   that greedy already finds the optimum.
3. **60% of pairwise solver comparisons are inside the noise, 93% within the
   top six.** Any statement in this document about the order of two solvers
   inside the top group is not supported by the measurement.
4. **The magnitudes are not comparable with the reference's.** +30–42% here
   against +6–10% there is a statement about lane heterogeneity on this board,
   not about the solvers.
5. **`best-of-fast` is scored on the same runs as its members**, as in the
   reference. Its wall time is the honest sum of all six.
6. **The four CP-SAT entries are not bit-reproducible** — `num_search_workers=8`
   makes CP-SAT non-deterministic, which the reference measured at 2.73% mean
   spread cold. `reproduce.py` checks the eight deterministic solvers exactly
   and says so.
7. **The scaled rungs are latency targets, never accuracy results.** Non-default
   `dronet` and `mlp_control` rungs keep seeded random init and `fastdepth` has
   no checkpoint at all. `yolov8_nano` rungs are the exception and do load COCO
   weights.
8. **Cross-family absolute comparisons are unsound**, as in the reference. The
   valid comparison is within a workload cell, where all twelve solvers see
   identical durations.

---

## 8. What this answers, in one paragraph

The reference sweep's headline is a ranking of twelve solvers by predicted
makespan, never executed. Executed on a QRB5165, **its grouping holds and its
ordering does not**: the same six solvers lead, `cpsat:warmbest` leads them in
both, `decomposed` sits at greedy in both, `heft` carries the worst tail in
both — but 93% of the pairwise orderings inside that leading group are smaller
than this board's rep-to-rep spread, and `pso`'s structural "never worse than
greedy" guarantee, which holds by construction in the cost model, does not
survive contact with the hardware. The cost model itself is good: outside the
one family that lives entirely inside the accelerators' power-collapse regime,
it predicts the measured makespan to 3.0% at the median over 121 points. The
largest single risk to an experiment like this turned out not to be the
scheduler or the noise floor at all, but Phase 1: fifteen of sixteen networks
could not reach the board's fastest lane as exported, and had that been
recorded rather than fixed, the study would have measured a scheduling problem
one lane narrower than the silicon actually offers.
