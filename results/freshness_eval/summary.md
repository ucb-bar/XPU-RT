# Decision Gate A — does local deadline success hide stale outputs?

Workload `data/toplevel/freshness_canon_300ms.json`, epoch 300 ms. Producer `dronet` T=50 ms L=17.973 ms; consumer `mlp_control` T=10 ms L=0.546 ms (on `gemmini`).

**A0 = 60.546 ms** is the measured uncontended input-age ceiling; the uncontended age set is `[20.55, 30.55, 40.55, 50.55, 60.55]` ms. Windows are `phi = A0 + delta`, so every point has real headroom over the sampling rate.

## Restatement history

These numbers were restated once, on 2026-08-03, after the evaluator gained an inclusive-boundary tolerance. The schedules did not change: all 100 cells were re-evaluated from the cached fixtures with byte-identical makespans, so the difference is entirely evaluator-side.

**23 of 100 cells moved, every one of them upward, by exactly one consumer invocation (two in one cell).** An input age is compared against phi with a relative tolerance (`BOUNDARY_RTOL = 1e-9`) so that an age landing exactly on the window counts as valid. Such exact hits are systematic here rather than incidental: both A0 and phi carry the consumer's 0.546074 ms latency term, so an age measured at output lands precisely on phi whenever delta is a multiple of the 10 ms consumer period. The affected records compute to e.g. 70.54607400000002 against a phi of 70.546074 -- strictly larger in floating point, and valid only because of the tolerance.

The correction makes the reported divergence *smaller* (largest single change +0.067, typical +0.033). It works against the headline rather than for it, and no qualitative conclusion changes.

## Timing assumptions

- source: **firesim_measured**, ModelBlaster/benchmarks/profile_db (see gen/profile/**/_provenance.json)
- backends: {'cpu_p': 'gemmini', 'cpu_e': 'rvv_opu'}, target `firesim_gemmini_opu`
- derivation: mean_time_ms = cycles / 1e6 at an assumed 1000.0 MHz
- The assumed 1 GHz is NOT the Alveo U250 bitstream frequency (25-30 MHz). Raw cycles are preserved in the exported results.csv `cycles` column. Absolute millisecond claims must restate this assumption.
- producer instance attribution: inferred_from_schedule_timestamps -- no dataflow exists between the producer and consumer networks in either XPU-RT or ModelBlaster, so the consumed instance is inferred, never recorded
- post-passes: compaction=False, automerge=False

## Validity vs contention at phi = 80.5 ms (= A0 + 20)

`output-valid` splits into two failures that are NOT the same claim, so both are broken out. **stale** means the consumer acted on an input that was too old — the phenomenon this study is about. **no-input** means there was no completed producer output at all, which a real controller handles by holding or faulting rather than by actuating on garbage. Reporting only their sum overstates the stale-input finding; at B=1 below it does so by about 5x.

A **!** marks a cell whose schedule OVERRUNS the epoch. Greedy then extends the horizon and adds instances, so that row's rates are computed over a longer trace with a different denominator (41 consumer invocations at B=3, 64 at B=4, against 30) and are **not comparable** to a fitting row. Such cells are excluded from every headline below.

| policy | B | consumer deadline-valid | freshness-valid | output-valid | stale | no-input | max input age | producer deadline-valid | producer max late | soft done |
|---|---|---|---|---|---|---|---|---|---|---|
| static_nominal | 0 | 1.000 | 0.933 | 0.933 | 0.000 | 0.067 | 60.5 ms | 1.000 | -32.8 ms | 0 |
| static_nominal | 1 | 1.000 | 0.633 | 0.633 | 0.067 | 0.300 | 100.5 ms | 0.667 | +37.2 ms | 1 |
| static_nominal | 2 | 1.000 | 0.400 | 0.400 | 0.300 | 0.300 | 130.5 ms | 0.333 | +37.2 ms | 2 |
| static_nominal | 3 ! | 1.000 | 0.220 | 0.220 | 0.561 | 0.220 | 130.5 ms | 0.000 | +37.2 ms | 3 |
| static_nominal | 4 ! | 1.000 | 0.000 | 0.000 | 0.750 | 0.250 | 500.5 ms | 0.000 | +433.3 ms | 4 |
| edf | 0 ! | 0.300 | 0.100 | 0.067 | 0.700 | 0.200 | 215.1 ms | 0.000 | +138.1 ms | 0 |
| edf | 1 ! | 0.300 | 0.100 | 0.067 | 0.700 | 0.200 | 215.1 ms | 0.000 | +138.1 ms | 0 |
| edf | 2 ! | 0.300 | 0.100 | 0.067 | 0.700 | 0.200 | 215.1 ms | 0.000 | +138.1 ms | 0 |
| edf | 3 ! | 0.300 | 0.100 | 0.067 | 0.700 | 0.200 | 215.1 ms | 0.000 | +138.1 ms | 0 |
| edf | 4 ! | 0.300 | 0.100 | 0.067 | 0.700 | 0.200 | 215.1 ms | 0.000 | +138.1 ms | 0 |
| heft | 0 | 0.033 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 ms | 0.167 | +240.6 ms | 0 |
| heft | 1 ! | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 ms | 0.167 | +249.1 ms | 0 |
| heft | 2 ! | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 ms | 0.000 | +269.9 ms | 0 |
| heft | 3 ! | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 ms | 0.000 | +336.5 ms | 0 |
| heft | 4 ! | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0 ms | 0.000 | +384.6 ms | 0 |
| static_conservative | 0 | 1.000 | 0.900 | 0.900 | 0.000 | 0.100 | 70.5 ms | 1.000 | -28.2 ms | 0 |
| static_conservative | 1 | 1.000 | 0.600 | 0.600 | 0.067 | 0.333 | 110.5 ms | 0.667 | +41.8 ms | 1 |
| static_conservative | 2 | 1.000 | 0.333 | 0.333 | 0.333 | 0.333 | 140.5 ms | 0.333 | +41.8 ms | 2 |
| static_conservative | 3 ! | 1.000 | 0.146 | 0.146 | 0.610 | 0.244 | 140.5 ms | 0.000 | +41.8 ms | 3 |
| static_conservative | 4 ! | 1.000 | 0.000 | 0.000 | 0.770 | 0.230 | 580.6 ms | 0.000 | +505.5 ms | 4 |

## Structural floor versus contention-induced loss

B=0 is the same workload with no soft work, so validity lost there is structural — the first consumer invocations precede any producer completion however idle the machine is. The contention-induced loss is the drop below the B=0 control.

| policy | phi (ms) | output-valid at B=0 (floor) | loss at B=1 | loss at B=2 | loss at B=3 | loss at B=4 |
|---|---|---|---|---|---|---|
| static_nominal | 65.5 | 0.933 | −0.333 | −0.667 | −0.860 | −0.933 |
| static_nominal | 70.5 | 0.933 | −0.300 | −0.567 | −0.787 | −0.933 |
| static_nominal | 80.5 | 0.933 | −0.300 | −0.533 | −0.714 | −0.933 |
| static_nominal | 90.5 | 0.933 | −0.267 | −0.467 | −0.616 | −0.933 |
| static_nominal | 110.5 | 0.933 | −0.233 | −0.300 | −0.372 | −0.933 |
| edf | 65.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| edf | 70.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| edf | 80.5 | 0.067 | −0.000 | −0.000 | −0.000 | −0.000 |
| edf | 90.5 | 0.100 | −0.000 | −0.000 | −0.000 | −0.000 |
| edf | 110.5 | 0.133 | −0.000 | −0.000 | −0.000 | −0.000 |
| heft | 65.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| heft | 70.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| heft | 80.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| heft | 90.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| heft | 110.5 | 0.000 | −0.000 | −0.000 | −0.000 | −0.000 |
| static_conservative | 65.5 | 0.733 | −0.300 | −0.600 | −0.733 | −0.733 |
| static_conservative | 70.5 | 0.900 | −0.333 | −0.667 | −0.827 | −0.900 |
| static_conservative | 80.5 | 0.900 | −0.300 | −0.567 | −0.754 | −0.900 |
| static_conservative | 90.5 | 0.900 | −0.300 | −0.567 | −0.754 | −0.900 |
| static_conservative | 110.5 | 0.900 | −0.233 | −0.333 | −0.363 | −0.900 |

## Operating points where deadline success hides invalid output

Criterion: `deadline_success_rate >= 0.95` and `output_valid_rate < deadline_success_rate - 0.10`.

**41 of 100 (policy, B, phi, seed) cells qualify**, of which **21 come from a schedule that fits the epoch** and are the only ones quoted below.

1 of those is at **B=0**, i.e. with no contention at all — that part of the divergence is structural, not contention-induced. See the floor table above.

| B | qualifying cells | phi range (ms) | worst divergence | of which stale |
|---|---|---|---|---|
| 0 | 1 | 65.5–65.5 | +0.267 (static_conservative) | 0.167 |
| 1 | 10 | 65.5–110.5 | +0.567 (static_conservative) | 0.233 |
| 2 | 10 | 65.5–110.5 | +0.867 (static_conservative) | 0.533 |

Largest divergence among epoch-respecting cells: **+0.867** at `static_conservative` B=2 phi=65.5 ms — consumer deadline-valid 1.000 but output-valid 0.133, of which **0.533 acted on stale input** and 0.333 had no input at all.

For completeness and **not for quotation**: 20 flagged cells come from schedules that overrun the epoch, the largest being +1.000 at `static_nominal` B=4. That rate is measured over a 815 ms trace against a 300 ms epoch, so it describes a different experiment rather than a worse policy.

## Did the intended protection policy protect?

Mean output-valid over the cells where BOTH policies' schedules fit the epoch, against `static_nominal`. A protection policy is not protective by intention or by name; this is the check.

| policy | cells compared | mean output-valid | nominal | margin | verdict |
|---|---|---|---|---|---|
| edf | 0 | — | — | — | *no epoch-comparable cell* |
| heft | 5 | 0.000 | 0.933 | -0.933 | **WORSE than doing nothing** |
| static_conservative | 15 | 0.587 | 0.669 | -0.082 | **WORSE than doing nothing** |

`static_conservative` reserves the fast accelerator for the perception producer — the protection mechanism the specification proposed. It measures worse than the unprotected schedule at every contention level. That is the finding that forced the candidate ladder to be assembled from mechanisms measured to work rather than mechanisms expected to, and it is why `candidates.py` refuses to build a selector on unvalidated rungs.

## Sensitivity to the freshness window

output-valid rate; rows are phi, columns are B


`static_nominal`

| phi (ms) | delta | B=0 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|---|
| 65.5 | +5 | 0.933 | 0.600 | 0.267 | 0.073 | 0.000 |
| 70.5 | +10 | 0.933 | 0.633 | 0.367 | 0.146 | 0.000 |
| 80.5 | +20 | 0.933 | 0.633 | 0.400 | 0.220 | 0.000 |
| 90.5 | +30 | 0.933 | 0.667 | 0.467 | 0.317 | 0.000 |
| 110.5 | +50 | 0.933 | 0.700 | 0.633 | 0.561 | 0.000 |

`edf`

| phi (ms) | delta | B=0 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|---|
| 65.5 | +5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 70.5 | +10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 80.5 | +20 | 0.067 | 0.067 | 0.067 | 0.067 | 0.067 |
| 90.5 | +30 | 0.100 | 0.100 | 0.100 | 0.100 | 0.100 |
| 110.5 | +50 | 0.133 | 0.133 | 0.133 | 0.133 | 0.133 |

`heft`

| phi (ms) | delta | B=0 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|---|
| 65.5 | +5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 70.5 | +10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 80.5 | +20 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 90.5 | +30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 110.5 | +50 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

`static_conservative`

| phi (ms) | delta | B=0 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|---|
| 65.5 | +5 | 0.733 | 0.433 | 0.133 | 0.000 | 0.000 |
| 70.5 | +10 | 0.900 | 0.567 | 0.233 | 0.073 | 0.000 |
| 80.5 | +20 | 0.900 | 0.600 | 0.333 | 0.146 | 0.000 |
| 90.5 | +30 | 0.900 | 0.600 | 0.333 | 0.146 | 0.000 |
| 110.5 | +50 | 0.900 | 0.667 | 0.567 | 0.537 | 0.000 |

## Robustness across seeds

- 20 (policy, B) cells checked across seeds [0].
- every cell produced an identical schedule across seeds; all four policies are deterministic, so seeds are a control rather than a source of variance. Seed-to-seed variability is therefore not evidence of robustness — the robustness axes here are B and phi.

## What this does and does not show

- Freshness is **imposed on the schedule and evaluated analytically**, not observed. No dataflow exists between the two networks, so the consumed producer instance is inferred from timestamps and never recorded — on hardware too.
- The consumer is 0.546 ms of work in a 10 ms window on a two-cluster machine, so its deadline success is close to 1.0 **by construction**. The divergence is real but is bounded from above only by the oversubscribed high-B points; it is not an open-ended gap.
- Makespan is pinned near the last consumer release at every B and is not a contention metric here.
- The oracle row is a post-hoc upper bound, not a deployable policy.
- These are solver schedules, not hardware traces.
