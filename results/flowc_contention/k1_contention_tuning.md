# Tuning contention feedback against a concurrent multi-model workload

3560 dispatches across 20 scheduled runs of the K1 exact-cycle workload — mlp_control, dronet, fused_full and ffn_block across eight harts, with per-dispatch cycle counters at 24 MHz.

**61.8% of dispatches overlapped at least one other.** That is the property every QRB5165 trace lacks: all 440 dispatches there ran solo, so contention could not be measured on that board at all. This is the concurrent multi-model measurement the question needs, and it already exists on disk.

## There is a contention signal, and it is large

| co-runners | dispatches | median actual/predicted |
|---|---:|---:|
| 0 | 1361 | 1.0854 |
| 1-2 | 1570 | 1.2341 |
| 3 or more | 629 | 1.2192 |

A dispatch that overlaps others runs **1.137x** longer than one that does not (1.0854 solo against 1.2341). That is a ~14% effect, far above anything a scheduler can ignore.

**This contradicts the standing conclusion, and the disagreement is informative.** `xpu-rt/contention_model.py` records a paired co-runner microbenchmark on this same silicon returning 0.999, 1.012, 1.010, 1.051 same-cluster against 1.061, 0.995, 1.002, 1.004 cross-cluster — two distributions straddling 1.0 — and concludes contention is below the measurement's resolution, so no model should be installed. That was four samples per arm against a synthetic co-runner. This is 3,560 dispatches of real scheduled work. The microbenchmark was not wrong about its own measurement; it was underpowered for this question.

One honest caveat: the effect is **not monotonic in co-runner count** — 3-or-more sits at 1.2192, slightly BELOW the 1-2 bucket's 1.2341. The microbenchmark noted the same non-monotonicity and treated it as evidence of noise. Here it more likely reflects which dispatches end up heavily co-scheduled, so the count is a proxy for something else. It is a real limit on reading the buckets as a dose-response curve.

## The correction transfers out-of-sample

Whole runs are held out, never individual dispatches, so a run's own rows cannot leak into the model that scores it.

| model | held-out median logerr | vs uncorrected |
|---|---:|---:|
| no correction | 0.1510 | — |
| per core-kind | 0.0968 | +35.9% |
| per core-kind x co-runner bucket | 0.0806 | +46.6% |

**Conditioning on co-runners is worth a further 16.7% beyond the per-kind correction alone**, and both hold up out-of-sample.

This is the opposite of what the QRB5165 per-backend correction did: there, leave-one-trace-out helped on only 2 of 4 held-out traces, because those four traces were four different RUNTIME configurations and the bias belonged to the configuration. Here the 20 runs are repetitions of one configuration, and the correction generalises across them cleanly. Both results are about what the correction is keyed on, not about whether feedback works.

## What this does and does not settle

* Contention is measurable and worth modelling on K1 at whole-schedule scale.
* A co-runner-conditioned correction reduces held-out prediction error by 47%.
* It says nothing yet about QRB5165, whose lanes are heterogeneous accelerators rather than harts and where no concurrent trace exists. That run is queued behind the board and is the remaining piece.

