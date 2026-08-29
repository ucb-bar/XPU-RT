# What it costs a dispatch to read what the previous one wrote, from elsewhere

`runtime/scripts/k1_cost_by_pred.py`. The artifact
(`artifacts/k1_run/cost_by_pred.json`) is not in git — `artifacts/` is
untracked and the script regenerates it from the board in about fifteen
minutes. This document is the part worth keeping.

## The result

Measured twice, independently, all three arms disjoint both times:

    placement       ratio vs same_hart      run 1     run 2
    same_hart       predecessor on our own hart      1.000     1.000
    same_cluster    another hart, shared L2          1.058     1.068
    other_cluster   the other L2 domain              1.094     1.111

    run 2 separation
      same_hart     max  9.061 ms  <  same_cluster   min  9.470 ms   DISJOINT
      same_cluster  max  9.511 ms  <  other_cluster  min  9.814 ms   DISJOINT

So on this board:

* moving a dispatch off the hart that produced its input costs about **6%**,
* moving it to the **other cluster** costs about **10%**.

Monotonic, in the order the cache hierarchy predicts, and the distributions do
not touch. 16 of DroNet's 21 dispatches show the full ordering individually.

## Why this and not the contention sweep

`docs/k1_contention.md` asked whether two dispatches running *at the same time*
on different harts slow each other down, and measured a null: the same-cluster
and cross-cluster distributions overlapped completely, and the arms were not
monotonic in co-runner count.

That is a different mechanism, and it is not the one the rest of the data
points at. The multi-core sweep found DroNet **slower on eight harts than on
four** (5.32 vs 5.25 ms) while yolov8_nano was not — a working-set effect, not
a concurrency one. Harts 4–7 are a second L2 domain, so a dispatch whose input
was produced across the boundary has to fetch it. That is a property of the
**producer–consumer edge**, which is exactly what `cost_by_pred` models:
`workload_factory` has read a per-dispatch `{"CPU_P#0->CPU_E#0": ms, …}` map
since it was written, and the MILP has consumed it. Nothing had ever measured
one.

So: on this board the cross-cluster cost shows up when a dispatch is *placed
away from its producer*, not when two dispatches sit on opposite sides at the
same time. Do not expect co-runner placement to reveal it.

## The experiment

DroNet is a chain — every dispatch consumes the previous one's output. It is
run three ways, changing only which hart each dispatch lands on:

    same_hart       every dispatch on CPU_P#0
    same_cluster    alternating CPU_P#0 / CPU_P#1      different L1, shared L2
    other_cluster   alternating CPU_P#0 / CPU_E#0      different L1 and L2

**All three are serial.** Each dispatch starts when its predecessor finishes,
so nothing is ever concurrent and contention cannot be the explanation for any
difference between them. That is the control, and it is the reason this
measurement can be read as being about the edge.

The walker runs one worker per (core_kind, hart) inside **one process**, so the
buffers are genuinely shared memory and the consumer genuinely has to fetch
what the producer wrote. `check_schedule_feasibility` passes on all three
variants before any of them is deployed.

**Interleaved, not blocked.** The three variants run A B C A B C, because two
solo runs of DroNet twenty minutes apart differed by 2.6% with nothing else on
the board — the lesson from the contention sweep, where an unpaired design made
drift and effect indistinguishable. Interleaving spreads drift across arms
instead of confounding it with them.

## The map it emits

`cost_by_pred` in the artifact is per dispatch, keyed
`"<pred machine>-><this machine>"`, covering all 64 hart pairs — the form
`workload_factory` already parses.

It is a **model fitted to three measured classes**, not 64 independent
measurements: every pair in the same class (same hart / same cluster / other
cluster) gets that class's measured cost. The artifact says so in its
`derivation` field, and anything quoting it should say so too.

## What is not established

**That the ~6% and ~10% generalise past DroNet.** One model, one chain shape,
one set of tensor sizes. The multi-core sweep already showed the cross-cluster
penalty depends on working-set fit — it reversed DroNet's 8-hart total and did
not reverse yolov8_nano's — so a model with a larger working set may well show
a different number. Re-run the script against another template schedule before
quoting these figures for another network.

**That the same-cluster cost is an L1 effect specifically.** `same_cluster`
changes the L1 and keeps the L2; that it costs less than `other_cluster` is
consistent with the L2 being shared, but the measurement does not separate L1
misses from anything else that differs between two harts of one cluster.

**The largest dispatch shows the smallest effect**, and that is a check rather
than an anomaly: DroNet's first convolution (112×112 input) moves 1.04× across
clusters against a 1.11× total, because its input is the model input rather
than a predecessor's output — there is less for it to fetch. An effect that is
really about the producer–consumer edge should be weakest exactly there.
