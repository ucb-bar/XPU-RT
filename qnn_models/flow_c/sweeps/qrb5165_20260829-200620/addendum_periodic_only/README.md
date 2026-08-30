# Addendum: the six periodic-only points

The sweep in the parent directory generated 18 points and ran 12. Six were
rejected before build by a pre-registered predicate:

    no non-periodic work: nothing to pack against the periodic load

This addendum runs those six. It is an **addendum, not a re-run**: the parent
sweep's `generated.json`, `results.json`, `provenance.json` and `state.json`
are untouched, and the six keep `status: REJECTED` there. Everything here
lands in this directory via `drive.py`'s `SWEEP_OUT`, and reuses the parent's
frozen `cost_model.json` so the two are directly comparable.

    baseline_seed4   mlp_control x6, dronet_a x3, dronet_b x3
    fused_seed4      mlp_control x6, fused_split x3
    baseline_seed5   mlp_control x9, dronet x3
    fused_seed5      mlp_control x9, fused_split x3
    baseline_seed7   mlp_control x3, dronet x3
    fused_seed7      mlp_control x3, fused_split x3

## Why run them

Q1 asks whether `fused_full` beats `dronet` in the mid-size periodic slot. In
all 12 accepted points a `yolov8n_head` finishes last, so the mid-size slot
can only move the makespan second-order -- which is why Q1 came back weak (3
of 5 pairs, one clearing the noise floor). These six have no yolov8n at all,
so the makespan is owned by periodic work.

**How well that reasoning survived contact: partially.** In a purely periodic
workload the makespan is `last release of some periodic network + its
duration`, and which network that is depends on the draw:

  * **seeds 4 and 5** -- the highest-count network (`mlp_control`, 6 and 9
    instances) is released last, so the makespan is set by the release
    schedule and comes out *identical* across arms (20.107 / 20.107 and
    48.107 / 48.107 ms). Uninformative for Q1.
  * **seed 7** -- the mid-size network is released last and owns the
    makespan (16.645 = dronet's last release at 16 ms + 0.645). This is the
    one informative pair, and it goes *against* fused: 16.813 vs 16.645,
    +1.0%.

So the addendum yields one informative Q1 pair, not three. Seeds 4 and 5 are
still worth their board time: they measure predicted-vs-actual in a
periodic-only regime that no point in the main sweep covers.

## What had to be fixed to run them at all

The predicate turned out to guard a real limitation, not just a preference.
Three independent places in the XPU-RT scheduler stack degrade when a
workload has zero non-periodic operations:

| site | behaviour with zero non-periodic ops |
|---|---|
| `xpu-rt/scheduler.py:510` | `C_max` is unconstrained from below, the MILP objective is trivial, and MOSEK raises `SolverError`. The greedy fallback then reports a `0.00 ms` non-periodic makespan. |
| `xpu-rt/postprocessing.py:312` | trim is a no-op. Benign and already handled. |
| `xpu-rt/workload_factory.py:415` | collapses every periodic network to **1 instance, ignoring an explicit `num_instances`**. |

The third is a bug and is **fixed** in this branch. The comment called it a
default, but it beat an explicit setting: the workload asks for
`num_instances: 6`, the override at `workload_factory.py:460` exists so a
toplevel JSON can pin instance counts, and the early return jumped over it.
The result was a 12-operation workload silently scheduled as 3 operations.
The fix honours `num_instances` on that path and changes nothing else --
it only executes when there are no non-periodic ops, which no accepted point
in the main sweep has.

*Regression check:* `baseline_seed0` re-solved after the fix gives 20
operations, `optimal`, **67.571 ms** -- identical to its pre-registered
value.

The first site is worked around per-point rather than fixed, by setting
`restrict_makespan_to_nonperiodic: false` in these six workloads so `C_max`
covers all operations. That is a deliberate, recorded deviation, noted in
each workload's `_comment`.

**Consequence:** predicted makespans here are *not* comparable to the main
sweep's, because the objective differs. The Q1 comparison is internal to this
addendum -- `baseline_seedN` vs `fused_seedN`, same objective, same frozen
cost model -- and is unaffected.

## Reproducing

    cd qnn_models/flow_c/sweeps/qrb5165_20260829-200620
    python3 build_addendum.py
    export SWEEP_OUT=$PWD/addendum_periodic_only
    python3 drive.py solve
    python3 drive.py runtime
    python3 drive.py stage
    python3 drive.py run --reps 3
    python3 drive.py results
    python3 analyse_addendum.py

## Results

18 runs, 6 points x 3 reps, all `ok`, all entries dispatched. Tables from
`../analyse_addendum.py`; raw data in `results.json`.

### Q1: dronet wins, and it is not close

    seed  ops b/f   pred b  pred f    d%  |   act b   act f     d%  | noise b/f %  verdict
    4      12/15     20.11   20.11   0.0  |   20.04   20.46    2.1  |  0.0/0.1     dronet wins
    5      12/18     48.11   48.11   0.0  |   48.04   48.05    0.0  |  0.0/2.0     within noise
    7       6/12     16.64   16.81   1.0  |   18.19   18.97    4.3  |  0.4/1.1     dronet wins

Two of three pairs favour dronet by margins far outside the rep noise, and
the third is a tie. **This is the opposite direction from both the main sweep
(fused better in 3 of 5, weakly) and the FPGA (fused better in 4 of 5).**

The mechanism is not that fused_split dispatches more slowly. It is that it
is simply *more work* -- it fuses several sensor branches, so substituting it
into the mid-size slot adds compute:

    point            entries  busy ms  makespan  lanes used
    baseline_seed4        12    4.887    20.043  {DSP: 6, CPU: 6}
    fused_seed4           15   13.903    20.450  {HTA: 4, DSP: 6, CPU: 5}
    baseline_seed7         6    5.297    18.182  {DSP: 3, CPU: 3}
    fused_seed7           12    7.899    18.974  {DSP: 4, CPU: 6, HTA: 2}

fused_seed4 executes 2.8x the busy time of its baseline inside the same 20 ms
release window. So the honest statement of Q1 is not "which model is faster"
but **"does the extra fused work fit in the schedule's slack?"** -- and that
depends entirely on what sets the makespan:

  * **release-gated** (seeds 4 and 5 baseline, seed 5 fused): the makespan is
    `last mlp_control release + duration`. The extra fused work hides in the
    slack and costs 0.0-2.1%.
  * **compute-bound** (seed 7): the mid-size model owns the critical path and
    the extra work is fully exposed, costing 4.3%.

In the main sweep a yolov8n_head always owns the critical path, which is a
third regime -- there the mid-size slot is masked entirely, and fused's small
apparent win never clears the noise floor. Across all three regimes the
consistent reading is that **fused_full's advantage on this target is not
reproducible; where it is visible at all it is a cost.**

### Prediction quality, and a clean decomposition of the 1.17x

    point                  pred ms  median act   ratio
    baseline_seed4           20.11       20.04   1.000
    baseline_seed5           48.11       48.04   1.000
    fused_seed5              48.11       48.05   1.000
    fused_seed4              20.11       20.46   1.020
    baseline_seed7           16.64       18.19   1.090
    fused_seed7              16.81       18.97   1.130

    median ratio, periodic-only: 1.010x   (main sweep, mixed: 1.17x)

This splits the main sweep's 1.17x cleanly. Where the makespan is
**release-gated the model is exact** -- three points at 1.000, because the
schedule is waiting on wall-clock releases the runtime hits to the
microsecond. Where it is **compute-bound the error appears** -- 1.09x and
1.13x, the same magnitude as the main sweep's per-lane ratios (hta 1.092, dsp
1.165, cpu 1.314). The 1.17x is therefore not a uniform modelling bias; it is
concentrated entirely in the compute-bound portion of the schedule, and the
release-gated portion contributes zero error.
