# Cost-model A/B: frozen standalone cells vs rebuilt in-situ cells

Three sweep points re-solved and re-run with an identical taskset, identical
bindings and identical board conditions. The only variable is which cost cells
the solver was given.

    point            | BEFORE (frozen)          | AFTER (in-situ)
                     | pred   actual   ratio    | pred   actual   ratio
    baseline_seed0   | 67.57   82.98   1.230    | 80.23   88.91   1.110
    baseline_seed3   | 48.40   52.68   1.090    | 53.00   55.65   1.050
    fused_seed0      | 66.34   83.23   1.250    | 73.23   82.76   1.130

    median |ratio - 1|   23.0%  ->  11.0%

## What improved, and what did not

Per-tile accuracy improved a lot: **11 of 14 cells now predict within +/-15%,
against 5 of 18 before**, and the median per-tile ratio went 1.147 -> 1.043.

Makespan accuracy improved by much less, 1.230 -> 1.110. That difference is
the finding: the parts are now predicted well and the whole still is not, so
the remaining ~10% is **scheduling** -- gate waits, dependency stalls and lane
queueing that make a schedule take longer than the sum of its correctly
predicted tiles. More cell measurement will not close it; it needs a
scheduling-side term.

**A more accurate model is not automatically a faster schedule.** Measured
wall time got slightly worse on two of the three points (+7.2%, +5.6%) and
marginally better on the third (-0.6%), because truer costs moved the
placement (baseline_seed0 went dsp 8 -> 9, hta 6 -> 5). Accuracy and quality
are separate axes and this change bought the first, not the second.

## Convergence was tested, not assumed

Promoting in-situ medians is circular in principle: the values encode the
contention of the schedule that produced them, and installing them changes the
schedule. The measurements notes already record that loop diverging for ViNT's
decoder under the per-run `feedback` stage.

So iteration 2 pooled the runs produced by iteration 1's own schedules and
re-derived every cell:

  * iteration 1 moved cells by +914% / +287% / +72% / -39% / ...
  * iteration 2 moved 13 of 15 cells by **<= 4%**, several by 0%.

It converges. The one exception is `fused_tail@cpu` at +18% -- a
multi-threaded CPU tile, the same class as the cells deliberately excluded,
and worth watching rather than trusting.

## Reproducing

    cd qnn_models/flow_c
    python3 rebuild_cost_model.py --min-n 5                 # dry run
    python3 rebuild_cost_model.py --min-n 5 --write
    cd sweeps/qrb5165_20260829-200620
    export SWEEP_OUT=$PWD/../../costmodel_validation
    python3 drive.py solve && python3 drive.py runtime
    python3 drive.py stage && python3 drive.py run --reps 3
    python3 drive.py results
    python3 ../../costmodel_validation/compare.py

    # convergence check: pool this run back in and see how far cells move
    python3 ../../rebuild_cost_model.py --min-n 5 --also ../../costmodel_validation/results.json

`fused_vint_seed0` was dropped from the A/B: its two cells are the ones
deliberately left alone (`vint_encoders` has too thin a sample,
`vint_decoder@cpu` is schedule-dependent), so it could not move, and its
40-operation MILP exceeded the driver's solve budget.
