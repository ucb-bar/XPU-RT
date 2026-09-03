# FPGA schedule sweep — results

Setup and intent: SETUP.md (written before the run).

## Sweep A — baseline vs fused, 10 points, ALL VALIDATED
16 generated, 10 passed validation (seeds 4/5/7 rejected in BOTH arms: they
drew yolov8_nano count=0, leaving no non-periodic work to pack against).
All 10 ran on FPGA, all provenance-verified (uartlog `entries=` == the
submitted schedule's dispatch count).

  point              entries  ratio act/pred  per-op p50 err
  baseline_seed0        1061       1.000          1.4%
  baseline_seed1        1195       1.000          1.3%
  baseline_seed2        1117       1.000          1.4%
  baseline_seed3         541       0.999          2.6%
  baseline_seed6        1272       1.000          1.7%
  fused_seed0            991       1.000          1.7%
  fused_seed1           1145       1.000          1.6%
  fused_seed2           1092       1.000          1.6%
  fused_seed3            511       0.999          1.7%
  fused_seed6           1212       1.000          1.7%

Cost model holds to 0.1% on real hardware across every point.

**Caveat on the arm comparison.** The FPGA trace span is the PERIODIC horizon,
which is identical by construction in both arms (320.53 / 406.53 / 330.53 /
196.53 / 408.53 ms). The baseline-vs-fused difference lives in the
NON-PERIODIC completion time, which the scheduler reports and the trace span
does not isolate. Scheduler-side, fused_full won 4 of 5 seeds (mean ~-2.5%,
seed2 went +3.6% the other way), and a dedicated 3-net FPGA comparison
measured 38.22 vs 40.82 ms (-6.4%). So fused_full is generally but not
universally faster than dronet.

## Sweep B — fused_vint, 4 points, ALL CLEAN
Required three fixes (below). Final run: no faults, full traces, 4/4 model
outputs per point, all provenance-verified.

  point                entries  pred ms  act ms  ratio  per-op p50 err
  fused_vint_seed0        1989   5592.2  9108.4  1.629      1.5%
  fused_vint_seed1        2096   5616.0  9125.4  1.625      1.3%
  fused_vint_seed2        1546   3013.8  4629.7  1.536      1.6%
  fused_vint_seed3        1721   5636.1  9089.9  1.613      1.4%

**These ratios are NOT a cost-model result.** Per-op error stays at 1.3-1.6%
-- individual kernel costs are predicted as well as in Sweep A. The ~1.6x
makespan inflation is the per-dispatch IRQ guard (fix 1). A/B on the same
seed2 schedule: ratio 1.014 without the guard vs 1.534 with it, and the sum
of dispatch durations grows 4636 -> 6841 ms (+49%), so the guard slows the
kernels themselves, not just scheduling latency. Sweep A needs no guard and
is the clean validation; Sweep B demonstrates completion, not timing fidelity.

## Fixes required to make Sweep B run

1. **Per-dispatch IRQ guard** (pipeline/generate_xpurt_main.py). Every failure
   symbolised to `z_riscv_vstate_restore_thread` called from a ViNT vector
   kernel. harness/src/main.c documents the defect: "a trap taken while a
   vector kernel is executing can come back with EXACTLY ONE scalar register
   corrupted ... invisible on spike", and works around it by masking IRQs for
   a whole inference. That file also warns a thread pool "must NOT do this" --
   about holding a lock across BLOCKING work. Here the lock spans ONE kernel
   call; every k_sem wait and scheduler decision is outside it, the two
   workers are pinned one-per-hart, the intra-op pool has 0 helpers, and
   irq_lock() masks only the calling hart. XPURT_DISPATCH_IRQ_GUARD=0 restores
   the fault. Effect: runs went from dying at 46 lines to executing every
   dispatch.

2. **Bounded profile-record loop** (same file). The loop trusted n_records
   from model_<m>_profile_records_<bs>() unbounded, walked off the array and
   faulted in printf's strnlen() on a garbage name pointer -- AFTER the run,
   its full XPURT_TRACE and all four OUTPUT blocks had been emitted. Now
   clamped to MODEL_<M>_OP_COUNT.

3. **Alias flattener base list** (experiments/sweep_fpga/schedules_in/
   alias_fix.py). Was hardcoded to (mlp_control, dronet, yolov8_nano), so
   fused_full_a/_b and vint_a/_b never flattened and ingest rejected them.
   Now derived from the model bank.

## Two hypotheses that were WRONG (recorded so they are not retried)

* **Concurrent same-network instances corrupting shared buffers.** buffers.c
  really is one scratch set per model (884 arrays / 30.9 MB for vint, header:
  "must be linked EXACTLY ONCE per model"), and vint overlap correlated with
  the failures. But serialising instances removed every same-network overlap
  (verified 0/0/0/0) and all points still faulted, just as fast. The
  correlation was coincidental: ViNT is simply the model with enough long
  vector kernels to be preempted mid-kernel. The buffer-sharing hazard is
  real and still unfixed -- it is just not what caused these crashes.
  Serializer kept at schedules_in/serialize_instances.py.

* **CONFIG_RISCV_V_KERNEL_ONLY.** Added on the theory that auto-vectorised
  non-kernel code left V state live. It REGRESSED seed2 from passing to
  faulting. Reverted.

## Generator note for the next sweep
seeds 4/5/7 were lost to `yolov8_nano count=0`. Raising that count.min from 0
to 1 in the model bank would recover ~37% of points in the baseline/fused arms.
