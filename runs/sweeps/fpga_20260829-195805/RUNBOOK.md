# Reproducible FPGA sweep runbook

End-to-end steps to reproduce a scheduled-workload FPGA sweep. `SETUP.md` is
the experiment design; this file is the mechanics. The `drivers/` here are the
originals, copied out of a session scratchpad -- they were NOT in the repo
before, which is why the FPGA half of this sweep was not reproducible.

**All six steps below are now automated by one maintained script**,
`soc/sw/xpu-rt/scripts/repro_fpga_sweep.sh` (the FPGA counterpart of
`scripts/repro_mlp_dronet_yolo_spike.sh`). Every step has a `--skip-*` flag,
and `--dry-run` does steps 1+2 and prints exactly what it would build and
submit without touching an FPGA. Sweep A above:

    bash scripts/repro_fpga_sweep.sh --seeds 0-7 --arms baseline,fused \
        --max-ops 2000 --out-dir runs/sweeps/<TAG> [--dry-run]

Read the rest of this file for the *why*; run the script for the *how*. The
`drivers/` are kept as the provenance of the recorded results, not as the
go-forward path.

Prereqs: `docs/FPGA_QUEUE_USAGE.md` (AWS fq queue), an F2 run host in the pool,
and `~/.ssh/firesim.pem`. All profiling runs on AWS.

## 0. Environment
    cd soc/sw/xpu-rt/zephyr-chipyard-sw
    source scripts/activate_conda.sh
    source scripts/set_envvars_sdk.sh          # NOTE: this reassigns REPO_ROOT
    export PYTHONPATH=$PWD

## 1. Generate + validate + schedule the workloads
    cd soc/sw/xpu-rt
    python3 scripts/sweep_unbounded_nonperiodic.py \
        --seeds 0-7 --arms baseline,fused --max-ops 2000 \
        --hardware f2_gemmini_q31_opt --schedule \
        --out-dir runs/sweeps/<TAG>

Arms are matched point-for-point (same RNG stream), so `baseline_seedN` and
`fused_seedN` differ ONLY in the mid-size periodic model. Available arms:
baseline, fused, baseline_vint, fused_vint, vint_only, all.

Workloads failing validation are REJECTED and never built. Predicates:
periodic coverage >= 0.9*horizon; no non-periodic task carries a release
window; uniform stop time across groups; at least one non-periodic job.

**Arms containing `vint` MUST add `--no-horizon-covers-nonperiodic`.**
Otherwise the horizon is extended to cover ViNT's ~9 s at a ~16 ms control
period => ~566 mlp instances => ~12k dispatches, which the greedy solver does
not finish (measured: >1h39m CPU on a single point, killed). With the flag:
21 instances, ~4 min for 4 points.

    python3 scripts/sweep_unbounded_nonperiodic.py \
        --seeds 0-3 --arms fused_vint --max-ops 4000 \
        --no-horizon-covers-nonperiodic --schedule \
        --out-dir runs/sweeps/<TAG>_vint

Scheduled JSONs land in `soc/sw/xpu-rt/schedules/scheduled_<point>_greedy_profiled.json`.

## 2. Flatten rate-group aliases  (REQUIRED -- builds fail without it)
The generator names rate groups `dronet_a`, `fused_full_b`, ... but
`ingest_xpurt_schedule` only knows base model names and rejects them:
`ValueError: schedule entry 'dronet_a0_dispatch_0' references unknown network`.

    mkdir -p runs/sweeps/<TAG>/schedules_flat
    for f in schedules/scheduled_*_greedy_profiled.json; do
        python3 runs/sweeps/<TAG>/drivers/alias_fix.py "$f" \
            runs/sweeps/<TAG>/schedules_flat/$(basename "$f")
    done

Expect `N jobs -> N instances; dangling 0`. `alias_fix.py` derives base names
from the model bank (it was previously hardcoded to three models, silently
leaving fused_full/vint aliases unflattened).

## 3. Build + submit each point
    bash runs/sweeps/<TAG>/drivers/drive_fpga.sh          # baseline/fused arms
    bash runs/sweeps/<TAG>/drivers/drive_fpga_vint.sh     # vint arms

Per point the driver: deletes stale ELFs, builds with
`BACKENDS=gemmini_q31,rvv_f16`,
`REGISTRY=cores/chipyard_dual_rocket_gemmini_q31_f16.json`,
`CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv_f16`, `XPURT_TRACE=1`,
`STOP_AFTER=build RUNNER=firesim`; gates on `rc=0` AND an ELF newer than the
build start; then `fq submit`s and records `point<TAB>job<TAB>expected_entries`
in `jobs.tsv`.

**Do not skip the freshness gate.** A failed build leaves the previous point's
ELF in place and `ls -t` will happily pick it up -- that produced false results
twice in this project, once reporting a prior sweep point's 1.000 ratio as if
it were ViNT's.

Builds are SERIAL (they share one build dir); FPGA runs are parallel across
lanes, so build(N+1) overlaps run(N).

## 4. Collect + verify provenance + analyse
    ssh -i ~/.ssh/firesim.pem ubuntu@<MGR> \
        "cd /home/ubuntu && tar czf /tmp/res.tgz swres_*/uartlog"
    scp -i ~/.ssh/firesim.pem ubuntu@<MGR>:/tmp/res.tgz runs/sweeps/<TAG>/
    (cd runs/sweeps/<TAG> && tar xzf res.tgz)

For every point assert the uartlog's embedded `entries=N` equals the third
column of `jobs.tsv`. A mismatch means a stale binary ran; discard the point.

Predicted-vs-actual comes from the `MODELBLASTER_XPURT_TRACE` CSV block
(`predicted_start_ms, predicted_duration_ms, actual_start_cycles,
actual_end_cycles`). Drop rows with `dispatch_id < 0` and a zero timestamp --
sentinels that never executed.

**Cycle units: divide by 1000 to get ms.** The cycle columns come from the
guest's `k_cycle_get_64()`, i.e. mtime ticks at
`CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC`, which is `1000000` for this board -- so
1 tick == 1 us and 1000 ticks == 1 ms. Check it against the trace itself:
`mlp_control` instance 20 carries `predicted_start_ms=320.000000` next to
`actual_start_cycles=320000`. (An earlier revision of this file said "FireSim
TARGET cycles at a nominal 1 GHz target clock, so 1 cycle == 1 ns" -- that
wording is wrong; the divisor the recorded `fpga_results.json` actually used
was 1000, not 1e6.) The 60 MHz in the bitstream name is the HOST FPGA
emulation frequency and is irrelevant here; using 60 would scale results
16.7x.

To turn a single-model uartlog into a scheduler-ingestible profile:
    python3 drivers/uartlog_to_profile.py --uartlog U --model M --quant int8 \
        --backend {gemmini_q31|V256D128_rvv} --cpu firesim_f2_armB \
        --cores 0 --clock-mhz 1000 --out-root soc/sw/xpu-rt/gen/profile

## 5. Expected results
Note the two `*.json` in a sweep dir are different things: `results.json` is
the *generator's* validation report (which points passed, and are therefore
built), `fpga_results.json` is the predicted-vs-actual analysis.

Sweep A (no vint): ratio act/pred 0.999-1.000, per-op median err 1.3-2.6%.
Sweep B (with vint): completes, but ratio ~1.5-1.6 because of the
per-dispatch IRQ guard -- see "Known workarounds" below. Per-op error stays
1.3-1.6% in both, i.e. the cost model is accurate either way.

## Known workarounds currently in the tree
* **Per-dispatch IRQ guard** (`generate_xpurt_main.py`, `XPURT_DISPATCH_IRQ_GUARD`,
  default 1). Without it, ViNT runs fault in `z_riscv_vstate_restore_thread`
  from a vector kernel -- the trap-corrupts-a-scalar-register defect that
  `harness/src/main.c` documents. Costs ~1.5x MAKESPAN (not kernel time:
  kernel sum is within 0.3% of unguarded). Set to 0 to reproduce the fault.
* **Bounded profile-record loop** (same file). Was unbounded, walked off the
  array and faulted in printf's `strnlen` AFTER the run had completed.
* **`harness_xpurt/backends/rvv_f16.conf`** must exist. Without it a scheduled
  build with an rvv_f16 backend gets no Kconfig overlay,
  `CONFIG_RISCV_ISA_EXT_V` is unset, and the first `vsetvli` traps.
* `serialize_instances.py` removes same-network instance overlap. NOT needed
  for the vstate fault (that hypothesis was disproven) but the underlying
  hazard is real: `buffers.c` is one scratch set per model, so concurrent
  instances of one network do corrupt each other.

## Gotcha index
* `set_envvars_sdk.sh` reassigns `REPO_ROOT`; save/restore it around the source.
* `--backend` in the per-model flow is the GENERATOR (reference|llm); the
  hardware target is `--target`.
* Multi-input models need `--model-gen-dir NET=PATH` (arity differs by quant:
  fused_full has 1 input at fp16, 3 at int8).
* `ls` output carries ANSI colour here; parse filenames with python/glob, not
  `ls | grep`.
