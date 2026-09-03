# FPGA schedule sweep — 20260829-195805

Written BEFORE the run. Everything below is the intended setup; results land
in results.json / ANALYSIS.md in this same directory.

## Question

1. **Does fused_full beat dronet across many workloads?** A single matched
   FPGA pair already says yes (38.22 vs 40.82 ms measured, 3-net schedule),
   but that is one workload shape. This sweeps the same substitution over
   many randomly-drawn tasksets.
2. **How does a heavy non-periodic vision model reshape a schedule?** ViNT is
   605 dispatches / ~2.7 s standalone, ~100x anything else in the bank.

## Arms (matched point-for-point)

  baseline    mlp_control + dronet     + yolov8_nano
  fused       mlp_control + fused_full + yolov8_nano
  fused_vint  mlp_control + fused_full + yolov8_nano + vint

Every arm is generated for every seed with the SAME rng stream, so
baseline_seedN and fused_seedN differ only in which model occupies the
mid-size periodic slot. Verified on the pilot: seed0 gives
mlp_control(16ms x21) + {dronet,fused_full}_a/b(48ms x7), horizon 330 ms,
identical non-periodic set in both arms.

## Plan

  Sweep A (matched pair)   arms baseline,fused   seeds 0-7   --max-ops 2000
  Sweep B (heavy vision)   arm  fused_vint       seeds 0-1   --max-ops 12000

  A = 16 points, each 23-65 instances / 210-413 ms horizon  -> minutes each
  B =  2 points, each ~900 instances / ~9 s horizon         -> ~20 min each
  18 points total across 4 FPGA lanes.

Sweep B needs the larger op budget because the horizon is extended to cover
the non-periodic work: at a 14-16 ms control period, covering ViNT's ~9 s
takes ~570 mlp_control instances. At the default 8000 the max-ops loop would
shrink the horizon back toward the hyperperiod and the periodic groups would
stop ticking early -- the exact truncation documented in
experiments/sweep_fpga/GENERATOR_PROPOSALS.md.

## Workload generation

  scripts/sweep_unbounded_nonperiodic.py --arms ... --seeds ... --max-ops ...

**Non-periodic tasks carry NO release window** (--unbounded-nonperiodic).
They have no min_start_t/max_end_t and are packed as early as dependencies
allow; periodic tasks stay period-bound. This is what avoids the f2opt_v1
degeneracy, where the cursor-based window layout stranded 30 of 70 declared
non-periodic jobs past the horizon and left 13% median utilisation.

Every workload is validated BEFORE it reaches an FPGA. Hard predicates:
  1. periodic coverage      num_instances*period >= 0.9 * horizon
  2. sporadic containment   no non-periodic task carries a window
  3. uniform stop time      max(n*p)/min(n*p) <= 1.25 across groups
  4. non-empty              at least one non-periodic job to pack against
A point that fails is REJECTED and never built.

## Hardware / software under test

  bitstream   f2_dual_small_norose_tacit_q31_60mhz  (SatGemDualSmallTacitConfig)
              2x Rocket; hart0 = Gemmini Q0.31 + Saturn, hart1 = Saturn
              Saturn has Zfh+Zvfh unconditionally (WithRocketVectorUnit
              hardcodes vfh=true, minFLen=16 for every tile)
  farm        4x f2.6xlarge, fq lanes f2-00..03, one host per lane
  registry    cores/chipyard_dual_rocket_gemmini_q31_f16.json
              (e-core kind relabelled rvv_f16 so schedules with fp16 ops
               resolve; identical hardware)
  backends    gemmini_q31 (CPU_P) + rvv_f16 (CPU_E)
  profiles    gen/profile/{gemmini_q31,V256D128_rvv}/firesim_f2_armB/...
              FireSim TARGET cycles at a nominal 1 GHz target clock, so
              1 cycle == 1 ns. The 60 MHz is the HOST FPGA frequency.

Kernels in play (all curated, bit-exact):
  conv2d_s8         gemmini_tiled_conv / gemmini_im2col_full_C
  conv2d_s8_pc      gemmini_im2col_full_C   <- added for ViNT (per-channel)
  linear_s8_pc      gemmini_tiled_matmul    <- added for ViNT (per-channel)
  fp16 + elementwise/reduction ops -> curated rvv_f16 on CPU_E

## Two bugs this sweep depends on (both fixed, both would break it)

  * harness_xpurt/backends/rvv_f16.conf was MISSING, so a scheduled build
    with an rvv_f16 backend got no Kconfig overlay, CONFIG_RISCV_ISA_EXT_V
    stayed unset, Zephyr never enabled mstatus.VS, and the first vsetvli
    trapped (mcause 2). Every fp16 arm here needs it.
  * generate_xpurt_main.py assumed ONE input per model. fused_full is
    multi-input (input0/input1/input2) and its arity differs by quant
    (1 at fp16, 3 at int8), so run.sh now passes --model-gen-dir per network.

## Provenance discipline

Each point verifies, from its own uartlog, that the binary that ran is the
binary intended: the embedded schedule name and `entries=N` must match the
schedule JSON's dispatch count. Stale-ELF reuse has produced false results
twice in this project (once reporting a previous sweep point's 1.000 ratio
as if it were ViNT's), so the driver deletes ELFs before each build, gates
on build rc=0, and requires an ELF newer than the build start.

## Outputs

  workloads/      generated + validated workload JSONs
  schedules/      scheduled_*.json from the greedy solver
  uartlogs/       raw FPGA uartlog per point
  results.json    per-point predicted vs actual + provenance
  ANALYSIS.md     written after the run
