# QRB5165 schedule sweep — PLANNED (mirrors fpga_20260829-195805)

Written BEFORE the run, in the same shape as
`RoSE/soc/sw/xpu-rt/runs/sweeps/fpga_20260829-195805/SETUP.md`, so the two
can be read side by side. Results land in results.json / ANALYSIS.md here.

## Question

Same two as the FPGA sweep, plus one this target raises on its own:

1. **Does fused_full beat dronet across many workloads?** Matched arms,
   same rng stream, only the mid-size periodic slot differs.
2. **How does a heavy non-periodic vision model reshape a schedule?**
3. **Does the answer to (2) even survive when the accelerator is fast?**
   On the FPGA, ViNT is ~2.7 s standalone — 100x anything else in the bank.
   On this board its encoder tile runs 14.2 ms on the DSP and its whole
   split critical path is ~25 ms, roughly 1.5x dronet's HTA tile. ViNT is
   not a monster here, so the FPGA's "heavy vision reshapes everything"
   result may simply not reproduce. That is a finding either way, and it is
   the reason to run the sweep rather than assume the shapes transfer.

## Arms (matched point-for-point, same as the FPGA sweep)

    baseline    mlp_control + dronet     + yolov8n
    fused       mlp_control + fused_full + yolov8n
    fused_vint  mlp_control + fused_full + yolov8n + vint

Model substitution note: the FPGA bank uses `yolov8_nano` (64x64); this
board runs the 640x640 `yolov8n` export, because that is what has DLCs and
measured cells here. The arms stay matched to each other; they are NOT
matched to the FPGA's absolute numbers, and no cross-target latency
comparison should be drawn from them.

## What ports, and what does not

| FPGA sweep | here |
|---|---|
| `scripts/sweep_unbounded_nonperiodic.py` | **ports as-is** — it emits the same workload schema this flow consumes (`hardware.machines`/`profile_hw`/`networks[].dispatch_deps_path`/`num_instances`). Only the hardware block changes. |
| 2 machines (gemmini_q31 CPU_P, rvv_f16 CPU_E) | **3 machines**: HTA=CPU_P, DSP=CPU_E, CPU=CPU_X. A third lane changes the packing problem, so arms are matched within this target only. |
| 4 FPGA lanes, 18 points in parallel | **one board, three tenants.** Points run strictly serially behind `flock /tmp/qnn_board.lock`. |
| FireSim TARGET cycles, 1 cycle = 1 ns, deterministic | **wall clock on real silicon.** Cells carry ~±10% sweep-to-sweep spread; CPU tiles are load-dependent. Every point therefore runs **3 reps** and reports medians, and records the governor and whether another tenant held the lock. |
| Zephyr ELF per point; provenance from uartlog `entries=N` | **context binaries + a generated C++ runtime per point**; provenance from the trace block. |
| greedy solver, --max-ops 2000 / 12000 | **solver by size**: MOSEK is optimal at 66-86 ops and did not converge at 5576. Sweep A uses MILP with a 300 s limit and falls back to `greedy_periodic` on timeout; Sweep B uses `greedy_periodic` outright. |

## Plan

    Sweep A (matched pair)   arms baseline,fused   seeds 0-7   --max-ops 800
    Sweep B (heavy vision)   arm  fused_vint       seeds 0-1   --max-ops 2000

A = 16 points, B = 2. Budget per point: generate+validate+solve is host-side
(seconds to minutes); the board cost is one context-binary staging pass plus
3 runs of <1 s each, so a point is dominated by the solve, not the hardware.
Cap the op budget far below the FPGA's: at 900 instances this flow's own
horizon logic produced 5,576 operations and MOSEK never converged.

## Validation — a point that fails is REJECTED, never built

The FPGA sweep's four predicates carry over unchanged:

  1. periodic coverage      num_instances*period >= 0.9 * horizon
  2. sporadic containment   no non-periodic task carries a window
  3. uniform stop time      max(n*p)/min(n*p) <= 1.25 across groups
  4. non-empty              at least one non-periodic job to pack against

Three more are specific to this target, where a schedule can be feasible on
paper and unbuildable in fact:

  5. every tile has at least one backend that COMPOSED (from
     measurements/qrb5165_v66.json `cells` + `compose_failures`), so the
     solver is never offered a cell the board rejects
  6. every (tile, backend) the schedule selects has a context binary
     staged on the board — checked before the run, not discovered during it
  7. no capability-excluded sentinel appears in the chosen placement

## Hardware / software under test

    board       QRB5165 (SM8250) at 10.44.120.201, QAIRT 2.45
    machines    HTA (hta0, core 7) / Hexagon v66 DSP (dsp0, core 6) /
                Kryo 585 CPU (cpu0, cores 5,4)
    registry    qnn_models/flow_c/cores/qrb5165_qnn.json
    runtime     generated per point by flowc/emit_runtime.py,
                --lane-mode kind-network, built on-board with g++ 9.4
    conditions  `--tuned`: performance governor on all 8 cores plus one
                warm-up walk (the trace reports walk 2), governor restored
    profiles    gen/profile/<HW>/qrb5165_flowc/... written by modelblaster's
                profile_writer from on-board profile_seg medians

Measured cells this sweep will solve against (performance governor, us):

    dronet          hta 2030   dsp 645
    mlp_control     cpu 28.5   dsp 404
    yolov8n         backbone dsp 13268 / hta 13910 ; head dsp 15377
    fused_full      vision_conv hta 931 / dsp 459 ; depth_conv hta 1569 /
                    dsp 393 / cpu 14 ; tail dsp 4303 / cpu 355
    vint            encoders dsp 14213 ; decoder cpu 12694

## Known confounds, stated up front

* **CPU cells are load-dependent.** ViNT's decoder measures 12.7 ms alone
  and 31-38 ms in a schedule; `feedback` promotion oscillates rather than
  converging. Expect the fused_vint arm's CPU-lane predictions to be the
  least accurate part of the sweep, and report per-tile ratios, not just
  makespan.
* **Two other agents share this board.** Every point records lock wait time;
  a point whose lock wait exceeds its run time is re-run.
* **HTA cells on sub-2 ms graphs are order-of-magnitude only** (measured
  ±4% best case, but 0.26-2.96 ms p99 spread on the dispatch probe).

## Provenance discipline

The FPGA sweep gates on the uartlog's embedded schedule name and
`entries=N`. The equivalent here, all checked per point:

  * the emitted `dispatch_table.h` sha256 recorded with the point
  * `[summary] N/N entries executed` matches the schedule's dispatch count
  * every `[bringup]` line's context filename matches the binding manifest
  * the trace's per-entry `ctx` column matches the intended placement
  * board build gated on rc=0 with the binary newer than the build start

## Outputs

    workloads/    generated + validated workload JSONs
    schedules/    scheduled_*.json
    runtimes/     dispatch_table.h + runtime_main.cpp per point (sha256'd)
    runs/         run.log + trace.csv per point per rep
    results.json  per-point predicted vs actual, per-tile ratios, provenance
    ANALYSIS.md   written after the run
