# Reproduce the AOT ↔ runtime ↔ HIL co-design loop (on a new machine or a new target)

This is the top-level guide to reproduce the whole SoC/NN co-design pipeline end to end: profile a target,
schedule with XPU-RT, feed back what the *run* differs from the *scheduled Gantt*, adjust ahead-of-time (AOT)
with ModelBlaster, re-schedule, and close with an in-sim HIL drone-flight ablation. It is written to be
**target-agnostic** — SpaceMiT K1 is the worked example, but the seams for a new SoC are called out.

Deep references (this guide points to them, does not duplicate): `docs/the_loop.md` (the definitive loop
doc), `docs/k1_modelblaster_xpurt_closed_loop.md` (operational runbook), `docs/board_calibration_codesign.md`
(the predicted-vs-actual runtime feedback), `XPU-RT/hil/README` (the HIL bridge). Per-figure commands live in
`docs/figure_runbook.md`.

> **Honest scope.** The **scheduling-lever loop** (shard, IME width, board-calibration re-solve) is a genuine
> closed loop. The **graph-rewrite arm** (fuse/unfuse/split via ModelBlaster) is real per-verb on the board
> but **driver-mediated end to end** — only `fuse_with_successor` has run advice→bridge→board fully
> automatically. The "HIL" here is **in-sim zero-order-hold latency injection** (+ a RoSE-lite software-lockstep
> mock target), *not* live FPGA/board co-sim; the physical-K1 path is unimplemented. Board-calibration comes
> from offline K1 traces. Keep these distinctions when presenting.

---

## 0. Environment (two interpreters)

The scheduler and the Isaac flight sim have separate environments.

| Role | Interpreter | Needs |
|---|---|---|
| Scheduler + ModelBlaster + figures | XPU-RT venv: `/scratch2/agustin/XPU-RT/.venv/bin/python` | OR-Tools (CP-SAT), cvxpy + a MILP backend (MOSEK/others), numpy, matplotlib |
| Isaac flight sim | conda env: `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python` | Isaac Sim / IsaacLab (source added to path by the eval scripts), torch |

Always `export XPURT_CPSAT_WORKERS=0` before any CP-SAT solve — the default (1 worker) cripples it.
On a fresh machine: create the XPU-RT venv from the repo requirements (`docs/xpurt_env_setup.md`), install
IsaacLab per its own install, and point the sim scripts at the IsaacLab source.

---

## 1. The loop, stage by stage (one command each)

Run from the repo root `/scratch2/agustin/XPU-RT` (scheduler side) unless noted.

**(1) Profile the target** → per-dispatch cost model. ModelBlaster generates kernels + a dispatch graph and
IREE/`iree-benchmark-module` times each dispatch, producing the profiled CSVs under `gen_mb/…`. The scheduler
reads them via `xpu-rt/profile_loader.py`. (For a new target, this is the main thing you regenerate — see §2.)

**(2) Schedule** with any solver:
```
export XPURT_CPSAT_WORKERS=0
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json \
  --solver milp --scheduler cpsat --profiled --max-periodic-iters 1   # or --solver greedy, or --scheduler mosek
```
→ `schedules/scheduled_<spec>_<solver>_profiled.json` (+ `_metrics.json`, `_report.json`).

**(3) Emit feedback** (two channels):
- Scheduling hints while solving: add `--emit-feedback` → `schedules/xpurt_feedback.json`
  (hints: `prefer_finer`, `prefer_coarser`, `consider_fuse_with_pred`, `pin_target=<combo>`).
- Compile advice from *measurements*: `.venv/bin/python scripts/emit_compile_advice.py …` →
  `compile_advice.json` (verbs: `split`, `fuse_with_successor`, `choose_implementation`, `shard`, `unfuse`).

**(4) Runtime feedback — how the run differs from the Gantt.** The additive Gantt under-predicts the board
(+26–31% on K1, per-op execution inflation, not contention). The measured correction lives in
`results/codesign_feedback/k1_board_calibration.json` and is injected back into *every* solver's cost model:
```
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json <spec> --solver milp --scheduler cpsat \
  --profiled --board-calibration    # scales each dispatch at the profile-load boundary (profile_loader.py)
```
(See `docs/board_calibration_codesign.md`. `scripts/evaluate_exact_cycle_board.py` validates a recalibrated
schedule against ≥10 real board runs.)

**(5) Adjust AOT + re-schedule automatically** — the driver that ties (2)–(4) into rounds:
```
.venv/bin/python scripts/run_codesign_loop.py --workload data/toplevel/<spec>.json   # levers: shard, IME
```
It starts from a clean baseline, proposes each lever, **re-solves each candidate**, and accepts the best
*measured* win with 0 new misses; emits per-round `specs/…`, `round_*_gantt.png`, `loop_report.json`.
The calibration-aware variant (CP-SAT + `--board-calibration` every round) is
`scratchpad/auto_feedback_loop.py`. Graph-rewrite levers (fuse/unfuse/split) go through the bridges
`scripts/advice_to_*_hint.py` → `ModelBlaster/pipeline/apply_*_hint.py` → `generate_kernels.py` (driver-mediated).

**(6) HIL-in-loop flight ablation** (Isaac): sweep drone speed × command frequency, log crash/success:
```
<env_isaaclab>/python sims/scripts/sweep_rate_demo.py --headless --controller rl \
  --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sim_dt 0.01 --decimation 1 \
  --moment_scale 0.0055 --cruise_speed <v> --sched_latency_ms <ms> --episodes 6 --sweep-csv out.csv
```
`sched_latency_ms` is the schedule's worst response; the sim holds each motor command for
`ceil(latency/control_dt)` steps (ZOH). Grid driver: `scratchpad/hil_ablation_grid.sh`; scatter:
`scratchpad/hil_ablation_scatter.py`.

---

## 2. Porting to a NEW target (the seams)

Everything above is target-agnostic except the *profiled cost model* and the *board-calibration table*. To
retarget from K1 to another SoC:

1. **Kernels + profiles**: regenerate ModelBlaster kernels for the new target and re-profile — `.venv/bin/python
   ModelBlaster/pipeline/generate_kernels.py --backend reference|llm --target <your_target>` then time each
   dispatch. Output goes under `gen_mb/…/<target>/…`; the scheduler finds it through `xpu-rt/profile_loader.py`
   (the profile CSV layout is keyed by `network/dispatch_id`).
2. **Machine/topology in the workload spec**: set `hardware.machines` (e.g. `cpu_p`, `cpu_e` counts),
   `hardware.profile.target`, `profile_hw`, and `machine_combination_mode` in `data/toplevel/<spec>.json` to
   your SoC's core layout. IME/accelerator routing is a per-dispatch impl the scheduler picks only if the
   profile says it's faster.
3. **Board-calibration table** (optional but recommended): capture ≥N real runs on the new board, then emit a
   `<target>_board_calibration.json` in the schema of `k1_board_calibration.json`
   (`schema_version: k1_board_calibration/v2`, `aggregate_multiplier`, per-`network/dispatch_id` multipliers,
   `per_op_multiplier` fallback). Pass it via `--board-calibration <path>`. With no table, the flow runs
   additive-only (predicted, no runtime correction).
4. **What stays fixed**: the solvers (greedy/CP-SAT/MOSEK), the loop driver, the feedback emission/consume, the
   figure scripts, and the Isaac flight sim are all target-agnostic — they read the profiled costs.

**K1-specific facts to not hard-code elsewhere**: IME lives on cluster 0 only (harts 0–3); the +31% board gap
and the 1.26× YOLO conv multiplier are K1 measurements; the physical-K1 runtime harness is not implemented.

---

## 3. Regenerate every figure

```
bash scripts/make_all_codesign_figures.sh          # rebuilds the headline figures from on-disk artifacts
```
Notes which steps need the GPU (the warehouse mega plots + the HIL scatter re-run Isaac; the schedule Gantts,
solver-win, and evolution plot are CPU-only from cached schedules). Per-figure commands + inputs are in
`docs/figure_runbook.md`.

## 4. Relevant files — where to look for each concern (code map)

| Concern | Start here |
|---|---|
| **Schedule a workload** | `scripts/run_xpurt_schedule.py` (entry; flags `--solver {milp,greedy}`, `--scheduler {cpsat,mosek,…}`, `--profiled`, `--board-calibration`, `--emit-feedback`) → `xpu-rt/scheduler_cpsat.py`, `xpu-rt/scheduler.py` (MOSEK MILP), `xpu-rt/greedy_scheduler.py` |
| **Cost model / profiles** | `xpu-rt/profile_loader.py` (`load_profiled_processing_times`; the `--board-calibration` scaling at `base_t`); profile CSVs under `gen_mb/…/<target>/…` |
| **The AOT feedback loop** | doc `docs/the_loop.md`; drivers `scripts/run_codesign_loop.py` (auto, levers shard/IME), `scratchpad/auto_feedback_loop.py` (CP-SAT + board-cal every round) |
| **Feedback emission** | `xpu-rt/feedback.py` (`--emit-feedback` → `xpurt_feedback.json`), `xpu-rt/compile_advice.py` + `scripts/emit_compile_advice.py` (`compile_advice.json`, verbs split/fuse/unfuse/shard/choose_impl) |
| **Feedback consume / corroborate** | `xpu-rt/feedback_join.py`, `xpu-rt/advice_join.py` (identity-safety) |
| **Graph rewrite (ModelBlaster)** | bridges `scripts/advice_to_{fusion,split,unfuse,shard}_hint.py`, `advice_to_kernel_choice.py` → `ModelBlaster/pipeline/apply_*_hint.py` → `ModelBlaster/pipeline/generate_kernels.py` |
| **Runtime feedback (run vs Gantt, +31%)** | doc `docs/board_calibration_codesign.md`; `results/codesign_feedback/k1_board_calibration.json`; validate on board with `scripts/evaluate_exact_cycle_board.py` |
| **Isaac warehouse flight** | `sims/scripts/sweep_rate_demo.py` (metrics/ablation), `sims/scripts/record_sensor_demo.py` (video + `--dump_figure_data`), task `sims/isaaclab_tasks/warehouse_nav/` (`HANDOFF.md`, `SPEC.md`) |
| **HIL bridge (mock target)** | `hil/` (`hil_host_isaac.py`, `hil_target_mock.py`, `run_hil_pipeline.sh`, `README`) |
| **Figures** | `scripts/compose_solver_win.py`, `scripts/compose_feedback_evolution.py`, `scripts/plot_solver_gantt_annotated.py`, `sims/scripts/compose_mega_figure.py`; regenerate via `scripts/make_all_codesign_figures.sh`; per-figure recipes in `docs/figure_runbook.md` |
| **Solvers reference** | `docs/solvers.md` (which `--solver`/`--scheduler` combos exist; MOSEK non-convergence caveat) |
| **Workload spec format** | `docs/workload_specs.md`; examples in `data/toplevel/*.json` |

## 5. Run each experiment from scratch

**A. Solver-beats-greedy** (CP-SAT vs greedy on a contended workload):
```
export XPURT_CPSAT_WORKERS=0
for s in greedy "milp --scheduler cpsat --time-limit 900"; do
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<contended-spec>.json \
    --solver $s --profiled --max-periodic-iters 1
done
.venv/bin/python scripts/compose_solver_win.py --greedy <greedy.json> --cpsat <cpsat.json> --spec <spec>
```
Read `_metrics.json` (`deadline_miss_count`, `makespan_ms`) + `_report.json` (`solver_status`). See `docs/solvers.md`.

**B. The AOT feedback loop** (og → levers → runtime feedback, each round re-solved):
```
.venv/bin/python scripts/run_codesign_loop.py --workload data/toplevel/<spec>.json
# emits per-round specs + round_*_gantt.png + loop_report.json (accept/reject per lever)
.venv/bin/python scripts/emit_compile_advice.py …          # compile_advice.json from measurements
# graph rewrites (driver-mediated): scripts/advice_to_<verb>_hint.py → ModelBlaster/pipeline/apply_<verb>_hint.py
```
Full walk-through + the nine-term acceptance rule: `docs/the_loop.md`; operational board runbook:
`docs/k1_modelblaster_xpurt_closed_loop.md`.

**C. Isaac warehouse flight** (a real closed-loop flight through the gate course):
```
<env_isaaclab>/python sims/scripts/record_sensor_demo.py --headless --controller rl \
  --weights sims/models/warehouse/nav_fused_v12_cnn.pt --prop_density 0.35 --obstacle_level 8 \
  --fixed_speed 1.2 --episodes 6 --seed 2000 [--sched_latency_ms <ms>] [--dump_figure_data <dir>] [--save_video out/f.mp4]
```
Scene/asset notes (crate towers, colliders, gate course, cameras): `sims/isaaclab_tasks/warehouse_nav/HANDOFF.md`.
The onboard schedule's control latency enters as a zero-order hold (`--sched_latency_ms`): the motor command
is held for `ceil(latency / control_dt)` steps.

**D. HIL speed × frequency ablation** (the scatter): `bash scratchpad/hil_ablation_grid.sh` (sweeps cruise ×
control-rate × seeds, real Isaac flights → `hil_ablation.csv` via `sweep_rate_demo.py --sweep-csv`) then
`scratchpad/hil_ablation_scatter.py`. Honest scope: in-sim ZOH latency injection (+ RoSE-lite mock in `hil/`),
not live FPGA/K1 co-sim.

## 6. What is real vs stitched (say this when you present)

| Loop stage | Status |
|---|---|
| AOT scheduling levers (shard, IME width) | real, board-proven, auto-driven + re-solved each round |
| Run on the target | real (K1: 60+ RT runs under audited SCHED_FIFO, bit-exact golden checks) |
| Runtime feedback (run vs Gantt, +31%) | real, quantified, injected into the cost model |
| AOT graph rewrite (unfuse) | **automatic in the loop** — `run_codesign_loop.py`'s `unfuse` lever adopts ModelBlaster's rewritten dispatch graph + re-solves + accept/reject, no hand steps. `fuse`/`split` are the same pattern once their rewritten graph is built. |
| Kernel regeneration + board re-profile | **physical boundary, not hand-stitching**: `generate_kernels.py` needs the RISC-V cross-toolchain (`setup_spacemit_toolchain.sh`), and re-profiling needs the physical target — the loop consumes their outputs (the unfused graph + profiles) automatically. |
| HIL flight ablation | in-sim ZOH latency injection (+ RoSE-lite mock), **not** live FPGA/K1 co-sim |
