# Benchmarks

Index of the benchmark families in this repo, what each one is for, how to run
it, and where its outputs land. Every benchmark consumes the same `Workload`
substrate (`xpu-rt/workload.py`) and runs against the scheduler registry in
`xpu-rt/schedulers.py`, so any scheduler can be slotted into any benchmark
without further plumbing.

All result directories under `results/` are git-ignored — they are produced by
the scripts listed below.

## Hero benchmark — robotic packing

Pack `dronet @ f_d Hz + mlp_wide @ f_m Hz` into YOLOv8n's end-to-end latency
envelope on a real heterogeneous SoC (QRB5165 or Chipyard / Firesim). Sweeps
the (`f_d`, `f_m`) frontier and reports the maximum sustainable packing
density per scheduler.

```
python scripts/run_robotic_packing.py \
    --soc both \
    --schedulers heft,cpsat,cpsat_memory,mosek,fastest_device,critical_path,edf \
    --dronet-hz 5,10,20,30 \
    --mlp-hz 50,100,200 \
    --out results/robotic_packing/
```

Outputs: per-(soc, f_d, f_m, scheduler) Gantt PNG, per-SoC packing-frontier
plot, cross-SoC summary table, full metrics CSV. Headline result: CP-SAT
sustains zero-miss packing on all 32 evaluated cells (4 frequencies × 4 SoC
combos × 2 mixes).

## Diagnostic scenarios

Small, hand-built workloads that isolate single effects (transfer dominance,
fusion opportunity, memory pressure, etc.) so we can see *why* a scheduler
wins on the hero benchmark.

```
python scripts/run_scenarios.py --scenario all --schedulers fifo,heft,cpsat,cpsat_memory,mosek
```

Outputs: per-(scenario, scheduler) Gantt + a winners-vs-expected-winners table.
Used as a regression scoreboard for scheduler changes.

## Unified benchmark driver — `scripts/run_benchmark.py`

One driver, six subcommands. All produce a `metrics.csv` and `report.md`
under a sibling subdirectory of `results/`.

### `--target scaling`
Scaling sweep across workload sizes (N ∈ {20, 50, 100, 200, 500, …}). Reveals
the solver cliff per scheduler. Source workloads built by chaining replicas
of real DAGs and using synthetic generators.
Output: `results/scaling/`.

### `--target robustness`
Multiplicative Gaussian noise on `Operation.processing_times`
(σ ∈ {0%, 10%, 25%, 50%}), 10 seeds per cell, 18-scheduler grid. Surfaces
schedulers that overfit clean profiles.
Output: `results/robustness/`.

### `--target realtime`
Real-frequency QRB5165 packing — no time scaling. Packs camera@30Hz,
IMU@200Hz, control@100Hz, planning@10Hz, monitor@1Hz. Replaces the synthetic
`--time-scale` thumb in earlier sweeps with a defensible real-silicon
number.
Output: `results/realtime_packing/`.

### `--target literature`
Pegasus literature DAGs (Montage, CyberShake, Epigenomics). Standard
heterogeneous-scheduling reference set; HEFT vs lower-bound ratio is
reported.
Output: `results/literature/`.

### `--target stress`
Five hand-designed stress scenarios:
- `dominator_packing` — one heavy model, several light models packing into
  its makespan.
- `multi_granularity_dronet` — same model at three op granularities
  (1 fused, 3-stage chain, full dispatch graph).
- `mixed_size_stack` — wide mix of model sizes concurrent.
- `solver_killer` — pathological DAG topology targeting MILP/CP-SAT.
- `frequency_sweep_breaking_point` — increases periodic rate until every
  scheduler drops deadlines.
Output: `results/stress/`.

### `--target milp_compare`
Runs the existing MILP formulation through every MILP-capable CVXPY backend
installed on the machine (MOSEK, Gurobi, HiGHS, SCIP, CBC) on a workload set
of three Pegasus DAGs + two real models on Chipyard + three small
scenarios. Lets us answer *"is the win formulation-driven or solver-driven?"*
independently of which solver vendor is available.

```
python scripts/run_benchmark.py --target milp_compare --time-limit 30 --milp-max-ops 30
```

Output: `results/milp_comparison/`. Empirically on the small/medium grid:
HiGHS matches MOSEK on 6/7 feasible cases at comparable wall-time; the
remaining case (CyberShake-27) is where MOSEK finds the proven optimum
while HiGHS terminates on a suboptimal feasible point. The gap there is a
*solver* effect, not a *formulation* effect.

## Closed-loop optimization

Fusion / split rewrite candidates ranked against a deterministic, learned, or
LLM oracle. Used to validate that the rewrite primitives actually move the
needle on a real schedule.

```
python scripts/run_closed_loop.py \
    --workload model:dronet@chipyard \
    --scheduler cpsat \
    --memory-aware \
    --max-candidates 20 \
    --ranker deterministic
```

Output: `results/closed_loop/<tag>/` with an optimization-trace JSON, the
Pareto plot, and before/after Gantts.

## ML training data pipeline

Independent of the benchmark drivers, but listed here for completeness.
Produces the CP-SAT-labelled training corpus consumed by `scheduler_ml.py`,
`scheduler_gnn.py`, and `scheduler_rl.py`.

```
python scripts/gen_training_data.py --stage synthesize
python scripts/gen_training_data.py --stage label
python scripts/train_ml.py --target cost_model
```

Output: `data/training/`, `data/models/` (both git-ignored).

## What is *not* a benchmark

The `xpu-rt/tests/` tree holds correctness gates (validator round-trip,
oracle gap on a 6-op instance, registry smoke, etc.). They run fast and are
expected to pass on every commit; treat them as the regression net the
benchmark suite leans on, not as benchmarks themselves.
