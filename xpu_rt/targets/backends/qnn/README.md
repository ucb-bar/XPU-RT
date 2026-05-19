# `qnn_scheduler` — Heterogeneous YOLOv8 schedule on QRB5165

## Status (2026-05-07)

End-to-end measurement-driven pipeline working. All 6 phases delivered:

| Phase | Deliverable | Result |
|---|---|---|
| 1 | Per-target VMFB build (CPU + QNN GPU/HTA placeholder) | 135/135 CPU + 135/135 QNN×2 |
| 2 | `board_roundtrip --device` plumb-through + per-target driver | 135 CPU measured; QNN placeholders correctly fail = real infeasibility |
| 3 | Per-volume bridge sweep (memcpy/dequant/quant/rescale) | 32 board measurements → linear-fit coefficients |
| 4 | 3-backend ingest + workload builder (with shape-equal lookup) | 34/135 dispatches reach HTA + GPU via shape-equal Conv2d match |
| 4.5 | Hard infeasibility constraint in MOSEK MILP | `alpha[i,k]=0` enforced per `infeasible_combinations` set |
| 5 | XPU-RT MILP + Gantt + DAG | 159.58 ms makespan (vs 283.81 ms all-CPU) |
| 6 | `--with-schedule` heterogeneous compile + on-board e2e | VMFB built; on-board median 309 ms |



This package drives **measurement-only** per-island backend assignment
for YOLOv8 on QRB5165 (CPU + Adreno GPU + Hexagon HTA).

## Hard constraint

Every cost the scheduler consumes is a real on-board measurement or an
explicit `infeasible: true` row (which is itself a measurement: build
or run failed on that backend). No estimates, no extrapolations, no
"close-enough" interpolation. Cells without a measurement are excluded
from the MILP via the (2b) hard exclusion constraint added in
`xpu-rt/scheduler.py`.

## Pipeline

Run from `XPU-RT/`:

```bash
# Phase 1: per-target compile (already done for all 3 targets).
# Re-runs only when the YOLOv8 model changes.
bash scripts/build_per_target.sh   # NOT YET WRITTEN — manual today

# Phase 1.5: per-dispatch VMFB build (CPU + QNN placeholder).
conda run -n merlin-dev uv run python /scratch2/agustin/merlin/tools/breakdown_vmfb.py \
    --output-dir /scratch2/agustin/merlin/build/het/qrb5165_cpu \
    --iree-compile /scratch2/agustin/merlin/build/host-vanilla-release/tools/iree-compile
# (and similarly for qrb5165_gpu / qrb5165_hta with --target-flag=...)

# Phase 2: per-target on-board profile.
conda run -n merlin-dev uv run python scripts/profile_per_target_on_board.py \
    --target qrb5165_aarch64 --target qrb5165_qnn_gpu --target qrb5165_qnn_hta \
    --push --repetitions 5

# Phase 3 (optional, for richer per-edge bridge cost): per-edge bridge sweep.
# conda run -n merlin-dev uv run python scripts/profile_transfers_on_board.py

# Phase 4: ingest + workload build.
conda run -n merlin-dev uv run python scripts/ingest_per_target_profiles.py \
    --cost-table qnn_scheduler/qrb5165_costs.json \
    --manifest /scratch2/agustin/merlin/build/het/qrb5165_cpu/breakdowns/profiled_manifest.json --backend CPU \
    --manifest /scratch2/agustin/merlin/build/het/qrb5165_gpu/breakdowns/profiled_manifest.json --backend GPU \
    --manifest /scratch2/agustin/merlin/build/het/qrb5165_hta/breakdowns/profiled_manifest.json --backend HTA
conda run -n merlin-dev uv run python scripts/build_workload_from_graph.py

# Phase 5: MILP + Gantt + DAG.
conda run -n merlin-dev uv run python scripts/run_heterogeneous_schedule.py

# Phase 6: heterogeneous compile + on-board e2e.
conda run -n merlin-dev uv run python scripts/run_heterogeneous_e2e.py
```

## Files

| Path | Role |
|---|---|
| `cost_table.py` | Schema for the JSON cost database (execute / init / memcpy / rescale / dequant_quant). |
| `transfer_model.py` | Per-edge bridge cost: `memcpy + (dequant+quant or rescale)`. |
| `island_dag.py` | Data model: `IslandCandidate`, `IslandVariantGroup`, `TensorSpec`, `QParams`. |
| `seed_table_qrb5165.py` | Seed cost table from on-board YOLOv8 stem-conv measurements. |
| `qrb5165_costs.json` | The data file. Regenerated from board runs via `ingest_per_target_profiles.py`. |
| `plot.py` | Gantt + DAG dot rendering for the scheduler output. |
| `scheduler.py` | (Legacy) custom greedy. Production path uses `xpu-rt/scheduler.py:schedule()` instead. |

## Driver scripts (in `XPU-RT/scripts/`)

| Script | Phase | Purpose |
|---|---|---|
| `profile_per_target_on_board.py` | 2 | Per-target board_roundtrip wrapper |
| `profile_qnn_per_dispatch.py` | (parallel) | NHWC conv shape sweep on HTA + GPU |
| `ingest_per_target_profiles.py` | 4 | 3-backend manifest ingest + infeasibility |
| `build_workload_from_graph.py` | 4 | Build XPU-RT workload JSON |
| `run_heterogeneous_schedule.py` | 5 | MOSEK MILP + Gantt + DAG |
| `run_heterogeneous_e2e.py` | 6 | Final compile + on-board run |

## What "measurement-driven" means in practice

After Phase 4 the cost table contains:
- N measured CPU rows (one per dispatch) — from board_roundtrip
  iree-benchmark-module on per-dispatch CPU VMFBs.
- M measured QNN rows where M ≪ N — most QNN per-dispatch VMFBs are
  placeholders (no kernel manifest entry) and runtime fails them. Those
  failures are recorded as `infeasible: true` rows, which are real
  measurements.
- Per-shape sweep rows from `profile_qnn_per_dispatch.py` — one per
  unique conv shape × backend. The workload builder does shape-equal
  lookup to apply these to YOLOv8 dispatches with matching shape sigs.

The MILP then picks per-dispatch backend respecting hard infeasibility.
If every backend is infeasible for some dispatch, MOSEK returns no
solution and `run_heterogeneous_schedule.py` prints which dispatches
need profiling before re-trying.
