# Migration Guide — Original XPU-RT → Integrated Branch

This document maps **every file and directory** from the original
[`ucb-bar/XPU-RT`](https://github.com/ucb-bar/XPU-RT) repository to its new
location in this integrated branch (`integration/xpu-rt-cleanup`).

The integrated branch unifies two codebases:

1. **Original XPU-RT** — CVX/MILP scheduler + native C/CUDA runtime + QNN
   backend + IsaacLab sim tasks. This is the codebase mapped below.
2. **Compiler generator** — the LLM-driven graph compiler (previously
   developed in a sibling repo, now folded under `xpu_rt/` as a peer of the
   scheduler). All `CompGen*` identifiers have been renamed to `XpuRt*`.

If you're looking for something from the original XPU-RT, find its row
below.

---

## High-level moves

| Original (XPU-RT root)          | Integrated location                | Notes                                                          |
| ------------------------------- | ---------------------------------- | -------------------------------------------------------------- |
| `xpu-rt/` (the Python package)  | `xpu_rt/scheduler/`                | The flat scheduler module became a sub-package of `xpu_rt/`.   |
| `qnn_scheduler/`                | `xpu_rt/targets/backends/qnn/`     | Folded under the QNN target backend.                           |
| `qnn_models/`                   | `models/qnn/`                      | Renamed; ONNX → TFLite → QNN DLC conversion tools.             |
| `runtime/`                      | `runtime/`                         | Unchanged. Native C/CUDA runtime sits at the repo root.        |
| `sims/`                         | `sims/`                            | Unchanged. IsaacLab tasks + training scripts (now gitignored). |
| `data/`                         | `data/`                            | Unchanged. Op-definition KB + per-model fixtures.              |
| `merlin/`                       | `third_party/merlin/`              | Now a git submodule.                                           |
| `scripts/`                      | split: `scripts/` + `tools/...`    | User-facing → `scripts/`; dev/demos/profiling → `tools/`.      |
| `docs/`                         | `docs/`                            | Old integration notes deleted; new user-facing docs.           |
| `paper/`, `plots/`, `sessions/` | gitignored / not in integrated tree | Local-only build/session artifacts.                            |
| `.mcp.json`                     | removed                            | MCP setup is now `xpu-rt mcp install`.                         |
| `setup.py`                      | folded into `pyproject.toml`       | Build system migrated to Hatchling + uv.                       |

---

## `xpu-rt/` → `xpu_rt/scheduler/`

The flat scheduler module at the upstream root became a sub-package.

| Original                                   | Integrated                                          |
| ------------------------------------------ | --------------------------------------------------- |
| `xpu-rt/__init__.py`                       | `xpu_rt/scheduler/__init__.py`                      |
| `xpu-rt/scheduler.py`                      | `xpu_rt/scheduler/scheduler.py`                     |
| `xpu-rt/comparison.py`                     | `xpu_rt/scheduler/comparison.py`                    |
| `xpu-rt/feedback.py`                       | `xpu_rt/scheduler/feedback.py`                      |
| `xpu-rt/fusion.py`                         | `xpu_rt/scheduler/fusion.py`                        |
| `xpu-rt/packing.py`                        | `xpu_rt/scheduler/packing.py`                       |
| `xpu-rt/plot.py`                           | `xpu_rt/scheduler/plot.py`                          |
| `xpu-rt/postprocessing.py`                 | `xpu_rt/scheduler/postprocessing.py`                |
| `xpu-rt/profile_loader.py`                 | `xpu_rt/scheduler/profile_loader.py`                |
| `xpu-rt/profile_metrics.py`                | `xpu_rt/scheduler/profile_metrics.py`               |
| `xpu-rt/schedule_validation.py`            | `xpu_rt/scheduler/schedule_validation.py`           |
| `xpu-rt/streaming_feedback.py`             | `xpu_rt/scheduler/streaming_feedback.py`            |
| `xpu-rt/workload.py`                       | `xpu_rt/scheduler/workload.py`                      |
| `xpu-rt/workload_factory.py`               | `xpu_rt/scheduler/workload_factory.py`              |
| `xpu-rt/pytorch_workload/`                 | `xpu_rt/scheduler/pytorch_workload/`                |
| `xpu-rt/pytorch_workload/dronet/`          | `xpu_rt/scheduler/pytorch_workload/dronet/`         |
| `xpu-rt/pytorch_workload/fastdepth/`       | `xpu_rt/scheduler/pytorch_workload/fastdepth/`      |
| `xpu-rt/pytorch_workload/mlp/`             | `xpu_rt/scheduler/pytorch_workload/mlp/`            |
| `xpu-rt/pytorch_workload/onnx_compilation_script/` | `xpu_rt/scheduler/pytorch_workload/onnx_compilation_script/` |
| `xpu-rt/pytorch_workload/samples/`         | `xpu_rt/scheduler/pytorch_workload/samples/`        |
| `xpu-rt/pytorch_workload/simple_mlp.onnx`  | `xpu_rt/scheduler/pytorch_workload/simple_mlp.onnx` |
| `xpu-rt/tests/test_feedback_derivation.py` | `xpu_rt/scheduler/test_feedback_derivation.py`      |

New peers added alongside the migrated files:

- `xpu_rt/scheduler/bridge.py` — bridge between `xpu_rt.solve` and the CVX scheduler.
- `xpu_rt/scheduler/qnn_model_loader.py` — load real QNN models for the scheduler.
- `xpu_rt/scheduler/qnn_real_workload.py` — real QNN workloads (vs synthetic).

Tests of the scheduler at the top-level `tests/` tree:

- `tests/scheduler/test_qnn_model_loader.py`
- `tests/scheduler/test_qnn_real_workload.py`
- `tests/scheduling/test_*` — scheduling **policy** tests (feedback loop, granularity, etc.); these test the wider compile-time scheduling layer, not the CVX scheduler proper.

---

## `qnn_scheduler/` → `xpu_rt/targets/backends/qnn/`

The standalone QNN scheduler was folded into the QNN target backend.

| Original                              | Integrated                                            |
| ------------------------------------- | ----------------------------------------------------- |
| `qnn_scheduler/__init__.py`           | `xpu_rt/targets/backends/qnn/__init__.py`             |
| `qnn_scheduler/scheduler.py`          | merged into `xpu_rt/targets/backends/qnn/*.py`        |
| `qnn_scheduler/cost_table.py`         | `xpu_rt/targets/backends/qnn/cost_table.py`           |
| `qnn_scheduler/island_dag.py`         | `xpu_rt/targets/backends/qnn/island_dag.py`           |
| `qnn_scheduler/transfer_model.py`     | merged into `xpu_rt/targets/backends/qnn/*.py`        |
| `qnn_scheduler/seed_table_qrb5165.py` | merged into `xpu_rt/targets/backends/qnn/cost_table.py` |
| `qnn_scheduler/qrb5165_costs.json`    | `xpu_rt/targets/backends/qnn/qrb5165_costs.json`      |
| `qnn_scheduler/plot.py`               | `xpu_rt/targets/backends/qnn/plot.py`                 |
| `qnn_scheduler/README.md`             | `xpu_rt/targets/backends/qnn/README.md`               |

New peers added under the same path: `board.py`, `contention.py`,
`context_builder.py`, `execute_schedule.py`, `granularity.py`,
`granularity_proposal.py`, `mosek_bridge.py`, `on_board_runner.py`,
`onnx_bridge.py`, `placement.py`, `profile_detailed.py`,
`profile_lookup.py`, `proof.py`.

---

## `qnn_models/` → `models/qnn/`

Verbatim rename. Contents preserved.

| Original                            | Integrated                       |
| ----------------------------------- | -------------------------------- |
| `qnn_models/benchmark_qnn.sh`       | `models/qnn/benchmark_qnn.sh`    |
| `qnn_models/benchmark_results.json` | `models/qnn/benchmark_results.json` |
| `qnn_models/deploy.sh`              | `models/qnn/deploy.sh`           |
| `qnn_models/Dockerfile.qnn-convert` | `models/qnn/Dockerfile.qnn-convert` |
| `qnn_models/dronet.py`              | `models/qnn/dronet.py`           |
| `qnn_models/export_mobilenet.py`    | `models/qnn/export_mobilenet.py` |
| `qnn_models/export_onnx.py`         | `models/qnn/export_onnx.py`      |
| `qnn_models/export_yolo.py`         | `models/qnn/export_yolo.py`      |
| `qnn_models/onnx2tf_convert.py`     | `models/qnn/onnx2tf_convert.py`  |
| `qnn_models/plot_benchmarks.py`     | `models/qnn/plot_benchmarks.py`  |
| `qnn_models/plots/`                 | `models/qnn/plots/`              |
| `qnn_models/README.md`              | `models/qnn/README.md`           |
| `qnn_models/run_dronet.py`          | `models/qnn/run_dronet.py`       |

The 13 files under `models/qnn/` are still tracked by git. The
`/models/` line in `.gitignore` only prevents *new* files from being
committed; it does not un-track existing files. Run
`git rm --cached -r models/` if you want it fully ignored.

---

## `runtime/`

Unchanged. The native C/CUDA runtime stays at the repo root with the same
internal layout.

| Path                       | Status                                                        |
| -------------------------- | ------------------------------------------------------------- |
| `runtime/CMakeLists.txt`   | Unchanged                                                     |
| `runtime/build_runtime.sh` | Unchanged                                                     |
| `runtime/scripts/`         | Unchanged                                                     |
| `runtime/tools/`           | Unchanged (includes `json_dispatch_runner`, `xpurt_scheduler_runner`) |
| `runtime/README.md`        | Unchanged                                                     |

The integrated tree adds `runtime/include/`, `runtime/src/`,
`runtime/templates/`, and `runtime/native/libxpu_rt/` (the merged C
runtime) under the same root.

---

## `sims/`

Unchanged path. Now in `.gitignore` (developer-local).

| Original                                | Integrated                              |
| --------------------------------------- | --------------------------------------- |
| `sims/IsaacLab/`                        | `third_party/IsaacLab/` (submodule)     |
| `sims/isaaclab_tasks/`                  | `sims/isaaclab_tasks/` (untracked)      |
| `sims/isaaclab_tasks/track_steering_vision/` | `sims/isaaclab_tasks/track_steering_vision/` |
| `sims/scripts/`                         | `sims/scripts/` (untracked)             |

---

## `data/`

Unchanged path. Benchmark calibration data preserved.

| Original                  | Integrated         |
| ------------------------- | ------------------ |
| `data/diffusion_scalar/`  | `data/diffusion_scalar/` |
| `data/dronet/`            | `data/dronet/`     |
| `data/dronet_rvv/`        | `data/dronet_rvv/` |
| `data/dronet_scalar/`     | `data/dronet_scalar/` |
| `data/dronet_ukernel/`    | `data/dronet_ukernel/` |
| `data/fastdepth/`         | `data/fastdepth/`  |
| `data/fastdepth_rvv/`     | `data/fastdepth_rvv/` |
| `data/fastdepth_scalar/`  | `data/fastdepth_scalar/` |
| `data/fastdepth_scalar_new/` | `data/fastdepth_scalar_new/` |
| `data/glpdepth/`          | `data/glpdepth/`   |
| (other per-model fixtures) | preserved with original names |

`data/compiler_optimization_kb.yaml` is also preserved.

---

## `merlin/` → `third_party/merlin/`

Promoted to a git submodule. Same upstream URL.

```
[submodule "third_party/merlin"]
    path = third_party/merlin
    url = https://github.com/ucb-bar/merlin.git
```

---

## `scripts/` → `scripts/` + `tools/...`

Original `scripts/` was a single flat directory mixing user-facing helpers,
demos, profiling tools, and scheduling drivers. Split into two top-level
trees:

- `scripts/` — user-facing entry points (small set).
- `tools/` — developer internals organised by purpose (`tools/dev/`,
  `tools/experiments/`, `tools/ci/`, `tools/e2e/`, `tools/demos/`,
  `tools/profiling/`, `tools/scheduling/`, `tools/board/`, `tools/misc/`).

| Original (`scripts/`)                  | Integrated                                              |
| -------------------------------------- | ------------------------------------------------------- |
| `build_workload_from_graph.py`         | `tools/scheduling/build_workload_from_graph.py`         |
| `compgen-mcp.sh`                       | `scripts/xpu-rt-mcp.sh` (renamed; CompGen scrub)        |
| `heterogeneous_loop.py`                | `tools/scheduling/heterogeneous_loop.py`                |
| `ingest_per_target_profiles.py`        | `tools/profiling/ingest_per_target_profiles.py`         |
| `merlin_adapter.py`                    | `tools/scheduling/merlin_adapter.py`                    |
| `packing_demo.py`                      | `tools/demos/packing_demo.py`                           |
| `plot_scheduled_json.py`               | `tools/scheduling/plot_scheduled_json.py`               |
| `profile_per_target_on_board.py`       | `tools/profiling/profile_per_target_on_board.py`        |
| `profile_qnn_per_dispatch.py`          | `tools/profiling/profile_qnn_per_dispatch.py`           |
| `profile_transfers_on_board.py`        | `tools/profiling/profile_transfers_on_board.py`         |
| `qnn_island_demo.py`                   | `tools/demos/qnn_island_demo.py`                        |
| `qnn_island_demo_v2.py`                | `tools/demos/qnn_island_demo_v2.py`                     |
| `run_greedy_schedule.py`               | `tools/scheduling/run_greedy_schedule.py`               |
| `run_heterogeneous_e2e.py`             | `tools/scheduling/run_heterogeneous_e2e.py`             |
| `run_heterogeneous_schedule.py`        | `tools/scheduling/run_heterogeneous_schedule.py`        |
| `run_on_board_flow.py`                 | `tools/scheduling/run_on_board_flow.py`                 |
| `run_xpurt_schedule.py`                | `tools/scheduling/run_xpurt_schedule.py`                |
| `testing.py`                           | **deleted** (orphan; incomplete stub)                   |
| `worst_case_nonperiodic_duration.py`   | **deleted** (orphan math utility)                       |
| `worst_case_periodic_window_fraction.py` | **deleted** (orphan math utility)                     |

---

## `docs/`

Original `docs/` contained only two integration notes; both removed since
the integration they described is now this repo.

| Original                       | Integrated                                                |
| ------------------------------ | --------------------------------------------------------- |
| `docs/compgen_integration.md`  | **deleted** (this integration is now the repo itself)     |
| `docs/merlin_integration.md`   | **deleted** (Merlin is now `third_party/merlin/`)         |

The integrated `docs/` is a much larger user-facing tree built around the
new mkdocs site: `docs/getting-started/`, `docs/guides/`,
`docs/architecture/`, `docs/reference/`, `docs/scheduling/`,
`docs/model_setup/`, etc.

---

## Top-level files

| Original                            | Integrated                                                 |
| ----------------------------------- | ---------------------------------------------------------- |
| `README.md`                         | `README.md` (expanded for the integrated repo)             |
| `setup.py`                          | folded into `pyproject.toml`                               |
| `setup.sh`                          | `setup.sh` (unchanged); see also `scripts/bootstrap.sh`    |
| `env.yml`                           | `env.yml` (unchanged)                                      |
| `mosek.lic`                         | `mosek.lic` (unchanged)                                    |
| `.gitmodules`                       | merged + extended (more submodules under `third_party/`)   |
| `.mcp.json`                         | removed — MCP setup is via `xpu-rt mcp install`            |
| `.contributors-refresh`             | unchanged                                                  |
| `scheduled_qnn_island_demo.json`    | gitignored (regenerated when running the demo)             |
| `compgen_output/`                   | not migrated (was the legacy compiler-output staging dir)  |
| `paper/`, `plots/`, `sessions/`     | gitignored / not migrated (build/session artifacts)        |

---

## Renames (`CompGen*` → `XpuRt*`)

The integrated branch removed every `CompGen` / `compgen` identifier in
favour of `XpuRt` / `xpu_rt`. The most visible public API renames:

| Original symbol              | Renamed to                |
| ---------------------------- | ------------------------- |
| `CompGenDevice`              | `XpuRtDevice`             |
| `CompGenOptions`             | `XpuRtOptions`            |
| `CompGenLLMProtocol`         | `XpuRtLLMProtocol`        |
| `CompGenAdapter`             | `XpuRtAdapter`            |
| `CompGenBackend`             | `XpuRtBackend`            |
| `CompGenDriver`              | `XpuRtDriver`             |
| `CompGenError`               | `XpuRtError`              |
| `CompGenLauncher`            | `XpuRtLauncher`           |
| `CompGenPythonBackend`       | `XpuRtPythonBackend`      |
| `CompGenTorchEvalBackend`    | `XpuRtTorchEvalBackend`   |
| `CompgenAccel` / `CompgenAccelDialect` | `XpuRtAccel` / `XpuRtAccelDialect` |
| `CompgenOptError`            | `XpuRtOptError`           |
| `LiveCompGenAdapter`         | `LiveXpuRtAdapter`        |
| `COMPGEN_GEMINI_USAGE_DIR`   | `XPU_RT_GEMINI_USAGE_DIR` |
| `COMPGEN_OP_HANDLERS`        | `XPU_RT_OP_HANDLERS`      |
| `COMPGEN_SECTIONS`           | `XPU_RT_SECTIONS`         |
| `COMPGEN_CUSTOM`             | `XPU_RT_CUSTOM`           |
| `compgen-tracker` (pkg)      | removed entirely (was an external workspace dep)           |
| `compgen-mcp.sh`             | `xpu-rt-mcp.sh`           |
| Prose "CompGen" / "Compgen"  | "XPU-RT"                  |
| Lowercase identifier `compgen` | `xpu_rt`                |

---

## Quick-find: "I'm looking for…"

- **The CVX scheduler core** — `xpu_rt/scheduler/scheduler.py`.
- **QNN cost tables / island-DAG** — `xpu_rt/targets/backends/qnn/`.
- **Native C runtime** — `runtime/` (unchanged).
- **`run_xpurt_schedule.py` driver** — `tools/scheduling/run_xpurt_schedule.py`.
- **`qnn_island_demo.py`** — `tools/demos/qnn_island_demo.py`.
- **`heterogeneous_loop.py`** — `tools/scheduling/heterogeneous_loop.py`.
- **Profiling drivers (`profile_*.py`)** — `tools/profiling/`.
- **Merlin** — `third_party/merlin/` (submodule).
- **IsaacLab tasks** — `sims/isaaclab_tasks/` (local) +
  `third_party/IsaacLab/` (submodule).
- **QNN model-conversion scripts** — `models/qnn/`.

If a file you remember from the original XPU-RT is missing from the table
above, run `git log --follow --diff-filter=R -- '<old path>'` on this
branch to trace its move.
