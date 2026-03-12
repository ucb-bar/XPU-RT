# XPU-RT Scheduling and Runtime Integration

## Project Description

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.

Current project scope:

1. Adaptable full stack: enables E2E compilation and mapping of robotic multi-model graphs to heterogeneous shared-memory SoCs, while exposing extension points for new compilation/runtime capabilities.
2. Optimal AOT scheduler: integrates with IREE/MLIR-oriented flows to generate a pre-scheduled execution plan from computation graph structure and profiled signals (for example latency and energy), then statically maps operators to target compute resources.
3. Robotic timing model: supports robot-specific scheduling semantics where periodic tasks are treated as hard real-time deadline-driven workloads and non-periodic tasks are handled as soft real-time workloads tied to QoE-style objectives.
4. Hardware-in-the-loop and static profiling: supports hardware-informed mapping through closed-loop profiling and monitoring (for example perf, LLVM-MCA, and environment/hardware feedback).
5. Runtime and hardware mechanisms: focuses on synchronization and data-movement-aware execution support for efficient operator dispatch, monitoring, and coordination.
6. Performance validation goal: emphasizes RTL-level and hardware-evaluated improvements that outweigh scheduler/runtime overhead.

This repository contains:

1. A Python scheduling stack for multi-core dispatch scheduling experiments.
2. Runtime tooling that integrates with the Merlin/IREE runtime artifacts.
3. Data, scripts, and benchmark flows for end-to-end compile -> profile -> schedule.

## Example Usage (Robotics System Context)

In robotic deployments, different model pipelines run at different required frequencies and with different criticality. This repository is structured around that constraint:

1. Build and profile dispatch-level execution behavior from real hardware/toolchain artifacts.
2. Generate static schedules that respect periodic control/perception timing requirements.
3. Co-schedule non-periodic workloads to maximize utilization while keeping quality metrics and responsiveness acceptable.
4. Feed profiling and hardware observations back into future mapping/scheduling decisions.

## Repository Initialization

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd XPU-RT
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

Create Python environment (recommended):

```bash
conda env create -f env.yml
conda activate schedule
```

Install this repo in editable mode:

```bash
python -m pip install -e .
```

## Quick Start Commands

Run basic demos:

```bash
python scripts/testing.py
python scripts/packing_demo.py
python scripts/additional_obj_demo.py
```

Run hierarchical scheduler on top-level network graph:

```bash
python scripts/run_xpurt_schedule.py --profiled
```

Run greedy scheduler variant:

```bash
python scripts/run_greedy_schedule.py --use-grouped
```

## Directory Hierarchy

```text
XPU-RT/
├── xpu-rt/                    # Python scheduler core modules
│   ├── scheduler.py
│   ├── workload.py
│   ├── workload_factory.py
│   ├── packing.py
│   ├── plot.py
│   ├── schedule_validation.py
│   └── pytorch_workload/      # Sample model artifacts + dispatch JSON inputs
├── scripts/                   # Python entry points for experiments/scheduling
├── runtime/                   # C runtime tool + scripts for compile/profile flow
│   ├── tools/json_dispatch_runner.c
│   └── scripts/*.sh
├── data/                      # Collected benchmark/profile/scheduling outputs
├── merlin/                    # Git submodule (compiler/runtime/tooling upstream)
├── env.yml                    # Conda environment
└── setup.py                   # Editable pip install config
```

## Connection to the `merlin` Submodule

`merlin` is a required submodule defined in `.gitmodules`. This repo depends on it for model compilation and runtime static libraries.

### File-Level Integration Points

1. `runtime/scripts/compile_all_models.sh` -> calls `merlin/tools/compile.py`
2. `runtime/scripts/compile_all_models.sh` -> compiles models under `merlin/models/...`
3. `runtime/CMakeLists.txt` -> includes header from `merlin/xpu-rt/xpurt_scheduler_core.h`
4. `runtime/build_runtime.sh` -> links against Merlin-produced archive:
   `merlin/build/<build-name>/runtime/src/iree/runtime/libxpurt_iree_plugin_standalone.a`
5. `runtime/build_runtime.sh` -> can trigger Merlin target build:
   `xpurt_iree_plugin_standalone`

### Data/Artifact Flow Between This Repo and `merlin`

1. `runtime/scripts/compile_all_models.sh` generates VMFB + graph artifacts into `gen/vmfb/...` (using Merlin compiler).
2. `runtime/scripts/profile_remote.sh` runs topology benchmarks remotely and writes CSV results to `gen/profile/...`.
3. `scripts/run_xpurt_schedule.py` reads profiled CSVs from `gen/profile/...` and combines them with dispatch graph JSON inputs to produce schedules.
4. Final scheduling outputs and logs are stored under `data/...` and script output directories.

## Notes

1. The Python scheduler modules are sourced from `xpu-rt/*.py` and installed via `setup.py`.
2. Runtime C tooling in `runtime/` is separate from Python scheduling code and is focused on Merlin/IREE integration.
3. If submodule contents are missing, runtime build/profile scripts will fail early.
