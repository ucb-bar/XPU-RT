# XPU-RT Scheduling and Runtime Integration

## Project Description

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.

This project is under active development. If you would love to contribute or if you find any issues, please do so by opening a [pull request](https://github.com/ucb-bar/XPURT/pulls) or [filing an issue](https://github.com/ucb-bar/XPURT/issues) on GitHub.

### Documentation

* [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md)
  — full walkthrough from a multi-network workload spec to a FireSim
  run with trace plots (scheduler → codegen → build → run → analyze),
  on the Saturn-Gemmini-Q31 path.
* [`zephyr-chipyard-sw/agents/examples/microros_demo/ROS_FLOW.md`](zephyr-chipyard-sw/agents/examples/microros_demo/ROS_FLOW.md)
  — micro-ROS fixed-pinning baseline flow (the reference against which
  the scheduler is benchmarked).
* The sections below cover the Merlin/SpacemiT path (BananaPi).

### Repository Initialization

```bash
git clone https://github.com/ucb-bar/XPU-RT.git
cd XPU-RT
git submodule update --init --recursive
```

### Set up `merlin`

Merlin provides the compiler toolchain, IREE runtime, and cross-compilation
support used by XPU-RT. It ships as a git submodule under `merlin/`.

#### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Mamba)
- Internet access for initial setup (toolchain downloads, submodule clones)

#### 1) Install Environment

```bash
conda env create -f merlin/env_linux.yml
conda activate merlin-dev
uv sync
```

#### 2) Build compiler tools, target runtime and toolchain

The one-step setup script handles everything (toolchain + host compiler + target runtime):

```bash
bash setup.sh
```

Or run merlin commands individually (from within the `merlin/` directory):

```bash
cd merlin

# Install SpacemiT cross-compilation toolchain
uv run tools/merlin.py setup toolchain --toolchain-target spacemit

# Build host compiler tools
uv run tools/merlin.py build --profile vanilla --config release

# Build SpacemiT target runtime (includes xpu-rt plugin)
uv run tools/merlin.py build --profile spacemit --config perf

cd ..
```

See `merlin/docs/getting_started.md` and `merlin/docs/reference/cli.md` for the
full Merlin CLI reference.

#### 3) Compile models and profile on target

After building merlin, run these from the XPU-RT root:

```bash
runtime/scripts/compile_all_models.sh   # compile VMFB files for all models
runtime/scripts/profile_remote.sh       # profile on target device (e.g. BananaPi)
```

For runtime-specific setup and usage details, see [runtime/README.md](runtime/README.md).

#### Developer note: using a separate merlin checkout

During active merlin development, you can use a standalone merlin checkout
instead of the submodule. Either symlink it:

```bash
rm -rf merlin && ln -s /path/to/your/merlin merlin
```

Or set the `MERLIN_DIR` environment variable (respected by `setup.sh`,
`compile_all_models.sh`, and `build_runtime.sh`):

```bash
export MERLIN_DIR=/path/to/your/merlin
bash setup.sh
```

#### Config your model parameters
Create `data/toplevel/networks_periodic_profile.json` if there is none, and add entries like:

```text
"mlp": {
  "id": 1,
  "identifier":            # model name
  "dispatch_deps_path":    # path to model json 
  "period":                # Duration in millisec between excution windows (inverse of frequency)
  "window_duration":       # Duration in millisec for model to finish after window start 
}
```

### Run XPU-RT Scheduler
Run basic demos on top-level network graph:

```bash
python scripts/run_xpurt_schedule.py --profiled
```
The optimal schedule of your workloads on your target will be found in `schedules/scheduled_networks_periodic_profiled.json` with visualization in 'plots/iree_combined_schedule_period.png' after it finishes.


### [Optional] Run Baseline greedy Scheduler

Run greedy scheduler variant:

```bash
python scripts/run_greedy_schedule.py --use-grouped
```

## Repository Map

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
├── runtime/                   # Scripts for compile/profile flow + optional custom tools
│   ├── scripts/*.sh           #   compile_all_models, profile_remote, etc.
│   └── tools/                 #   Custom tool sources (links merlin's xpu-rt archive)
├── data/                      # Collected benchmark/profile/scheduling outputs
├── merlin/                    # Git submodule (compiler/runtime/tooling upstream)
│   ├── tools/merlin.py        #   Unified CLI: build, compile, setup, benchmark, ...
│   ├── samples/common/xpu-rt/ #   XPU-RT runtime library (baseline + scheduler runners)
│   ├── samples/SpacemiTX60/   #   SpacemiT-specific sample binaries
│   └── models/                #   Model definitions (MLIR/ONNX sources)
├── env.yml                    # Conda environment
└── setup.py                   # Editable pip install config
```


### File-Level Integration Points

1. `runtime/scripts/compile_all_models.sh` -> calls `merlin/tools/merlin.py compile`
2. `runtime/scripts/compile_all_models.sh` -> compiles models under `merlin/models/...`
3. Pre-built runner binaries come from `merlin/build/<profile>/runtime/plugins/merlin-samples/`:
   - `merlin-baseline-async` — baseline topo-order dispatch runner
   - `merlin-dispatch-scheduler` — two-cluster scheduled dispatch runner
4. XPU-RT runtime C API headers: `merlin/samples/common/xpu-rt/*.h`
5. Standalone archive for custom tools / Zephyr:
   `merlin/build/<build-name>/runtime/src/iree/runtime/libxpurt_standalone.a`

### Data/Artifact Flow Between This Repo and `merlin`

1. `runtime/scripts/compile_all_models.sh` generates VMFB + graph artifacts into `gen/vmfb/...` (using Merlin compiler).
2. `runtime/scripts/profile_remote.sh` runs topology benchmarks remotely and writes CSV results to `gen/profile/...`.
3. `scripts/run_xpurt_schedule.py` reads profiled CSVs from `gen/profile/...` and combines them with dispatch graph JSON inputs to produce schedules.
4. Final scheduling outputs and logs are stored under `data/...` and script output directories.

## Notes

1. The Python scheduler modules are sourced from `xpu-rt/*.py` and installed via `setup.py`.
2. Runtime C tooling in `runtime/` is separate from Python scheduling code and is focused on Merlin/IREE integration.
3. If submodule contents are missing, runtime build/profile scripts will fail early.
