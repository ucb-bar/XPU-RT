# XPU-RT Scheduling and Runtime Integration

## Project Description

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.

### Repository Initialization

```bash
git clone https://github.com/ucb-bar/XPU-RT.git
cd XPU-RT
```
```bash
git submodule update --init --recursive
```

### Set up `merlin` Submodule
#### 1) Intall Environment
```bash
conda env create -f merlin/env_linux.yml
conda activate merlin-dev
uv sync
```

#### 2) Build compiler tools, target runtime and toolchain for your device target 
```bash
bash setup.sh
```
After installing merlin, run the following scripts in XPU-RT:
```bash
runtime/scripts/compile_all_models.sh # build all the vmfb files for your models
runtime/scripts/profile_remote.sh # run on your device target i.e. spacemit_x60 for banana pi
```

For runtime-specific setup and usage details, see [runtime/README.md](runtime/README.md).

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
├── runtime/                   # C runtime tool + scripts for compile/profile flow
│   ├── tools/json_dispatch_runner.c
│   └── scripts/*.sh
├── data/                      # Collected benchmark/profile/scheduling outputs
├── merlin/                    # Git submodule (compiler/runtime/tooling upstream)
├── env.yml                    # Conda environment
└── setup.py                   # Editable pip install config
```


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
