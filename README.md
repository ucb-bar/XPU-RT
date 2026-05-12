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

```bash
conda env create -f merlin/env_linux.yml
conda activate merlin-dev
uv sync
```

```bash
bash setup.sh
```
```bash
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
├── data/                      # Collected benchmark/profile/scheduling outputs
├── merlin/                    # Git submodule (compiler/runtime/tooling upstream)
├── env.yml                    # Conda environment
└── setup.py                   # Editable pip install config
```


### File-Level Integration Points

2. `runtime/scripts/compile_all_models.sh` -> compiles models under `merlin/models/...`

### Data/Artifact Flow Between This Repo and `merlin`

1. `runtime/scripts/compile_all_models.sh` generates VMFB + graph artifacts into `gen/vmfb/...` (using Merlin compiler).
2. `runtime/scripts/profile_remote.sh` runs topology benchmarks remotely and writes CSV results to `gen/profile/...`.
3. `scripts/run_xpurt_schedule.py` reads profiled CSVs from `gen/profile/...` and combines them with dispatch graph JSON inputs to produce schedules.
4. Final scheduling outputs and logs are stored under `data/...` and script output directories.

## Notes

1. The Python scheduler modules are sourced from `xpu-rt/*.py` and installed via `setup.py`.
2. Runtime C tooling in `runtime/` is separate from Python scheduling code and is focused on Merlin/IREE integration.
3. If submodule contents are missing, runtime build/profile scripts will fail early.
