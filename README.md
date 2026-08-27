# XPU-RT Scheduling and Runtime Integration

## Project Description

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.

This project is under active development. If you would love to contribute or if you find any issues, please do so by opening a [pull request](https://github.com/ucb-bar/XPURT/pulls) or [filing an issue](https://github.com/ucb-bar/XPURT/issues) on GitHub.

## Framework model: bring your own compiler

The scheduler and runtime core (`xpu-rt/`, `runtime/`) are **plug-and-play** —
compiler- and codegen-agnostic. Anything that can (a) emit a per-op profile in
the IREE dispatch-shape CSV schema and (b) build a single binary that dispatches
per-op kernels according to the schedule's core assignment can sit on the
"compiler" side of the flow. Two integrations exist today:

| flow | compiler / codegen | target | profiling | docs |
|---|---|---|---|---|
| **A — ModelBlaster** | PyTorch → quantized Zephyr/RISC-V; curated + LLM-agentic kernel-gen | chipyard (Saturn/Gemmini, RISC-V) | spike / FireSim | [Flow A section below](#flow-a-modelblaster-as-the-compiler-backend), [`zephyr-chipyard-sw/modelblaster/README.md`](zephyr-chipyard-sw/modelblaster/README.md) ("Workflow: integrating with XPURT"), [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md) |
| **B — merlin** *(this README)* | merlin → IREE → VMFB | SpacemiT (BananaPi) | on-device, via `profile_remote.sh` | sections below |

Both flows feed the same `xpu-rt/scheduler.py` and read/write the same
`gen/profile/.../results.csv` + `schedules/*.json` shapes — the compiler and
target hardware are the only things that change. Flow B (merlin as the
compiler backend) is documented in the sections immediately below; Flow A
(ModelBlaster) has its own walkthrough further down, in
["Flow A: ModelBlaster as the compiler backend"](#flow-a-modelblaster-as-the-compiler-backend).
Flow A brings its own compiler (ModelBlaster's codegen pipeline, including an
LLM-driven kernel generator) and its own Zephyr/chipyard build+run path
instead of merlin/IREE.

### Documentation

* [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md)
  — full walkthrough from a multi-network workload spec to a FireSim
  run with trace plots (scheduler → codegen → build → run → analyze),
  on the Saturn-Gemmini-Q31 path (Flow A).
* [`zephyr-chipyard-sw/agents/examples/microros_demo/ROS_FLOW.md`](zephyr-chipyard-sw/agents/examples/microros_demo/ROS_FLOW.md)
  — micro-ROS fixed-pinning baseline flow (the reference against which
  the scheduler is benchmarked).
* The sections below cover the Merlin/SpacemiT path (BananaPi) — Flow B.

## Flow B: merlin as the compiler backend

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

## Flow A: ModelBlaster as the compiler backend

ModelBlaster brings PyTorch → quantized Zephyr/RISC-V codegen (curated
kernels + an LLM-agentic kernel generator) instead of merlin/IREE, and
profiles on spike/FireSim instead of a physical BananaPi. It plugs into the
same `xpu-rt/scheduler.py` as Flow B — only the compiler and target change.

### Repository layout

ModelBlaster ships as a git submodule **nested inside `zephyr-chipyard-sw`**
(its canonical location — the same one the standalone xpu-rt flow uses), not at
the top level:

```bash
git submodule update --init --recursive zephyr-chipyard-sw   # pulls in modelblaster (+ KernelBlaster)
```

```text
XPU-RT/                (this repo)
├── merlin/                        submodule — Flow B compiler
└── zephyr-chipyard-sw/            submodule — Zephyr BSP + samples
    └── modelblaster/              submodule — Flow A compiler
```

ModelBlaster's own scripts (`scripts/run_xpurt_scheduler*.py`,
`benchmarks/runners/firesim.py`, `examples/xpurt_demo/run.sh`, ...) default to
finding XPU-RT as a **sibling** checkout (`XPURT_ROOT` defaults to
`../XPU-RT`) — that assumption predates the submodule and no longer holds
once ModelBlaster is nested *inside* XPU-RT. Set `XPURT_ROOT` to the XPU-RT root
explicitly when working from the submodule:

```bash
export XPURT_ROOT="$(cd ../.. && pwd)"   # run from inside zephyr-chipyard-sw/modelblaster
```

(No `pip install` of the `xpurt` package is required either way — the
bridge scripts import `xpu-rt/*.py` straight off the path `XPURT_ROOT`
resolves to.)

### 1) Profile each (model, backend) pair

From the ModelBlaster submodule, profile every model/backend combination this
workload needs on spike or FireSim — this is what fills in the per-op cycle
data the scheduler bridge reads in step 2:

```bash
cd zephyr-chipyard-sw/modelblaster
QUANT=int8 TARGET=rvv        RUNNER=firesim bash examples/dronet/run.sh
QUANT=int8 TARGET=gemmini_q31 RUNNER=firesim bash examples/dronet/run.sh
# ...one run per (model, backend) pair in the workload
```

### 2) Run the XPU-RT scheduler bridge

ModelBlaster ships two scheduler bridge scripts that import `xpu-rt/scheduler.py`
straight off this checkout (via `XPURT_ROOT`) and solve with MOSEK through cvxpy
— the same MILP as Flow B's `scripts/run_xpurt_schedule.py`, just invoked from
the ModelBlaster side:

```bash
cd zephyr-chipyard-sw/modelblaster
export XPURT_ROOT="$(cd ../.. && pwd)"

# single hetero workload
PYTHONPATH=. uv run python -m scripts.run_xpurt_scheduler \
    --workload dronet_hetero_int8 \
    --target-backends gemmini,rvv_opu \
    --runner firesim \
    --output schedule_fixtures/dronet_xpurt_mosek.json

# multi-network workload (YAML spec of networks + instance counts)
PYTHONPATH=. uv run python -m scripts.run_xpurt_scheduler_multi \
    --config configs/multi_3way_qrb.yaml \
    --output schedule_fixtures/3way_mosek_qrb.json
```

Both require a MOSEK license + cvxpy in the interpreter that runs them — see
`XPURT_PYTHON` below.

### 3) Build and run the scheduled binary

```bash
SCHEDULE_JSON=$PWD/schedule_fixtures/dronet_xpurt_mosek.json \
MODELS=dronet,mlp_control \
BACKENDS=scalar,rvv \
QUANT=int8 \
RUNNER=firesim \
XPURT_TRACE=1 \
bash examples/xpurt_demo/run.sh
```

`xpurt_demo/run.sh` links one object per (model × backend) and dispatches
each schedule entry to the right one. With `XPURT_TRACE=1`, the uartlog
carries per-entry begin/end timestamps that ModelBlaster's
`scripts/plot_xpurt_trace.py` renders as a Gantt chart against the predicted
timeline.

### Env vars ModelBlaster uses to find this checkout

| var | default | used by |
|---|---|---|
| `XPURT_ROOT` | `../XPU-RT` (a **sibling-checkout default** — override to `../..` when running from `zephyr-chipyard-sw/modelblaster`) | `scripts/run_xpurt_scheduler.py`, `scripts/run_xpurt_scheduler_multi.py`, `scripts/find_min_periodic_makespan*.py`, `benchmarks/runners/firesim.py`, `examples/xpurt_demo/run.sh` |
| `XPURT_PYTHON` | the `xpu-rt-schedule` conda env (derived from `CONDA_EXE`), else `python3` | `scripts/find_min_periodic_makespan_mosek.py` (needs cvxpy + MOSEK) |

Unlike `merlin`, this submodule reference *is* pinned to a commit (standard
submodule semantics). Because it is nested, bumping it means updating
`modelblaster` inside `zephyr-chipyard-sw`, committing that, then bumping the
`zephyr-chipyard-sw` pointer in this repo.

For the full ModelBlaster-side workflow (profiling knobs, workload JSON
schema, models in scope), see
[`zephyr-chipyard-sw/modelblaster/README.md`](zephyr-chipyard-sw/modelblaster/README.md), section
"Workflow: integrating with XPURT."

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
├── merlin/                    # Git submodule (compiler/runtime/tooling upstream) — Flow B
│   ├── tools/merlin.py        #   Unified CLI: build, compile, setup, benchmark, ...
│   ├── samples/common/xpu-rt/ #   XPU-RT runtime library (baseline + scheduler runners)
│   ├── samples/SpacemiTX60/   #   SpacemiT-specific sample binaries
│   └── models/                #   Model definitions (MLIR/ONNX sources)
├── zephyr-chipyard-sw/          # Git submodule — Zephyr BSP + samples
│   └── modelblaster/            #   nested submodule (PyTorch->Zephyr/RISC-V pipeline) — Flow A
│       └── third_party/KernelBlaster/  # nested submodule — originating research project
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

### Data/Artifact Flow Between This Repo and `ModelBlaster` (Flow A)

ModelBlaster is a submodule nested in `zephyr-chipyard-sw/modelblaster` — but its
own scripts still reach back into XPU-RT via the `XPURT_ROOT`/`MERLIN_DIR` env
vars and a `[tool.uv.sources]` entry rather than a relative import, so
`XPURT_ROOT` needs to be set to `../..` (not left at its sibling-checkout
default) when running from inside the submodule. See
["Flow A: ModelBlaster as the compiler backend"](#flow-a-modelblaster-as-the-compiler-backend)
above.

1. ModelBlaster profiles each (model, backend) pair on spike/FireSim and emits
   an IREE-shape `results.csv` (same schema `xpu-rt/profile_loader.py` expects
   from the merlin path).
2. `xpu-rt/scheduler.py` (imported live from this checkout via `XPURT_ROOT`)
   reads those CSVs and computes a core-assignment schedule, same as Flow B.
3. ModelBlaster's `examples/xpurt_demo/run.sh` builds a single Zephyr ELF from
   that schedule and runs it via `harness_xpurt/` — the chipyard/Zephyr
   equivalent of merlin's VMFB dispatch runners.

## Feedback-driven compilation: post-schedule granularity advisor

Motivating case: you partition a model at the compiler's default dispatch
granularity, profile it (Flow A or B), and the schedule xpu-rt computes still
misses its deadline. Often that's because a **non-periodic** (best-effort)
job got scheduled as one coarse, unfused dispatch that occupies a core far
longer than a **periodic** job's period — if the two ever share a core, that
one coarse dispatch blows through several periodic deadlines before
yielding. xpu-rt can't fix this itself: its only granularity lever
(`fusion_threshold` in `scheduler.schedule()`, via `xpu-rt/fusion.py`) merges
small dispatches into bigger ones — nothing here splits a coarse dispatch
into finer ones. That has to happen upstream, in whatever compiler produced
the dispatch graph (e.g. ModelBlaster's Model Partitioner / LLM-agentic
codegen). So `xpu-rt/granularity_advisor.py` is **advisory only**: it
compares each non-periodic job's worst-case dispatch duration against the
tightest **free slot** among periodic jobs in the same schedule — a periodic
job's period adjusted for how much of it its own dependency-chain critical
path actually occupies, not the raw period (a periodic job running close to
its own deadline can leave far less free room than its period alone
suggests) — and flags a mismatch, gating any "coarser" recommendation on the
job's dispatches actually forming a linear chain (the same shape
`xpu-rt/fusion.py`'s own fusion pass requires). A signal a human, or an
upstream optimizer, can act on.

Two ways to get the signal:
- **Inline**, every time `scripts/run_xpurt_schedule.py` runs: it's printed
  as a `WARN:` line, and also embedded in the output JSON's
  `metadata["granularity_advice"]` (plus `metadata["periodic_networks"]`,
  the inferred per-network periods) — no extra step required.
- **Retroactively**, against any already-saved schedule JSON (including
  ones from before this feature existed):
  ```bash
  python scripts/analyze_granularity.py schedules/scheduled_networks_deps_4cores_profiled.json
  ```
  Older files fall back to inferring periodicity from dispatch-key naming
  (`<instance>_dispatch_<n>` — e.g. `dronet0`, `dronet1`, ... share base
  `dronet`) rather than reading it from metadata that didn't exist yet when
  they were written; see the module docstring in `granularity_advisor.py`
  for the precision trade-off that implies.

## Notes

1. The Python scheduler modules are sourced from `xpu-rt/*.py` and installed via `setup.py`.
2. Runtime C tooling in `runtime/` is separate from Python scheduling code and is focused on Merlin/IREE integration.
3. If submodule contents are missing, runtime build/profile scripts will fail early.
