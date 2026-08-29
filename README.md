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
| **A — chipyard** | PyTorch → quantized Zephyr/RISC-V; curated + LLM-agentic kernel-gen | chipyard (Saturn/Gemmini, RISC-V) | spike / FireSim | [Flow A section below](#flow-a-modelblaster-as-the-compiler-backend), [`zephyr-chipyard-sw/modelblaster/README.md`](zephyr-chipyard-sw/modelblaster/README.md) ("Workflow: integrating with XPURT"), [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md) |
| **B — SpaceMiT K1** | the same ModelBlaster codegen, cross-compiled for Linux/riscv64 | SpaceMiT K1 (BananaPi), 8 harts + IME | on-device, over ssh | [Flow B section below](#flow-b-modelblaster-on-the-spacemit-k1-board), [`docs/the_loop.md`](docs/the_loop.md) |

Both flows feed the same `xpu-rt/scheduler.py` and read/write the same
`gen/profile/.../results.csv` + `schedules/*.json` shapes — the target hardware
and the runtime around the kernels are the only things that change.

**Flow B used to be a different compiler**: merlin → IREE → VMFB. It is not any
more. That path is retired, the merlin submodule is gone, and both flows now
build from ModelBlaster — so the two entries above differ in where the code
runs, not in what generates it. The `results.csv` schema is IREE-shaped for the
same reason a road can keep a Roman route: every reader already speaks it.

### Start here

* **[`docs/the_loop.md`](docs/the_loop.md)** — the index: every arrow of the
  compiler↔scheduler cycle and which script owns it. If you have been away,
  read this one.
* **[`docs/environment.md`](docs/environment.md)** — recreating the
  environment. Two flows, two environments, and neither is merlin's `.venv`.
* **[`docs/k1_board.md`](docs/k1_board.md)** — running on the K1: the
  commands, the timings, the two compiler traps, and what to do when it
  breaks.
* **[`examples/`](examples/)** — runnable, one per topic:
  `.venv/bin/python examples/run_all.py`

### Documentation

* [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md)
  — full walkthrough from a multi-network workload spec to a FireSim
  run with trace plots (scheduler → codegen → build → run → analyze),
  on the Saturn-Gemmini-Q31 path (Flow A).
* [`docs/mlp_dronet_yolo_spike_reproduction.md`](docs/mlp_dronet_yolo_spike_reproduction.md)
  — a simpler, no-FireSim variant of Flow A: same ModelBlaster codegen and
  checkout (`zephyr-chipyard-sw/modelblaster/`), profiled entirely on
  spike with the `greedy`/`greedy_periodic` solver (no MOSEK license
  needed). Driven by the same one-command script as Flow A,
  `scripts/repro_workload.sh <spec.json>`, which installs everything it
  needs via `scripts/install_xpurt_deps.sh` — see that doc's §0.
* [`zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md`](zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md)
  — micro-ROS fixed-pinning baseline flow (the reference against which
  the scheduler is benchmarked).
* [`docs/the_loop.md`](docs/the_loop.md)
  — the K1 board loop end to end (Flow B), and which script owns each arrow.

## Flow B: ModelBlaster on the SpaceMiT K1 board

Same compiler as Flow A, different target and runtime: ModelBlaster's
generated C, cross-compiled for **Linux/riscv64** and run on a SpaceMiT K1
(BananaPi) over ssh, rather than Zephyr on chipyard.

This flow used to be merlin -> IREE -> VMFB. It is not any more. Every kernel
that runs on this board now comes out of ModelBlaster's curated tree, the
merlin submodule is gone, and `runtime/` keeps only the four board scripts.
The one thing the live path still needed from merlin -- the SpaceMiT cross
toolchain -- is now fetched by `scripts/setup_spacemit_toolchain.sh`.

### 0) The toolchain, first, every time

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"     # exports CROSS
```

**Not optional.** GCC 13.2 -- what `CROSS` defaults to via chipyard's
riscv-tools -- reorders the RVV `vsetvl` intrinsics so a widening instruction
runs under the narrow vtype, and the board binary SIGILLs with no stdout at
all. The script refuses anything below 14.

GCC 14.3 has the opposite trap: it substitutes a wrong AVL on a *chained*
`vsetvl`, which is silent rather than loud. Pass the element count to every
width, and run `ModelBlaster/scripts/check_rvv_avl.py`.

### 1) Profile each (model, backend) pair on the board

```bash
PROFILE_OUT_ROOT=$PWD/gen_mb/profile \
  bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0
```

Correctness is not a separate step: `run_model_k1.sh` golden-compares in-binary
on every run. The profile lands as an IREE-shaped `results.csv` -- the schema
outlived the IREE path, because it is what `xpu-rt/profile_loader.py` reads.

`MB_CORES` drives the multi-core runs, and derives the worker-pool width, the
affinity mask and the profile's `topo_` tag from one place, so a run's tag
cannot disagree with the cores it actually used:

```bash
MB_CORES=0,1,2,3 ITERS=7 \
  bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0
```

The board has two 4-core L2 clusters: `CPU_P` is harts 0-3, `CPU_E` is 4-7.
**IME (`smt.vmadot`) exists only on cluster 0** -- an `ime` dispatch placed on
CPU_E SIGILLs, which is why `scripts/check_schedule_feasibility.py` refuses
that schedule before it is ever deployed.

### 2) Schedule

```bash
python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json
```

### 3) Build and run the scheduled binary

```bash
bash ModelBlaster/scripts/run_xpurt_k1.sh <schedule.json>
```

This emits a measured per-dispatch trace. `scripts/plot_k1_trace_gantt.py`
renders it, `scripts/join_k1_trace.py` joins it against the prediction, and
`scripts/compare_candidates.py` turns two solved schedules into an
accept/reject verdict with the term that decided it.

### Two board measurements worth reading before quoting

* [`docs/k1_contention.md`](docs/k1_contention.md) -- do concurrent dispatches
  slow each other down? Measured **null**: the distributions overlap and the
  arms are not monotonic in co-runner count.
* [`docs/k1_cost_by_pred.md`](docs/k1_cost_by_pred.md) -- what it costs to read
  what the previous dispatch wrote, from elsewhere. About 6% off-hart, 10%
  cross-cluster, and it is a model fitted to three measured classes rather than
  64 independent measurements.

### The whole loop

[`docs/the_loop.md`](docs/the_loop.md) is the index: profile -> schedule ->
advice -> hint -> rewrite -> verify -> reprofile -> verdict, and which script
owns each arrow.


## Flow A: ModelBlaster as the compiler backend

ModelBlaster brings PyTorch → quantized Zephyr/RISC-V codegen (curated
kernels + an LLM-agentic kernel generator) instead of merlin/IREE, and
profiles on spike/FireSim instead of a physical BananaPi. It plugs into the
same `xpu-rt/scheduler.py` as Flow B — only the compiler and target change.

### Repository layout

ModelBlaster ships as a git submodule **nested inside `zephyr-chipyard-sw`**
(its canonical location — the same one the spike-only reproduction flow
uses), not at the top level:

```bash
git submodule update --init --recursive zephyr-chipyard-sw   # pulls in modelblaster (+ KernelBlaster)
```

```text
XPU-RT/                          (this repo)
├── ModelBlaster/                submodule — the compiler, for BOTH flows
└── zephyr-chipyard-sw/          submodule — Zephyr BSP + samples
    └── modelblaster/            submodule — the same repo, same commit
```

**Two paths, one repo, and they should always name the same commit.**
ModelBlaster is reachable as XPU-RT's own top-level submodule (Flow B, and
what `scripts/install_xpurt_deps.sh` prefers) and again through
`zephyr-chipyard-sw` (Flow A's spike/firesim builds). Two checkouts of one
upstream at *different* commits means the two flows compile different kernels
from the same op names, with nothing to say so — so when you bump one, bump
the other. `git submodule update --init ModelBlaster` is enough for Flow B on
its own; an uninitialised submodule is an empty directory, not an error, which
is exactly how that goes unnoticed.

ModelBlaster's own scripts (`scripts/run_xpurt_scheduler*.py`,
`benchmarks/runners/firesim.py`, `examples/xpurt_demo/run.sh`, ...) default to
finding XPU-RT as a **sibling** checkout (`XPURT_ROOT` defaults to
`../XPU-RT`) — that assumption predates the submodule and no longer holds
once ModelBlaster is nested *inside* XPU-RT (two levels deep, inside
`zephyr-chipyard-sw`). Set `XPURT_ROOT` to the XPU-RT root explicitly when
working from the submodule:

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
the ModelBlaster side.

**Deps:** install the `milp` extra into the same `zephyr` conda env used for
everything else in this repo, from the top-level XPU-RT checkout:

```bash
pip install -e ".[milp]"   # adds cvxpy (the modeling layer) to the zephyr env
```

This is enough to exercise the scheduler bridge end to end against cvxpy's
free solvers (`--solver CLARABEL`, `SCS`, `HIGHS`, `OSQP`, `SCIPY`). **MOSEK
itself** — the solver these scripts default to (`--solver MOSEK`) — is a
separate, license-gated product: `pip install mosek` adds the Python
package (no license needed just to install it), but actually solving
requires a license file (`MOSEKLM_LICENSE_FILE`) from mosek.com.

(modelblaster's own `pyproject.toml` also declares a `scheduler` extra meant
for `uv sync --extra scheduler` + `uv run` — currently broken for this
nested-submodule layout: `uv.lock` resolution pulls in every
`[tool.uv.sources]` entry regardless of which extra you sync, including an
unrelated `smolvla`-extra path (lerobot) that isn't
checked out by default. Plain `python3` in the `zephyr` env, as below, is
the reliable path today.)

```bash
cd zephyr-chipyard-sw/modelblaster
export XPURT_ROOT="$(cd ../.. && pwd)"

# single hetero workload
PYTHONPATH=. python3 -m scripts.run_xpurt_scheduler \
    --workload dronet_hetero_int8 \
    --target-backends gemmini,rvv_opu \
    --runner firesim \
    --output schedule_fixtures/dronet_xpurt_mosek.json

# multi-network workload (YAML spec of networks + instance counts)
PYTHONPATH=. python3 -m scripts.run_xpurt_scheduler_multi \
    --config configs/multi_3way_qrb.yaml \
    --output schedule_fixtures/3way_mosek_qrb.json
```

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

This submodule reference is pinned to a commit (standard submodule
semantics). Because it is nested, bumping it means updating
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
├── runtime/                   # K1 board scripts (Flow B)
│   └── scripts/               #   deploy_k1, verify_ime_build, contention, cost_by_pred
├── data/                      # Collected benchmark/profile/scheduling outputs
├── tools/                     # Fetched artifacts (cross toolchain) — gitignored
├── ModelBlaster/               # Git submodule — the compiler, for BOTH flows
├── zephyr-chipyard-sw/         # Git submodule — Zephyr BSP + samples
│   └── modelblaster/           #   the SAME repo again, and it should be the same commit
│       └── third_party/KernelBlaster/  # nested submodule — originating research project
├── env.yml                     # cvxpy+MOSEK conda env ("xpu-rt-schedule") for
                                 #   ModelBlaster's own MOSEK bridge scripts (Flow A)
└── pyproject.toml              # xpu-rt's own deps (`pip install -e .`); see
                                 #   scripts/install_xpurt_deps.sh for the
                                 #   spike-only reproducible-flow's dependency set
```


### Data/Artifact Flow

1. **Profile.** ModelBlaster's `run_model_k1.sh` (Flow B, on the board) or the
   spike/firesim runners (Flow A) write an IREE-shaped `results.csv` under
   `gen_mb/profile/<impl>/<target>/<model>/<basename>/<topo_tag>/`. The
   `topo_tag` records which harts the run used, derived from `MB_CORES` in the
   same place as the pool width and the affinity mask, so a profile cannot
   claim a core count it did not run on.
2. **Schedule.** `scripts/run_xpurt_schedule.py` reads those CSVs plus the
   dispatch-graph JSON and writes `schedules/scheduled_*.json`.
3. **Run.** The scheduled binary emits a per-dispatch trace CSV.
   `scripts/join_k1_trace.py` joins it against the prediction;
   `scripts/plot_k1_trace_gantt.py` renders it.
4. **Adjudicate.** `scripts/emit_compile_advice.py` turns the measurement into
   advice, the `advice_to_*_hint.py` bridges turn advice into something
   ModelBlaster's `apply_*_hint.py` will accept, and
   `scripts/compare_candidates.py` scores the rewritten graph against the
   baseline — nine lexicographic terms, hard deadline misses first, standalone
   kernel cycles last.

`docs/the_loop.md` is the index for all of it.

### Data/Artifact Flow Between This Repo and `ModelBlaster` (Flow A)

ModelBlaster is a submodule nested in `zephyr-chipyard-sw/modelblaster` — but its
own scripts still reach back into XPU-RT via the `XPURT_ROOT` env var and a
`[tool.uv.sources]` entry rather than a relative import, so
`XPURT_ROOT` needs to be set to `../..` (not left at its sibling-checkout
default) when running from inside the submodule. See
["Flow A: ModelBlaster as the compiler backend"](#flow-a-modelblaster-as-the-compiler-backend)
above.

1. ModelBlaster profiles each (model, backend) pair on spike/FireSim and emits
   an IREE-shape `results.csv` — the same schema `xpu-rt/profile_loader.py`
   reads on both flows, which is why the name outlived the IREE path.
2. `xpu-rt/scheduler.py` (imported live from this checkout via `XPURT_ROOT`)
   reads those CSVs and computes a core-assignment schedule, same as Flow B.
3. ModelBlaster's `examples/xpurt_demo/run.sh` builds a single Zephyr ELF from
   that schedule and runs it via `harness_xpurt/` — the chipyard/Zephyr
   counterpart of Flow B's `run_xpurt_k1.sh`.

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

1. The Python scheduler modules are sourced from `xpu-rt/*.py`; deps declared in `pyproject.toml` (`pip install -e .`).
2. `runtime/` holds board scripts only — no compiler, runtime library or
   `.vmfb`. The merlin/IREE tooling that used to live there is retired.
3. If submodule contents are missing, runtime build/profile scripts will fail early.
