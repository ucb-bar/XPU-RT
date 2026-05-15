# XPU-RT

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and
scheduling flow for efficient mapping of robotic and AI workloads onto
heterogeneous shared-memory SoCs.

It combines two complementary subsystems:

1. **A compiler generator** (formerly *CompGen*) — an LLM-driven
   compiler-recipe generator for heterogeneous hardware targets. Given a
   PyTorch program and a hardware profile, it produces a verified deployment
   recipe: graph/lowering transforms, custom kernels, placement decisions, and
   runtime artifacts. The LLM is a *proposal engine*; deterministic
   verification decides what ships. Drive it through Claude Code via the
   bundled MCP server (`xpu-rt-mcp`).

2. **A multi-cluster scheduler + runtime** — a CVX/MILP-based two-cluster
   scheduling solver (CPU_P / CPU_E, with QNN-island and per-target cost
   models), a C runtime dispatch layer that targets SpacemiT, QRB5165, and
   host x86/ARM, and a Merlin/IREE integration for the actual compile-and-deploy
   path on embedded SoCs.

The two halves share `runtime/` (native libxpu_rt + dispatch runners), a
single Python package (`xpu_rt`), and one pyproject.toml. The compiler
pipeline produces deployment bundles that the scheduler+runtime then runs on
hardware.

## Quick start

```bash
git clone --recurse-submodules https://github.com/ucb-bar/XPU-RT.git
cd XPU-RT
./setup.sh                              # optional: existing XPU-RT bootstrap
uv sync                                 # installs the xpu_rt Python package
uv run xpu-rt --help                    # compiler generator CLI
uv run xpu-rt-mcp                       # MCP server for Claude Code
```

For the scheduler stack:

```bash
uv pip install -e ".[scheduler]"        # cvxpy / pandas / matplotlib / scipy
uv run python scripts/run_xpurt_schedule.py --help
```

## Layout

```
XPU-RT/
├── xpu-rt/python/xpu_rt/         # unified Python package
│   ├── agent/  capture/  ir/  passes/  stages/  ...   # compiler generator
│   └── scheduler/                # CVX two-cluster scheduler (XPU-RT origin)
├── xpu-rt/tests/                 # ~7500 pytest tests mirroring the package
├── xpu-rt/{configs,schemas,examples,benchmarks,userpacks,contrib,infra}/
├── runtime/                      # native runtime (libxpu_rt + dispatch runners)
│   ├── native/libxpu_rt/         # core C/CUDA runtime + drivers
│   ├── src/  include/  templates/
│   └── tools/                    # json_dispatch_runner, xpurt_scheduler_runner
│   └── targets/backends/qnn/     # QRB5165 cost model + island DAG scheduler
│                                 # (CompGen backend pattern; was qnn_scheduler/)
├── models/qnn/                   # ONNX → TFLite → QNN DLC conversion tooling
├── sims/IsaacLab/                # robotics simulation environment (submodule)
├── merlin/                       # SpacemiT/QRB5165 compiler toolchain (submodule)
├── zephyr-chipyard-sw/           # embedded RTOS support (submodule)
├── docs/                         # documentation
├── third_party/                  # vendored: autocomp, kernelblaster, llvm-project,
│                                 #           npu_model, pi0-quant, zephyr, cuda-tile
├── scripts/                      # heterogeneous_loop, qnn_island_demo,
│                                 # run_xpurt_schedule, profiling, MCP helpers
├── xpu-rt/data/                  # op-definition KB + per-model fixtures
├── xpu-rt/audit_seed/            # tracked input seeds for the audit framework
└── build/                        # gitignored: paper/, plots/, results/ generated outputs
```

## Compiler generator (`xpu-rt`)

The compiler generator drives compilation through bounded, declarative
LLM-proposed artifacts (transform scripts, kernel recipes, policies), which
deterministic compiler infrastructure executes. Only verified artifacts
are promoted into a deterministic recipe library.

```bash
xpu-rt mcp install                       # wire into Claude Code's ~/.claude.json
xpu-rt mcp doctor                        # verify MCP tools load
xpu-rt ext new provider my_chip          # scaffold a user-space extension
xpu-rt ext list                          # show discovered providers/dialects
```

Drop-in Python tools at `~/.xpu_rt/extensions/*.py` are discovered without
`pip install`. See [docs/getting-started/extension-authoring.md](docs/getting-started/extension-authoring.md).

The Python API:

```python
from xpu_rt.api import compile_model
recipe = compile_model(model, target="spacemit_x60", out_dir="bundle/")
```

## Scheduler + runtime (`xpu_rt.scheduler`)

The two-cluster CVX scheduler optimises makespan across CPU_P / CPU_E (or
arbitrary heterogeneous units), respecting transfer times, dependency DAGs,
and infeasible-machine constraints.

```bash
uv run python scripts/run_xpurt_schedule.py \
  --workload dispatches.json --proc-times profiles.json --transfers xfer.json
uv run python scripts/qnn_island_demo.py
uv run python scripts/heterogeneous_loop.py
```

The runtime targets are built with CMake:

```bash
cmake -B runtime/build -S runtime           # CompGen-style native libxpu_rt
cmake -B runtime/build -S runtime \
      -DXPURT_STANDALONE_LIB_PATH=merlin/build/.../libxpurt_standalone.a
cmake --build runtime/build --target xpurt_scheduler_runner json_dispatch_runner
```

## What ships in this repository

- `xpu_rt` Python package: 43+ subpackages covering capture (torch.export →
  payload IR), analysis, transforms, kernel search (autocomp adapter), kernel
  contracts, target backends, audit/verification, agent integration, MCP
  server, scheduler.
- `libxpu_rt` C runtime: event-tensor execution, CPU/CUDA drivers, command
  buffers, semaphores, perf-counter instrumentation, tracing hooks.
- QNN island scheduler with QRB5165 calibration data.
- Real-workload fixtures under `xpu-rt/tests/_fixtures/` (SmolVLA, Gemma,
  TinyLlama, Qwen-MoE, VLA-decoder).
- Paper LaTeX source and plotting infrastructure.

## Documentation

- [Operating manual for agents (CLAUDE.md)](CLAUDE.md)
- [Repository-local operating manual (AGENT.md)](AGENT.md)
- [Installation](docs/getting-started/installation.md)
- [MCP setup](docs/getting-started/mcp-setup.md)
- [Extension authoring](docs/getting-started/extension-authoring.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [CLI reference](docs/reference/cli.md)
- [Python API](docs/reference/python-api.md)
- [Extension points](docs/reference/extension-points.md)

## Contributors

XPU-RT brings together work from Dima Nikiforov, Kris Dong, Minh Nguyen,
ailsa-sun, Augustin Coppari Hollmann, and the UCB-BAR group. See
`git shortlog -sn` for the full credit list and `git log --grep=Phase` for
the architectural milestones.

## License

Apache License 2.0. See [LICENSE](LICENSE).
