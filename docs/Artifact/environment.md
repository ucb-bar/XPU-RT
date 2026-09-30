# Recreating the environment

Two flows, two environments, and they are not the same one.

| flow | environment | doc |
|---|---|---|
| **A — chipyard** (spike / FireSim, Zephyr) | conda, from zephyr-chipyard-sw's own installer | [`mlp_dronet_yolo_spike_reproduction.md`](../Demo/mlp_dronet_yolo_spike_reproduction.md) §0 |
| **B — SpaceMiT K1** (Linux/riscv64, on-device) | a plain venv, below | this page |
| Isaac Sim (the warehouse flights, the forest-trail demo) | the `env_isaaclab` conda env, from `requirements-isaac.txt` | the Environments section below; [`xpurt_env_setup.md`](xpurt_env_setup.md) walks the same install step by step |

**None of them is merlin's `.venv`.** That directory still exists on the
original dev machine and was, until this branch, the only Python on it with
`torch` installed — which is why so much tooling quietly ran from it. merlin is
retired; if you find yourself typing `merlin/.venv/bin/python`, the recipe
below is what you actually want.

## Environments

The record of the two interpreters the study ran with is the pair of requirements files at the
repo root. Every other environment note in `docs/` points here.

| interpreter | file | Python | what runs under it |
|---|---|---|---|
| **host** — `HOST_PY`, default `$REPO/.venv/bin/python` | `requirements-host.txt`, frozen from the study's host venv (numpy 2.5.3, matplotlib 3.11.1, ortools 9.15, cvxpy, mosek 11.2.3, pytest) | 3.13 | solvers, board calibration, trace reading, aggregation, every figure and verifier, the tests |
| **Isaac** — `ISAAC_PY` | `requirements-isaac.txt`, frozen from `env_isaaclab` (Isaac Sim 5.1.0.0 from PyPI, torch); IsaacLab is the checkout at `sims/IsaacLab`, installed editable | 3.11 | every flight (`sims/scripts/sweep_rate_demo.py`, `record_sensor_demo.py`) and the guidance-net training |

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-host.txt        # host
conda create -n env_isaaclab python=3.11 && conda activate env_isaaclab          # Isaac
pip install -r requirements-isaac.txt && pip install -e sims/IsaacLab/source/*
```

`pyproject.toml` declares the host set as lower bounds (`pip install -e .` on a newer interpreter);
`ortools` is a base dependency because CP-SAT is on the reproduction path. The `mosek` wheel installs
from the requirements file; using it needs a licence (`~/mosek/mosek.lic` or `MOSEKLM_LICENSE_FILE`),
and only `--solver milp --scheduler mosek` asks for one.

**Machine paths.** The repo root is wherever the clone lives. Every reproduction script sources
`scripts/env.sh`, which sets `REPO` (the clone), `SIM_TREE` (the tree holding `sims/IsaacLab` and
`sims/scripts`; the clone itself unless overridden — on the study machine
`/scratch/agustin/projects/DIMA/XPU-RT`), `ISAAC_PY`, `HOST_PY` and `RES=$REPO/results/codesign_feedback`.
Per-machine values go in `scripts/env.local.sh` (ignored by git; `scripts/env.local.sh.example` is the
template) or in the environment (`XPURT_REPO`, `XPURT_SIM_TREE`, `ISAAC_PY`, `HOST_PY`), which wins over
the file. The documents write `$REPO`, `$SIM_TREE`, `$HOST_PY` and `$ISAAC_PY` for these; an absolute
path under `results/` is a record of what ran, not an instruction.

**Solver settings.** A CP-SAT solve run by hand takes `export XPURT_CPSAT_WORKERS=0` (let CP-SAT
choose its parallelism; the default is one worker). The reproduction scripts export their own values
(`XPURT_CPSAT_WORKERS=8 XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1`, `solve_stage2_hard.sh`).

## Flow B: the K1 board flow

```bash
git clone --recurse-submodules https://github.com/ucb-bar/XPU-RT.git
cd XPU-RT

python3 -m venv .venv
.venv/bin/pip install -e . -e ModelBlaster
.venv/bin/pip install pytest                 # to run the suites
```

`-e ModelBlaster` is what brings in `torch`, `numpy`, `requests`, `pyyaml`,
`pillow` and `ultralytics`; `-e .` brings XPU-RT's own. Nothing else is
required to schedule, build, deploy, profile and adjudicate.

Verified on a clean venv on 2026-08-29 (with `cvxpy` also installed):

```
722 passed, 2 failed, 2 skipped        # both failures: "MOSEK is not installed"
```

Without `cvxpy` the two MOSEK tests skip instead, and the numbers are the ones
in the next section.

## What a fresh clone can and cannot do

Verified by cloning `dev` into an empty directory and following this page
verbatim — 1666 files, 389 MB, no board:

```
711 passed, 9 skipped        xpu-rt/tests tests
all 4 examples ran           examples/run_all.py
Makespan: 8.00 ms            a real solve on committed profiles
```

**Committed, so it works immediately:** the measured profiles
(`gen/profile_mb`, four core widths for dronet / ffn_block /
yolov8_nano_64x96), the dispatch graphs (`gen/mb/vmfb`), 69 workload specs,
`data/banks`, and two model checkpoints (`dronet`, `mlp_control`).

That is enough to schedule, produce real advice, bridge it, rewrite a graph
and reach a verdict — the whole loop except the board steps.

**Generated, so you make them:** the IR under `ModelBlaster/build/`. It is a
build artifact and not tracked, which is why
`examples/feedback_loop/one_revolution.py` stops at step 3 on a clean checkout.
One command, no board:

```bash
cd ModelBlaster
PYTHONPATH=.:src ../.venv/bin/python pipeline/extract_graph.py \
    --model ffn_block --quant int8 --out-dir build/k1/ffn_block/int8
```

≈1.7 s. Re-run the example and it goes through the bridge and the rewrite.

**Needs the network:** `yolov8_nano` pulls its weights through ultralytics on
first use. `ffn_block`, `attn_block`, `norm_block`, `lstm_tiny`,
`vitfly_frontend` and `fused_full` are synthetic and need nothing;
`dronet` and `mlp_control` have vendored checkpoints.

**Needs a board:** anything that profiles or runs. See
[`k1_board.md`](../K1/k1_board.md).

**Not in the repo at all:** the ViNT calibration data (`datasets/idsia/samples/sc`
and the IsaacLab forest renders; see [`external_data.md`](external_data.md)). Its two kernels are verified standalone on the
board but have never run inside the model.

### What the 9 skips are

Six are missing local artifacts and say so by name — `artifacts/k1_run/contention.json`,
multi-core profiles for models that only have `topo_0`, a sweep aggregate. Two
are `cvxpy not available`. One is an impl-aware schedule this tree has not
solved. None is a silent pass.

Installing `cvxpy` clears two of them even without a MOSEK licence:

```bash
.venv/bin/pip install cvxpy
```

### Optional: MOSEK

```bash
.venv/bin/pip install "cvxpy[MOSEK]"    # licence-gated; see the README
```

Only `--solver milp --scheduler mosek` needs it. `greedy`, `greedy_periodic`
and every list scheduler in the registry run without a licence, and
`docs/Feature/solvers.md` says which is which. Two tests fail without it and say so
in the failure text rather than skipping, which is deliberate: a silently
skipped MILP test is how a broken MILP path stays broken.

### The cross toolchain — not optional

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"        # exports CROSS
```

Fetches `riscv64-unknown-linux-gnu-gcc 14.3` into `tools/riscv-tools-spacemit/`
(≈6 GB, gitignored) if it is not already there.

**GCC 13.2 miscompiles the RVV intrinsics** — it reorders two
`__riscv_vsetvl_*` calls so a widening instruction executes under the narrow
vtype, and the board binary SIGILLs with no stdout at all. That is what `CROSS`
defaults to via chipyard's riscv-tools, so the script refuses anything below
14 rather than letting you find out on the board.

**GCC 14.3 is wrong in the opposite direction**, and worse: it substitutes a
wrong AVL on a *chained* `vsetvl`, computing a wrong answer instead of
crashing. Two committed kernels shipped that way (`lstm_s8` err=20,
`avgpool2d_s8` err=68). Pass the element count to every width, and run
`ModelBlaster/scripts/check_rvv_avl.py`, which exists to enforce it.

### The board

```bash
# ~/.ssh/config
Host k1
    HostName <board ip>
    User root
    IdentityFile <key>
```

Scripts read `MODELBLASTER_K1_HOST`, default `k1`. No credentials are read or
written by anything in this repo; ssh does whatever it is already configured to
do.

Check it is free before you take it:

```bash
ssh k1 'awk "/^cpu[0-9]/ {u=\$2+\$3+\$4; t=u+\$5; printf \"%s %.1f%%\n\", \$1, 100*u/t}" /proc/stat'
```

**Do not read `/proc/loadavg`.** It has a permanent floor of exactly **2.00**
from two D-state kernel threads (`vq0`, `vq1`) that never leave uninterruptible
sleep, so any idleness check based on it concludes the board is busy, forever.
Per-CPU `/proc/stat` is the only valid busy signal here.

## Running things

```bash
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json
.venv/bin/python -m pytest xpu-rt/tests tests -q
.venv/bin/python examples/run_all.py
```

ModelBlaster's own scripts take `PY` for the interpreter:

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"
PY=$PWD/.venv/bin/python PROFILE_OUT_ROOT=$PWD/gen/mb/profile \
    bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0
```

Its own test suite must run **from its own root**, because the curated-kernel
verify cross-compiles with repo-relative include paths:

```bash
cd ModelBlaster && ../.venv/bin/python -m pytest tests pipeline/tests -q
```

Without `CROSS` the curated kernels cannot be verified, so the picker correctly
falls back to the reference and the pin tests skip rather than fail.

## The ModelBlaster submodule

ModelBlaster is XPU-RT's top-level submodule, `ModelBlaster/`, and both flows build from it. An
uninitialised submodule is an *empty directory*, not an error, so initialise it before anything that
imports or builds from it:

```bash
git submodule update --init ModelBlaster                     # both flows
git submodule update --init --recursive zephyr-chipyard-sw   # Flow A's Zephyr BSP
```

## If something is missing

`scripts/install_xpurt_deps.sh` is the conda-based installer for Flow A, and it
resolves whichever ModelBlaster checkout is actually on disk. The venv recipe
at the top of this page is the one to use for Flow B, and it is the one that
has been verified end to end on a clean environment.
