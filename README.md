# XPU-RT Scheduling and Runtime Integration

## Project Description

**XPU-RT** is an adaptable full-stack end-to-end (E2E) compilation and scheduling flow for efficient mapping of robotic multi-model workloads onto heterogeneous shared-memory SoCs.

## Framework model: bring your own compiler

The scheduler core (`xpu-rt/`) is **plug-and-play** —
compiler- and codegen-agnostic. Anything that can (a) emit a per-op profile in
the IREE dispatch-shape CSV schema and (b) build a single binary that dispatches
per-op kernels according to the schedule's core assignment can sit on the
"compiler" side of the flow:

| flow | compiler / codegen | target | profiling | docs |
|---|---|---|---|---|
| **ModelBlaster** | PyTorch → quantized Zephyr/RISC-V; curated + LLM-agentic kernel-gen | chipyard (Saturn/Gemmini, RISC-V) | spike / FireSim | [ModelBlaster section below](#modelblaster-as-the-compiler-backend), [`zephyr-chipyard-sw/modelblaster/README.md`](zephyr-chipyard-sw/modelblaster/README.md) ("Workflow: integrating with XPURT"), [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md) |

The flow feeds `xpu-rt/scheduler.py` and reads/writes the
`gen/profile/.../results.csv` + `schedules/*.json` shapes; any other compiler
that produces the same two artifacts can be swapped in. ModelBlaster brings its own
compiler (ModelBlaster's codegen pipeline, including an LLM-driven kernel
generator) and its own Zephyr/chipyard build+run path; its walkthrough is in
["ModelBlaster as the compiler backend"](#modelblaster-as-the-compiler-backend).

### Documentation

* [`docs/end_to_end_xpurt_firesim.md`](docs/end_to_end_xpurt_firesim.md)
  — full walkthrough from a multi-network workload spec to a FireSim
  run with trace plots (scheduler → codegen → build → run → analyze),
  on the Saturn-Gemmini-Q31 path.
* [`docs/mlp_dronet_yolo_spike_reproduction.md`](docs/mlp_dronet_yolo_spike_reproduction.md)
  — a simpler, no-FireSim variant of the same flow: same ModelBlaster codegen and
  checkout (`zephyr-chipyard-sw/modelblaster/`), profiled entirely on
  spike with the `greedy`/`greedy_periodic`/`greedy_reserved`/`auto`
  solvers (no MOSEK license needed). Fresh clone to `OVERALL: PASS (3 models)` in six commands
  (see that doc for the full step-by-step + troubleshooting):
  ```bash
  git clone git@github.com:ucb-bar/XPU-RT.git && cd XPU-RT
  git submodule update --init zephyr-chipyard-sw
  git -C zephyr-chipyard-sw submodule update --init modelblaster

  cd zephyr-chipyard-sw
  source scripts/install_conda.sh && bash scripts/install_submodules.sh && bash scripts/install_toolchain_sdk.sh
  cd ..

  bash scripts/repro_mlp_dronet_yolo_spike.sh --trace
  ```
* [`docs/mlp_dronet_yolo_qnn_reproduction.md`](docs/mlp_dronet_yolo_qnn_reproduction.md)
  — the same three networks on a physical QRB5165 through QNN (Flow C):
  modelblaster's `extract_graph`/registry/emitters/schedule-ingest reused as
  a library, a QNN lane runtime instead of its codegen, and
  predicted-vs-actual gantts from `plot_xpurt_trace.py`. Includes a
  stage-by-stage diff against the standard modelblaster flow and a map to
  the key files.
* [The XPU-RT Runtime: schedule walker](#the-xpu-rt-runtime-schedule-walker)
  — the same generated walker on Zephyr/Chipyard and on any Linux target
  (x86-64 host, riscv64 QEMU, SpacemiT K1 and K3).
* [`zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md`](zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md)
  — micro-ROS fixed-pinning baseline flow (the reference against which
  the scheduler is benchmarked).

## Getting started

### Repository Initialization

```bash
git clone https://github.com/ucb-bar/XPU-RT.git
cd XPU-RT
```

**Don't run `git submodule update --init --recursive` here** — one of this
repo's top-level submodules, `hw/chipyard`, is very
large (it vendors 100+ of its own nested submodules) and
is not needed for the flows below; a recursive init will sit there
for a very long time pulling in gigabytes you don't need, and may get
effectively stuck on chipyard's own submodule tree. Initialize only what
the flow you're following actually needs (ModelBlaster's is in its own section
further down).

#### Describe the workload
A workload is a top-level JSON spec under `data/toplevel/` that lists the
networks to co-schedule, for example
[`data/toplevel/networks_mlp_dronet_yolo_spike.json`](data/toplevel/networks_mlp_dronet_yolo_spike.json):

```text
"mlp": {
  "id": 1,
  "identifier":            # model name
  "dispatch_deps_path":    # path to the model's dispatch-graph JSON (under gen/vmfb/)
  "period":                # Duration in millisec between execution windows (inverse of frequency)
  "window_duration":       # Duration in millisec for model to finish after window start
}
```

### Run XPU-RT Scheduler

```bash
python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_mlp_dronet_yolo_spike.json --solver auto
```
The schedule is written to `schedules/scheduled_<workload>[_<solver>]_profiled.json`
and its plot to `plots/<workload>[_<solver>]_profiled.png`.

#### Choosing a solver (`--solver`)

| solver | what it does | when to use |
| --- | --- | --- |
| `milp` (default) | global cvxpy MILP optimum; MOSEK by default, any MIP-capable cvxpy backend via `--cvxpy-solver` (e.g. `HIGHS`, `SCIPY`) | rarely usable — of the eleven spike/FireSim workloads only the 212-op `yolov8_only_spike` solves at all; see the scaling note below |
| `greedy` | list scheduling, earliest completion for every op | baseline heuristic |
| `greedy_periodic` | same, but non-periodic ops get picked first | non-periodic critical path you don't want fragmented |
| `greedy_reserved` | `greedy_periodic` ordering + periodic ops take the *least contended* lane that still meets their deadline instead of the fastest one | heterogeneous multi-lane targets where a periodic job would otherwise squat on the accelerator the makespan-critical job needs |
| `heft` | HEFT: order by upward rank (longest path to a sink) rather than earliest completion, place by earliest finish with gap insertion | best makespan on several workloads — but it is deadline-blind, so check the window misses before trusting it |
| `heft_edf` | HEFT's rank ordering for non-periodic ops, earliest-deadline-first for periodic ops, periodic banded above non-periodic | the strongest single picker by validity (8/11 RISC-V, 23/30 generated) |
| `decomposed` | two-phase: periodic ops are placed first, then non-periodic ops backfill the remaining gaps | periodic-heavy workloads; the only picker that lands every dronet window on FireSim dronet@50ms + yolov8 |
| `pso`, `sa` | particle swarm / simulated annealing over a random-key encoding, seeded from every heuristic | research use — they match the best heuristic and have not beaten it; `--search-budget` sets the per-pass wall clock |
| `milp_native` | the same MILP as `milp`, built directly against MOSEK's Optimizer API instead of through cvxpy | when you want a MIP start: cvxpy silently discards one, this accepts it. Warm-started from HEFT by default. Better than `milp` cold (90.9 vs 112.3 ms on a 242-op instance) but its best result only ties the heuristic that seeded it |
| `cpsat` | OR-Tools CP-SAT: interval variables + one `AddNoOverlap` per machine | the exact method that actually scales — beats MOSEK at every size measured (212, 242 and 271 ops) and still solves at 677 where the MILP cannot finish building. Needs `ortools` in `python3`, or `XPURT_CPSAT_PYTHON` pointing at an interpreter that has it |
| `auto` | runs all six constructive pickers and ranks them on (missed periodic windows, whether the refinement loop converged, makespan, total schedule length) | **recommended** — a few seconds even on the largest workloads here. It optimises for a *valid* schedule, so it will give up a little makespan to stop missing deadlines: 10/11 valid against 3-8 for any single picker, and it warns when no candidate is valid |

#### What the MILP already prunes, and where it still runs out

`xpu-rt/scheduler.py` is a big-M disjunctive formulation, and it carries
several model-reduction passes:

| pass | what it removes | default | reachable from the CLI / spec? |
| --- | --- | --- | --- |
| `prune_cross_period_constraints` | (3) precedence between ops whose windows are provably disjoint, and the whole (4)(5) non-overlap pair when two ops' windows cannot overlap | on | yes — `scheduler.prune_periodic`, `--prune-periods` / `--no-prune-periods` |
| `prune_overlap_constraints_for_dependency_chain` | (4)(5) for pairs the precedence DAG already orders transitively (bitset reachability over the topological order) | on | no — the default is the only way to get it |
| combination-overlap test | (4)(5) for `(k1, k2)` combination pairs that share no machine | always | n/a |
| `restrict_makespan_to_nonperiodic` | (6) `C_max` rows for periodic ops | on | yes — `scheduler.restrict_makespan_to_nonperiodic`, `--include-periodic-in-makespan` |
| `_compute_big_m` | nothing, but tightens `H` to `(Σ max durations + transfers) x 2`, floored at 5000, instead of a loose constant | always | no |
| `fusion_threshold` | operations themselves — fuses everything under a duration threshold before building the model | off | no — `schedule()` argument only |
| `schedule_window` + `packing.py` | solves per time-window and recombines | unused | no |

They work, and one of them is load-bearing. On
`networks_3way_dronet5ms_mlp2ms_yolov8_qrb5165` (191 operations, 3 lanes) a
naive model would emit all 18,145 operation pairs; pruning leaves 540, and
the whole model is 4,003 constraints built in 0.6 s. On the one RISC-V
workload the MILP can solve at all (`yolov8_only_spike`, 212 ops),
switching **`prune_overlap_constraints_for_dependency_chain` off is the
difference between a 615.01 ms answer in 620 s and no answer in 900 s** —
and it is the one pass reachable from neither the CLI nor the spec, working
purely by being default-on. `prune_cross_period_constraints` and
`restrict_makespan_to_nonperiodic`, by contrast, change nothing there,
because that workload has no periodic networks for them to act on — and
every RISC-V workload that does have periodic networks is too large for the
MILP to finish, so on this family that pass has never been exercised on a
model that completes. Fusion cuts both ways: on that same
workload it costs makespan at a fixed time limit (616.20 ms at
`fusion=0.5`, 617.55 at `2.0`, against 615.01 unfused), but on FireSim
dronet+yolov8 it is the difference between a solve and a timeout — 725 s at
`fusion=0.5` where both time-limit settings ran out at 900 s. Reach for it
when the model won't build, not to improve one that already solves.

What none of them touched was the **variable** count: `beta`, the pairwise
ordering matrix, was allocated dense as `operations x operations` before
any pruning ran, so MOSEK presolved an ordering bit for every pruned pair
too. It is now allocated per surviving pair instead. Identical model,
identical optimum, measured:

| workload | ops | constraints | variables (dense → sparse) | MOSEK solve (dense → sparse) |
| --- | --- | --- | --- | --- |
| flowc 4-way | 58 | 1,479 (unchanged) | 3,597 → 441 | 1.66 s → 1.18 s |
| 3-way dronet5ms+mlp2ms+yolov8n | 191 | 4,003 (unchanged) | 37,246 → 1,305 | 9.42 s → 3.60 s |
| 2x resnet50+dronet+mlp | 372 | 16,541 (unchanged) | 139,873 → 3,998 | 130.06 s → 20.87 s |

**Where it still runs out.** Almost everywhere on this family. The MILP is
also the only solver that doesn't reseed periodic networks to one instance
before its first pass, so it faces the workload_factory's full
horizon-derived counts: the spike repro workload is **7784 operations** for
it (against 705 for the greedy path), FireSim dronet@10 ms is 4112,
mlp10+dronet20 is 3072, dronet@20 ms is 2162. Of the six RISC-V workloads
small enough to be worth trying (212-803 ops), only the 212-op one finished
inside 900 s.

The mechanism: `_periods_overlap` can only prune a pair when
*both* ops carry `min_start_t`/`max_end_t`; a non-periodic op has neither,
so every (periodic, non-periodic) and (non-periodic, non-periodic) pair
survives by construction. On `networks_periodic_dronet5ms_yolov8_qrb5165`
— 1751 operations, because the MILP is also the only solver that doesn't
reseed periodic networks to one instance before its first pass, so it
faces the workload_factory's full horizon-derived counts — 364,293 of the
1,532,125 pairs survive, giving 1,465,895 constraints that take ~330 s
just to *build* in Python. Sparse `beta` cuts the ordering variables there
from 3,066,001 to 364,293 (whole model 3,070,255 → 369,547) but leaves the
constraint count untouched, and it still does not solve: a 900 s run peaked
at 41.9 GB and was killed inside cvxpy's own compilation, before MOSEK ever
saw the problem. Bounding non-periodic ops with ASAP/ALAP windows so those
pairs become window-prunable too is the obvious next lever; until then,
use `auto` on workloads of that size.

Where the MILP does finish it is worth having. On QRB5165 it proves `auto`
optimal three times over (33.57, 28.64 and 21.81 ms, matched exactly in
about a second). On `yolov8_only_spike` it beats every heuristic —
615.01 ms against 628.94 — but takes 620 s to do it, 440x the heuristic's
1.4 s, for 2.2%.

No single heuristic wins everywhere — hence `auto`. Measured over the
eleven spike/FireSim (RISC-V) workloads, reporting the objective
makespan in ms and whether the schedule is *valid*: misses no periodic
window, and holds enough periodic instances to cover its own makespan.
`!` = missed a window, `u` = under-covers.

| workload | `greedy` | `greedy_periodic` | `greedy_reserved` | `decomposed` | `heft` | `heft_edf` | `auto` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spike: mlp+dronet+yolov8 | 2512.41 u | 665.47 | **628.94** | 2600.83 u | 630.91 ! | 2511.61 u | **628.94** |
| spike: + gemmini_q31 | 126.60 | 106.62 ! | 105.14 ! | 119.86 | **103.02** | 110.88 | **103.02** |
| spike: yolov8 only | 628.94 | 628.94 | 628.94 | 628.94 | **628.14** | **628.14** | **628.14** |
| fsim: dronet+yolov8 | 115.86 | 110.68 | 110.68 | 116.91 | **98.41** | 104.58 | **98.41** |
| fsim: dronet50+yolov8 | 127.71 | 123.52 ! | 122.95 ! | 127.07 | **98.41** | 115.10 | **98.41** |
| fsim: dronet20+yolov8 | 158.59 ! | 168.04 ! | 185.51 ! | 173.47 | 114.57 ! | **148.79** | **148.79** |
| fsim: dronet10+yolov8 | 358.55 ! | 272.95 ! | 244.86 ! | 358.61 u | 132.39 ! | 316.28 u | 316.28 u |
| fsim: mlp10+dronet20+yolov8 | 174.67 ! | 114.11 ! | 113.17 ! | 173.99 u | 115.26 ! | **158.19** | **158.19** |
| fsim: mlp+dronet+yolov8 static | 175.55 ! | 177.54 ! | 328.07 ! | 175.13 | 150.53 ! | **158.19** | **158.19** |
| fsim: dronet50+yolov8 static | 158.16 | 158.16 ! | 158.16 ! | 158.22 | **158.14** | 158.16 | **158.14** |
| fsim: mlp+dronet het | 1163.46 u | 1163.46 u | **293.23** | 1163.46 u | 284.53 ! | 1125.11 u | **293.23** |
| **valid schedules** | 5/11 | 3/11 | 4/11 | 7/11 | 5/11 | 8/11 | **10/11** |

Every picker is the best on some workload and among the worst on another.
`greedy_reserved` is worth 4x on the FireSim heterogeneous workload and is the
worst of the seven on the static mlp+dronet+yolov8. `heft` gives the shortest
makespan on five workloads and misses deadlines on five. `heft_edf` is the
strongest *single* method by validity, and is 4x off on the spike 3-model
workload. Only `auto` is near the top everywhere.

Adding `heft`/`heft_edf` to `auto`'s candidate set improved eight of the
eleven: FireSim dronet50+yolov8 by 22.6% (127.07 -> 98.41 ms), dronet20 by
14.2%, gemmini_q31 by 14.0%, dronet+yolov8 by 11.1%, and it turned the
previously-invalid mlp10+dronet20+yolov8 into a valid 158.19 ms schedule.
One workload — FireSim dronet@10 ms + yolov8 — still has no valid answer from
any method; `auto` says so rather than returning its pick silently.

Beyond the constructive pickers, `--solver pso`, `sa` and `cpsat` exist and
are documented in
[`docs/scheduler_solver_study.md`](docs/scheduler_solver_study.md), along with
the cvxpy backend comparison (`--cvxpy-solver`) and a 30-workload generated
corpus. Short version: on a fixed 677-op instance the metaheuristics match the
best heuristic and never beat it; CP-SAT gets within 2.3% in 180 s, on an
instance whose MILP model cvxpy is still *compiling* after 22 minutes; and
MOSEK is 2.8x better than HiGHS at equal time budget.

### Tuning knobs, measured

- **`--max-periodic-iters` is a divergence multiplier, not a quality knob.**
  The refinement loop grows periodic instance counts from the makespan those
  same instances inflated. Where it does not converge, every extra pass makes
  the answer worse and less covered: `greedy` on the spike 3-network workload
  runs 657 -> 1272 -> 2512 -> 4992 -> 9952 ms as the cap goes 1 -> 2 -> 4 ->
  8 -> 16. The default of 4 is fine; raising it never helped on any of the 19
  workloads swept. A non-converged run now says so explicitly.
- **`prune_periodic` changed nothing** — identical results in all 44
  (workload x solver) cells on this family, on and off.
- **`restrict_makespan_to_nonperiodic` must stay on.** Turning it off skips
  the "seed each periodic network at one instance" step and hands the solver
  the workload_factory's full horizon-derived counts: the spike 3-network
  workload goes 2512 -> 12912 ms (`greedy`) and 665 -> 10430 ms
  (`greedy_periodic`).
- **`--reserved-max-slowdown`** (`scheduler.reserved_max_slowdown`) caps how
  much slower than its own fastest lane a periodic op may run to stay off a
  lane the non-periodic jobs need. Swept over {1, 2, 4, 8, unbounded}: **2.0**
  is best-or-tied on ten of the eleven RISC-V workloads and is the default.
  It is the smallest cap that still reaches the valid 293.23 ms schedule on
  the FireSim heterogeneous workload (1.0 collapses to 1163.46 ms). QRB5165
  workloads want a larger value — 2x-resnet50 needs 8.0 to find its 21.81 ms
  schedule — which is why this is a knob and not a constant.

## ModelBlaster as the compiler backend

ModelBlaster brings PyTorch → quantized Zephyr/RISC-V codegen (curated
kernels + an LLM-agentic kernel generator) and profiles on spike/FireSim.
It plugs into `xpu-rt/scheduler.py` like any other compiler backend.

### Repository layout

ModelBlaster ships as a git submodule **nested inside `zephyr-chipyard-sw`**
(its canonical location — the same one the spike-only reproduction flow
uses), not at the top level:

```bash
git submodule update --init --recursive zephyr-chipyard-sw   # pulls in modelblaster (+ KernelBlaster)
```

```text
XPU-RT/                          (this repo)
└── zephyr-chipyard-sw/          submodule — Zephyr BSP + samples
    └── modelblaster/            submodule — the compiler
```

ModelBlaster's `examples/xpurt_demo/run.sh` and `benchmarks/runners/firesim.py`
look for XPU-RT as a sibling checkout (`XPURT_ROOT` defaults to `../XPU-RT`).
Inside this repo ModelBlaster sits two levels down, so set `XPURT_ROOT` to the
XPU-RT root:

```bash
export XPURT_ROOT="$(cd ../.. && pwd)"   # run from inside zephyr-chipyard-sw/modelblaster
```

[`scripts/repro_mlp_dronet_yolo_spike.sh`](scripts/repro_mlp_dronet_yolo_spike.sh)
runs steps 1 to 3 end to end on spike; see
[`docs/mlp_dronet_yolo_spike_reproduction.md`](docs/mlp_dronet_yolo_spike_reproduction.md).

### 1) Profile each (model, backend) pair

From the ModelBlaster submodule, profile every model/backend combination the
workload needs, on spike or FireSim. This produces the per-op cycle data the
scheduler reads in step 2:

```bash
cd zephyr-chipyard-sw/modelblaster
QUANT=int8 TARGET=rvv         RUNNER=firesim bash examples/dronet/run.sh
QUANT=int8 TARGET=gemmini_q31 RUNNER=firesim bash examples/dronet/run.sh
# ...one run per (model, backend) pair in the workload
```

Then emit each model's dispatch graph with `pipeline/emit_dispatch_graph.py`
and copy the profiles into this repo's `gen/profile/` tree, where the scheduler
looks for them. Steps 1 to 3 of `scripts/repro_mlp_dronet_yolo_spike.sh` do
this for the three-network workload, with the exact environment and paths.

### 2) Schedule the workload

From the XPU-RT root, `scripts/run_xpurt_schedule.py` reads a network-dependency
JSON (which networks, their periods, the machine) and the profiles from step 1,
and writes the schedule to `schedules/`:

```bash
python3 scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_mlp_dronet_yolo_spike.json \
    --solver greedy_periodic --profiled
# -> schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json
```

`--solver` selects any solver from
[Choosing a solver](#choosing-a-solver---solver). The `milp` solver needs the
`milp` extra (`pip install -e ".[milp]"`, which adds cvxpy) and a MIP-capable
cvxpy backend chosen with `--cvxpy-solver`:
- **HIGHS** is free and open source.
- **MOSEK** needs a licence file (`MOSEKLM_LICENSE_FILE`) and, at equal time
  budget, finds schedules about 2.8x better on this model; see
  [`docs/scheduler_solver_study.md`](docs/scheduler_solver_study.md).

The greedy solvers need neither.

### 3) Build and run the scheduled binary

```bash
cd zephyr-chipyard-sw
SCHEDULE_JSON="$XPURT_ROOT/schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json" \
MODELS=mlp_control,dronet,yolov8_nano \
QUANTS=fp32,int8,int8 \
BACKENDS=scalar,rvv \
RUNNER=spike \
XPURT_TRACE=1 \
bash modelblaster/examples/xpurt_demo/run.sh
```

`xpurt_demo/run.sh` links one object per (model × backend) and dispatches each
schedule entry to the right one. With `XPURT_TRACE=1`, the log carries
per-entry begin and end timestamps, which ModelBlaster's
`scripts/plot_xpurt_trace.py` renders as a Gantt chart against the predicted
timeline. Use `RUNNER=firesim` to run on FireSim instead of spike.

### Env vars ModelBlaster uses to find this checkout

| var | default | used by |
|---|---|---|
| `XPURT_ROOT` | `../XPU-RT` (a sibling checkout; set it to the XPU-RT root when running from `zephyr-chipyard-sw/modelblaster`) | `examples/xpurt_demo/run.sh`, `benchmarks/runners/firesim.py` |

This submodule reference is pinned to a commit (standard submodule
semantics). Because it is nested, bumping it means updating
`modelblaster` inside `zephyr-chipyard-sw`, committing that, then bumping the
`zephyr-chipyard-sw` pointer in this repo.

For the full ModelBlaster-side workflow (profiling knobs, workload JSON
schema, models in scope), see
[`zephyr-chipyard-sw/modelblaster/README.md`](zephyr-chipyard-sw/modelblaster/README.md), section
"Workflow: integrating with XPURT."

## The XPU-RT Runtime: schedule walker

The XPU-RT runtime is the **schedule walker**: a C program, generated from an
XPU-RT schedule, that runs every dispatch on the core the scheduler assigned,
in dependency order and at its scheduled start time.

The walker is a single flow for Zephyr and Linux. One generated source builds
as a Zephyr image for Chipyard (spike / FireSim) or as an executable for any
Linux environment:
- an x86-64 host;
- riscv64 under QEMU user mode;
- RISC-V boards such as the SpacemiT K1 and K3.

### Components

All components live in ModelBlaster, under
[`zephyr-chipyard-sw/modelblaster/`](zephyr-chipyard-sw/modelblaster/):

| Component | File |
|---|---|
| Schedule → dispatch table | `pipeline/ingest_xpurt_schedule.py` |
| Walker source generator | `pipeline/generate_xpurt_main.py` |
| Per-(model, backend) kernels and weights | `pipeline/generate_skeleton.py`, `pipeline/backend_rename.py` |
| Intra-op parallel-for | `runtime/modelblaster_pool/` |
| POSIX platform layer | `runtime/mb_posix.h` |
| Linux build | `harness_xpurt/build_posix.sh` |
| Zephyr build | `harness_xpurt/` (CMake app), driven by `examples/xpurt_demo/run.sh` |

On Linux, `runtime/mb_posix.h` provides the platform services the walker uses:

| Service | Linux implementation |
|---|---|
| Semaphores | `sem_t` |
| Timestamps | `CLOCK_MONOTONIC_RAW`, in nanoseconds |
| Thread pinning | `pthread_setaffinity_np` |
| Exit | `exit(0)` |

`build_posix.sh` compiles every translation unit with
`-D_GNU_SOURCE -DMB_PLATFORM_POSIX`.

The walker prints its outputs, per-dispatch profile and trace as
`=== MODELBLASTER_* ===` markers on stdout. `scripts/plot_xpurt_trace.py`
reads them on every platform.

### Worker modes

`generate_xpurt_main.py --worker-granularity` selects how the walker maps
schedule entries to threads:

| Mode | Workers | Use for |
|---|---|---|
| `kind` (default) | one per core kind | Chipyard, where a core kind is one hart or one intra-op pool |
| `hart` | one pinned worker per `(core_kind, hart)` in the dispatch table | multi-hart boards such as the K1 and K3, where the schedule drives each hart (`CPU_P#0` … `CPU_P#7`) independently |

Harts come from the core registry. To use every hart of a board, list each
hart as a single-hart core of the same kind.

### Core kinds and backends

`--core-kinds` names the core kinds in the schedule and `--backends` names the
kernel backend each kind runs. When the two lists have the same length, they
pair by position. For example, on the K1 both core kinds run the `rvv_x60`
backend:

```bash
--core-kinds rvv,rvv_c1 --backends rvv_x60,rvv_x60
```

### Running on Linux

```bash
cd zephyr-chipyard-sw/modelblaster

# 1. schedule -> dispatch table
python -m modelblaster.pipeline.ingest_xpurt_schedule \
    --schedule sched.json --registry cores/<registry>.json \
    --ir net:graph.json --out out/sched.c --name sched

# 2. walker source
python -m modelblaster.pipeline.generate_xpurt_main \
    --schedule sched.json --out out/sched_main.c --name sched \
    --dispatch-table-header sched.h --core-kinds rvv,rvv_c1 \
    --backends rvv_x60,rvv_x60 --worker-granularity hart

# 3. build a static riscv64 executable
MODEL_NAMES=net MODEL_DIRS_BASE=/abs/gen MODEL_BACKENDS=rvv_x60 \
XPURT_SCHEDULE_C=$PWD/out/sched.c XPURT_MAIN_C=$PWD/out/sched_main.c \
XPURT_INCLUDE_DIR=$PWD/out OUT=$PWD/out/walker \
CC=${CROSS}gcc LDFLAGS=-static \
    bash harness_xpurt/build_posix.sh

# 4. run
./out/walker > run.log                                        # on a riscv64 board
qemu-riscv64-static -cpu max,vlen=256 ./out/walker > run.log  # under QEMU on an x86-64 host
```

- `CROSS` is the prefix of a riscv64 Linux cross toolchain, such as the
  SpacemiT toolchain.
- To build for the x86-64 host, omit `CC` and `LDFLAGS` and use the `scalar`
  backend.
- To run on a board, copy the executable to the board and run it there.

### Supported targets

The table lists results for a three-network schedule (`mlp_control`, `dronet`,
`yolov8_nano`: 826 dispatches, 46 instances) in both worker modes:

| Target | Build | Result |
|---|---|---|
| x86-64 Linux host | native, `scalar` | all dispatches complete; `mlp_control` and `dronet` bit-exact against the golden |
| riscv64, QEMU user mode (`-cpu max,vlen=256`) | static riscv64, `rvv_x60` | 46/46 instances; all three networks bit-exact |
| SpacemiT K1 (8 × X60, Bianbu, Linux 6.6) | static riscv64, `rvv_x60` | 46/46 instances; all three networks bit-exact |
| SpacemiT K3, X100 cluster (CPUs 0–7, Bianbu, Linux 6.18) | static riscv64, `rvv_x60`; run with `taskset -c 0-7` | 46/46 instances; outputs identical to the K1; about 3.3× faster than the K1 on `dronet` and `yolov8_nano` |

One `rvv_x60` build serves the K1, the K3 X100 cluster and QEMU:
- the backend targets `-march=rv64gcv`;
- the X60 and X100 both have VLEN 256;
- the executable is statically linked.

For more detail and the host test (`tests/test_xpurt_walker_host.py`), see
[`zephyr-chipyard-sw/modelblaster/harness_xpurt/LINUX.md`](zephyr-chipyard-sw/modelblaster/harness_xpurt/LINUX.md).

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
├── zephyr-chipyard-sw/         # Git submodule — Zephyr BSP + samples
│   └── modelblaster/           #   nested submodule (PyTorch->Zephyr/RISC-V pipeline)
│       └── third_party/KernelBlaster/  # nested submodule — originating research project
├── env.yml                     # cvxpy+MOSEK conda env ("xpu-rt-schedule") for
                                 #   ModelBlaster's own MOSEK bridge scripts
└── pyproject.toml              # xpu-rt's own deps (`pip install -e .`); see
                                 #   scripts/install_xpurt_deps.sh for the
                                 #   spike-only reproducible-flow's dependency set
```


### Data/Artifact Flow Between This Repo and `ModelBlaster`

ModelBlaster is a submodule nested in `zephyr-chipyard-sw/modelblaster` — but its
own scripts still reach back into XPU-RT via the `XPURT_ROOT` env
vars and a `[tool.uv.sources]` entry rather than a relative import, so
`XPURT_ROOT` needs to be set to `../..` (not left at its sibling-checkout
default) when running from inside the submodule. See
["ModelBlaster as the compiler backend"](#modelblaster-as-the-compiler-backend)
above.

1. ModelBlaster profiles each (model, backend) pair on spike/FireSim and emits
   a `results.csv` in the dispatch-shape schema `xpu-rt/profile_loader.py` reads.
2. `xpu-rt/scheduler.py` (imported live from this checkout via `XPURT_ROOT`)
   reads those CSVs and computes a core-assignment schedule.
3. ModelBlaster's `examples/xpurt_demo/run.sh` builds a single Zephyr ELF from
   that schedule and runs it via `harness_xpurt/`.

## Feedback-driven compilation: post-schedule granularity advisor

Motivating case: you partition a model at the compiler's default dispatch
granularity, profile it, and the schedule xpu-rt computes still
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
2. If submodule contents are missing, the build/profile scripts will fail early.
