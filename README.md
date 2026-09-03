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
| **A — chipyard** | PyTorch → quantized Zephyr/RISC-V; curated + LLM-agentic kernel-gen | chipyard (Saturn/Gemmini, RISC-V) | spike / FireSim | [Flow A section below](#flow-a-modelblaster-as-the-compiler-backend), [`zephyr-chipyard-sw/modelblaster/README.md`](ModelBlaster/README.md) ("Workflow: integrating with XPURT"), [`docs/Firesim/end_to_end_xpurt_firesim.md`](docs/Firesim/end_to_end_xpurt_firesim.md) |
| **B — SpaceMiT K1** | the same ModelBlaster codegen, cross-compiled for Linux/riscv64 | SpaceMiT K1 (BananaPi), 8 harts + IME | on-device, over ssh | [Flow B section below](#flow-b-modelblaster-on-the-spacemit-k1-board), [`docs/Feature/the_loop.md`](docs/Feature/the_loop.md) |

Both flows feed the same `xpu-rt/scheduler.py` and read/write the same
`gen/profile/.../results.csv` + `schedules/*.json` shapes — the target hardware
and the runtime around the kernels are the only things that change.

**Flow B used to be a different compiler**: merlin → IREE → VMFB. It is not any
more. That path is retired, the merlin submodule is gone, and both flows now
build from ModelBlaster — so the two entries above differ in where the code
runs, not in what generates it. The `results.csv` schema is IREE-shaped for the
same reason a road can keep a Roman route: every reader already speaks it.

### Start here

* **[`docs/Feature/the_loop.md`](docs/Feature/the_loop.md)** — the index: every arrow of the
  compiler↔scheduler cycle and which script owns it. If you have been away,
  read this one.
* **[`docs/environment.md`](docs/environment.md)** — recreating the
  environment. Two flows, two environments, and neither is merlin's `.venv`.
* **[`docs/K1/k1_board.md`](docs/K1/k1_board.md)** — running on the K1: the
  commands, the timings, the two compiler traps, and what to do when it
  breaks.
* **[`examples/`](examples/)** — runnable, one per topic:
  `.venv/bin/python examples/run_all.py`

### Headline feedback-loop result

On the same exactly repeating 100 ms K1 workload, XPU-RT feedback asks
ModelBlaster to expose measured multi-hart implementations and reduces the
**globally optimal worst critical-model response from 8.001335 ms to
4.890542 ms (38.88%)**. This is an implementation-space improvement, not a
solver-only comparison: independently validated feasible schedules attain an
analytic lower bound before and after feedback, so MOSEK, CP-SAT, Greedy, or
any other scheduler restricted to the original graph cannot close the gap.

Ten complete real-time K1 runs per phase corroborate the result: median worst
critical response falls from 10.491000 ms to 7.208521 ms (31.29%), with zero
deadline misses. See [`docs/Feature/the_loop.md`](docs/Feature/the_loop.md#4-the-strongest-result-a-solver-independent-separation)
for the interpretation and [`results/k1_feedback_exact/README.md`](results/k1_feedback_exact/README.md)
for the certificate, checked-in evidence, and exact reproduction command.

### Documentation

* [`docs/Firesim/end_to_end_xpurt_firesim.md`](docs/Firesim/end_to_end_xpurt_firesim.md)
  — full walkthrough from a multi-network workload spec to a FireSim
  run with trace plots (scheduler → codegen → build → run → analyze),
  on the Saturn-Gemmini-Q31 path (Flow A).
* [`docs/Demo/mlp_dronet_yolo_spike_reproduction.md`](docs/Demo/mlp_dronet_yolo_spike_reproduction.md)
  — a simpler, no-FireSim variant of Flow A: same ModelBlaster codegen and
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
* [`docs/Qualcomm/mlp_dronet_yolo_qnn_reproduction.md`](docs/Qualcomm/mlp_dronet_yolo_qnn_reproduction.md)
  — the same three networks on a physical QRB5165 through QNN (Flow C):
  modelblaster's `extract_graph`/registry/emitters/schedule-ingest reused as
  a library, a QNN lane runtime instead of its codegen, and
  predicted-vs-actual gantts from `plot_xpurt_trace.py`. Includes a
  stage-by-stage diff against the standard modelblaster flow and a map to
  the key files.
* [`zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md`](ModelBlaster/examples/microros_demo/ROS_FLOW.md)
  — micro-ROS fixed-pinning baseline flow (the reference against which
  the scheduler is benchmarked).
* [`docs/Feature/the_loop.md`](docs/Feature/the_loop.md)
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

* [`docs/K1/k1_contention.md`](docs/K1/k1_contention.md) -- do concurrent dispatches
  slow each other down? Measured **null**: the distributions overlap and the
  arms are not monotonic in co-runner count.
* [`docs/K1/k1_cost_by_pred.md`](docs/K1/k1_cost_by_pred.md) -- what it costs to read
  what the previous dispatch wrote, from elsewhere. About 6% off-hart, 10%
  cross-cluster, and it is a model fitted to three measured classes rather than
  64 independent measurements.

### The whole loop

[`docs/Feature/the_loop.md`](docs/Feature/the_loop.md) is the index: profile -> schedule ->
advice -> hint -> rewrite -> verify -> reprofile -> verdict, and which script
owns each arrow.

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

#### Choosing a solver (`--solver`)

| solver | what it does | when to use |
| --- | --- | --- |
| `milp` (default) | global cvxpy/MOSEK optimum | rarely usable — of the eleven spike/FireSim workloads only the 212-op `yolov8_only_spike` solves at all; see the scaling note below |
| `greedy` | list scheduling, earliest completion for every op | baseline heuristic |
| `greedy_periodic` | same, but non-periodic ops get picked first | non-periodic critical path you don't want fragmented |
| `greedy_reserved` | `greedy_periodic` ordering + periodic ops take the *least contended* lane that still meets their deadline instead of the fastest one | heterogeneous multi-lane targets where a periodic job would otherwise squat on the accelerator the makespan-critical job needs |
| `heft` | HEFT: order by upward rank (longest path to a sink) rather than earliest completion, place by earliest finish with gap insertion | best makespan on several workloads — but it is deadline-blind, so check the window misses before trusting it |
| `heft_edf` | HEFT's rank ordering for non-periodic ops, earliest-deadline-first for periodic ops, periodic banded above non-periodic | the strongest single picker by validity (8/11 RISC-V, 23/30 generated) |
| `pso`, `sa` | particle swarm / simulated annealing over a random-key encoding, seeded from every heuristic | research use — they match the best heuristic and have not beaten it; `--search-budget` sets the per-pass wall clock |
| `cpsat` | OR-Tools CP-SAT: interval variables + one `AddNoOverlap` per machine | the exact method that actually scales — beats MOSEK at every size measured (212, 242 and 271 ops) and still solves at 677 where the MILP cannot finish building. Needs `XPURT_CPSAT_PYTHON` pointing at an interpreter with `ortools` |
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
eleven spike/FireSim (Flow A / RISC-V) workloads, reporting the objective
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

**MOSEK** — the solver these scripts default to (`--solver MOSEK`) — is a
separate, license-gated product: `pip install mosek` adds the Python
package (no license needed just to install it), but actually solving
requires a license file (`MOSEKLM_LICENSE_FILE`) from mosek.com.

**These bridge scripts' `--solver` flag does not work today**, for either
value. `scripts/run_xpurt_scheduler.py:328` calls
`schedule(workload, cvxpy_solver=args.solver, ...)`, but
`xpu-rt/scheduler.py`'s `schedule()` has no `cvxpy_solver` parameter and no
`**kwargs`, so the call raises `TypeError` before any solve;
`scripts/run_xpurt_scheduler_multi.py:362` does
`from schedulers import get_scheduler`, and no `schedulers` module exists
under `xpu-rt/`. `xpu-rt/scheduler.py` hardcodes `solver=cp.MOSEK` at four
call sites (lines 289, 641, 643, 872), so MOSEK is the only backend the
MILP path can currently use.

For the record, of the solvers cvxpy has installed in the `xpurt` env —
`CLARABEL`, `DAQP`, `HIGHS`, `MOSEK`, `OSQP`, `SCIPY`, `SCS` — only
**HIGHS, MOSEK and SCIPY** can solve this model at all: the rest are
continuous-only and reject the boolean variables outright (verified by
handing each a 3-boolean toy MIP). HiGHS is free, open source, already
installed, and cvxpy-drivable, so it is the obvious first alternative to
benchmark once a `--cvxpy-solver` knob exists. OR-Tools CP-SAT is the
interesting non-cvxpy candidate — this is a disjunctive no-overlap
scheduling model, which is what its interval variables are built for — but
it is not installed and would be a reformulation, not a solver swap.

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
[`zephyr-chipyard-sw/modelblaster/README.md`](ModelBlaster/README.md), section
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

`docs/Feature/the_loop.md` is the index for all of it.

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
