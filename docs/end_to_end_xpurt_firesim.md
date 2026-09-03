# End-to-end: workload spec → optimized xpurt schedule → FireSim run

This is the canonical "I have a multi-network workload, get me a FireSim
result with a Gantt plot" walkthrough. It glues together the docs that
exist already for each individual stage; each section lists the script
you run, the file(s) it consumes, the file(s) it produces, and a
pointer to the more detailed doc.

```
data/toplevel/networks_<name>.json     ← you write this
        │
        │  scripts/run_xpurt_schedule.py
        ▼
schedules/scheduled_networks_<name>_<solver>_profiled.json
        │
        │  modelblaster/examples/xpurt_demo/run.sh
        ▼
zephyr-chipyard-sw/modelblaster/examples/xpurt_demo/<quant>/build/<targets>_firesim/zephyr/zephyr.elf
        │
        │  firesim runworkload (called by run.sh)
        ▼
/scratch2/dima/chipyard-fsim/.../uartlog  (per-dispatch trace CSV)
        │
        │  modelblaster/scripts/plot_ros_trace.py / scripts/plot_scheduled_json.py
        ▼
plots/<name>_<solver>_predicted_vs_actual.png
        │
        │  scripts/plot_microros_vs_xpurt.py  (optional)
        ▼
plots/microros_vs_xpurt_<name>.png         ← compare against the
                                             microros baseline
```

## 1. Workload spec

**Write:** `data/toplevel/networks_<name>.json`

Schema (excerpt — see existing files for full examples):

```json
{
  "_comment": "Short description.",
  "hardware": {
    "machines":   { "cpu_p": 1, "cpu_e": 1 },
    "profile_hw": { "cpu_p": "V256D128_rvv", "cpu_e": "V256D128_rvv" },
    "profile":    { "target": "firesim_rocket_saturn",
                    "topo_tag": "topo_0",
                    "topo_tag_override": true }
  },
  "scheduler": {
    "random_seed": 42, "solver_verbosity": 2, "time_limit": 60,
    "use_profiled": true, "prune_periodic": true,
    "restrict_makespan_to_nonperiodic": true
  },
  "networks": {
    "<net_name>": {
      "id": 0, "identifier": "<net_name>",
      "dispatch_deps_path": "<path/to/<net>_dispatch_graph.json>",
      "preferred_hw": "RVV",
      "period": 20,           // optional, ms; omit for one-shot
      "window_duration": 20   // optional, deadline ms
    }
  }
}
```

Existing spec files to clone from:

* `data/toplevel/networks_periodic_dronet50ms_yolov8_firesim.json` — 2-net dronet+yolov8
* `data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json` — 3-net with Q0.31 profile
* `data/toplevel/networks_dronet50_yolov8_firesim_static_int8.json` — int8 path

The `dispatch_deps_path` per network points at a graph JSON describing
that network's ops + their inter-op dependencies. Today we have two
canonical sources for these:

* From the **Merlin** flow (BananaPi / SpacemiT): see `FreshScheduler/README.md`,
  output under `gen/vmfb/<net>/.../<net>.fp32_dispatch_graph.json`.
* From the **modelblaster** pipeline (FireSim/Zephyr): output under
  `zephyr-chipyard-sw/gen/qnn_vmfb/<net>/.../<net>_dispatch_graph.json`
  (after running `modelblaster/examples/<net>/run.sh`).

The dispatch graph format is the same in both; pick whichever flow has
already generated graphs for the networks in your spec.

## 2. Profile data (per network × backend × target)

The scheduler needs per-op execution time on the target hardware. The
profile data is keyed by `(network, backend, target, topo_tag)`.

**Where it must live:** `profile_loader.find_profile_csv` globs
`<FreshScheduler root>/gen/profile/<hw>/<target>/<model>/<model>.<quant>/*/<topo_tag>/results.csv`
— i.e. the TOP-LEVEL `gen/profile/`, not `zephyr-chipyard-sw/gen/profile/`.
The `sweep_v8` tree under `zephyr-chipyard-sw/gen/profile/` is only a
backing store that the top-level tree symlinks into; a CSV that exists
in `sweep_v8` but has no top-level path is invisible to the scheduler.
The loader is `strict=True` by default, so a genuinely missing
(network, hw) cell aborts the run rather than silently substituting
random times — if the scheduler completed, every cell it needed was
present.

**Build with the curated kernels.** `examples/<net>/run.sh` picks
REFERENCE (scalar) kernels unless `GLOBAL_CURATED_DIR` points at
`modelblaster/kernels`. Profiling without it measures the wrong
binary — it made dronet-on-gemmini_q31 look like 527 M cycles instead
of 14 M (37x). Always:

```bash
export GLOBAL_CURATED_DIR="$(pwd)/modelblaster/kernels"
```

**Zero-cost ops.** `view` / `chunk*` ops carry a `dispatch_id` in the
IR (and therefore in the dispatch graph) but generate no kernel, so
they never appear in `MODELBLASTER_PROFILE`. The strict loader then
rejects the network for a missing dispatch_id. Pad them into
`results.csv` as explicit 0-cycle rows rather than dropping them from
the dispatch graph — the walker keeps them as `-1`-sentinel entries and
the DAG edges route through them (see
`modelblaster/notes/xpurt_walker_semantics.md` row 10).

**Run** (per network, once per backend you'll allow the scheduler to
pick):

```bash
cd zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh

TARGET=gemmini_q31 BACKEND=reference QUANT=int8 \
  RUNNER=firesim FIRESIM_TIMEOUT=600 \
  bash modelblaster/examples/<net>/run.sh

TARGET=rvv BACKEND=reference QUANT=int8 \
  RUNNER=firesim FIRESIM_TIMEOUT=600 \
  bash modelblaster/examples/<net>/run.sh
```

Each invocation:
1. Generates the network's IR (`extract_graph`) and codegen for that
   backend (`generate_skeleton`, `generate_kernels`).
2. Builds the harness ELF, runs on FireSim, captures per-op wall
   cycles into `gen/profile/<backend>/<target>/<net>/.../results.csv`.

See `modelblaster/notes/pipeline_overview.md` for the single-model flow
breakdown and `modelblaster/README.md` for the env-var reference (TARGET,
BACKEND, QUANT, OPTIMIZE, RUNNER).

## 3. Schedule

**Run** (from `FreshScheduler` root):

```bash
/scratch2/dima/miniforge3/envs/xpurt/bin/python scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_<name>.json \
  --solver greedy \
  --profiled
```

The flag is `--networks-json`, not `--networks`. Use the `xpurt` conda
env's interpreter (the scheduler needs numpy/cvxpy; the `zephyr` env
does not have them).

Solver choices:
* `greedy` — fast, list-scheduling heuristic. The working default.
* `greedy_periodic` — greedy with explicit periodic-priority handling
* `decomposed` — pruned MILP per periodic-window slice (best quality).
  Needs `ortools`/`pulp`, neither of which is installed in the `xpurt`
  env as of 2026-08, so this cannot run today.

**Output:**
* `schedules/scheduled_networks_<name>_<solver>_profiled.json` — the
  ready-to-execute schedule
* `plots/networks_<name>_<solver>_profiled.png` — Gantt of the
  predicted schedule
* (optional) `plots/<name>_predicted_vs_actual.png` once you've also
  recorded an actual trace from step 5 below

The schedule JSON format is documented in
`zephyr-chipyard-sw/modelblaster/notes/scheduler_investigation.md`.
xpurt-walker semantics (how an entry's `hardware_target`,
`time_dependency`, etc. become C dispatch records) live in
`modelblaster/notes/xpurt_walker_semantics.md`.

## 4. Codegen + build the harness binary

**Run:**

```bash
cd zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh

SCHEDULE_JSON=/scratch2/dima/misc_sw/FreshScheduler/schedules/scheduled_networks_<name>_<solver>_profiled.json \
  MODELS=<net1>,<net2>,...  \
  BACKENDS=gemmini_q31,rvv \
  QUANTS=int8,int8,fp32 \
  REGISTRY=$(pwd)/modelblaster/cores/chipyard_dual_rocket_gemmini_q31.json \
  CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv \
  RUNNER=firesim FIRESIM_TIMEOUT=900 \
  bash modelblaster/examples/xpurt_demo/run.sh
```

What this does (per `modelblaster/examples/xpurt_demo/run.sh`):

1. Re-runs each constituent `modelblaster/examples/<net>/run.sh` (unless
   `FORCE_REGEN=0`) so per-network IR + per-backend kernel TUs exist.
2. `ingest_xpurt_schedule.py` reads the schedule.json, validates each
   `(network, dispatch_id)` against that network's IR, resolves
   `hardware_target → (core_name, core_kind, hart)`, and emits a flat
   `dispatch_table.c`. Critical detail: it builds an
   `ir_dispatch_id → codegen_idx` remap because zero-cost ops
   (`view`, `chunk*`) are filtered from the codegen table —
   see `modelblaster/notes/xpurt_walker_semantics.md` for the trap.
3. `generate_xpurt_main.py` emits the `main.c` that initializes
   `agents_pool` and walks the dispatch table.
4. `west build` links every model × every backend you listed.
5. Boots on FireSim via `firesim_runner.py` (which now calls
   `firesim infrasetup` before `runworkload` — see
   `modelblaster/validation/firesim_runner.py`).
6. Captures `MODELBLASTER_XPURT_TRACE` rows from the uartlog if
   `XPURT_TRACE=1`.

### 4b. On AWS F2 (the `fq` job queue) — build only, run elsewhere

`firesim_runner.py` drives `firesim infrasetup` + `runworkload` against
a LOCAL chipyard tree (`/scratch2/dima/chipyard-fsim`, a U250 run farm).
The F2 campaign does not use that path: the manager is
`ubuntu@3.88.218.39` and the FPGA lanes are driven by `deploy/fpga_queue`.
So stop the flow after the link step and submit the ELF yourself:

```bash
cd zephyr-chipyard-sw
source scripts/activate_conda.sh && source scripts/set_envvars_sdk.sh
export PYTHONPATH=$(pwd)
export GLOBAL_CURATED_DIR=$(pwd)/modelblaster/kernels   # else REFERENCE kernels

SCHEDULE_JSON=<abs path to schedules/scheduled_*.json> \
  MODELS=mlp_control,dronet,yolov8_nano \
  QUANTS=fp32,int8,int8 \
  BACKENDS=gemmini_q31,rvv \
  REGISTRY=$(pwd)/modelblaster/cores/chipyard_dual_rocket_gemmini_q31.json \
  CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv \
  FORCE_REGEN=0 XPURT_TRACE=1 SCHED_NAME=<tag> \
  STOP_AFTER=build RUNNER=firesim \
  bash modelblaster/examples/xpurt_demo/run.sh
# -> examples/xpurt_demo/int8/build/gemmini_q31_rvv_firesim/zephyr/zephyr.elf
```

`STOP_AFTER=build` is honored by `xpurt_demo/run.sh` (it mirrors the
same knob in `examples/_run_lib.sh`). Then:

```bash
MGR=ubuntu@3.88.218.39 ; KEY=~/.ssh/firesim.pem
scp -i $KEY <elf> $MGR:/home/ubuntu/<name>.elf
ssh -n -i $KEY $MGR "cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock && \
  ./bin/fq submit --tree /home/ubuntu/chipyard-rose \
     --hw-config f2_dual_small_norose_tacit_q31_60mhz \
     --elf /home/ubuntu/<name>.elf --timeout 5400 --results /home/ubuntu/<name>_res"
```

`experiments/profile_matrix.sh` in the RoSE repo is a working reference
for the scp + submit + poll + collect pattern (including the ELF-name
and model-banner provenance check on the collected uartlog). Every
`ssh` in a collection loop needs `</dev/null` or it eats the rest of
the job list off the loop's stdin.

`QUANTS` is parallel to `MODELS`: `mlp_control` is fp32, `dronet` and
`yolov8_nano` are int8. `FORCE_REGEN=0` skips re-running each model's
`run.sh` when `generated/<backend>/model.h` already exists.

The per-target Kconfig overlay comes out right on its own here:
`xpurt_demo/run.sh` picks `firesim_chipyard_dual_gemmini.conf` (2 harts)
because `BACKENDS` contains a gemmini variant, and `harness_xpurt`
auto-appends its own `backends/rvv.conf` (the `CONFIG_RISCV_ISA_EXT_V`
stanza) because `BACKENDS` also contains `rvv`.

## 5. Inspect the FireSim run

The harness prints `MODELBLASTER_WALL_CYCLES [<net>]` per network (plus
`MODELBLASTER_WALL_CYCLES_INST [<net>#<i>]` per periodic instance) when
each finishes. On the local U250 path the raw uartlog lands at
`/scratch2/dima/chipyard-fsim/sims/firesim/firesim_rundir/sim_slot_0/uartlog`;
on the F2 path pull it out of the job's `--results` dir:

```bash
ssh -n -i ~/.ssh/firesim.pem ubuntu@3.88.218.39 \
  "find /home/ubuntu/<name>_res -name uartlog | head -1 | xargs -r cat" \
  > data/xpurt_<name>_q31_firesim.txt
```

Always check provenance before analysing — job ids are reused across
daemon restarts. The ELF name FireSim embeds in the guest command line
must match what you submitted.

Per-dispatch predicted-vs-actual Gantt — the tool is
**`plot_xpurt_trace.py`**, which reads the
`MODELBLASTER_XPURT_TRACE_BEGIN..END` block that `XPURT_TRACE=1` emits:

```bash
python -m modelblaster.scripts.plot_xpurt_trace \
  data/xpurt_<name>_q31_firesim.txt \
  --clock-mhz 1 --source firesim \
  --out plots/xpurt_<name>_q31_predicted_vs_actual.png \
  --csv plots/xpurt_<name>_q31_trace.csv
```

It renders both panels (XPU-RT's predicted schedule on top, measured
execution below, red border = ran past its predicted finish) and prints
the predicted/actual makespan ratio. `plot_ros_trace.py` is NOT the tool
for an xpurt log — it looks for `MODELBLASTER_ROS_TRACE` markers, which
only `harness_microros` emits.

> The trace's `actual_*_cycles` are `k_cycle_get_64()` mtime ticks, 1 µs
> each at the modeled 1 GHz SoC frequency, hence `--clock-mhz 1`. The
> separate per-op `MODELBLASTER_PROFILE` cycles are rdcycle-based at the
> same modeled 1 GHz, which is why `profile_writer` is fed
> `--profile-clock-mhz 1000`. Both end up in ms consistently.

Predicted-schedule-only Gantt (`plot_scheduled_json.py` takes a
POSITIONAL json path and `--save`; there is no `--schedule/--actual/--out`):

```bash
/scratch2/dima/miniforge3/envs/xpurt/bin/python scripts/plot_scheduled_json.py \
  schedules/scheduled_networks_<name>_<solver>_profiled.json \
  --save plots/<name>_<solver>_predicted_schedule.png
```

### Known defect: mixed-backend networks get the wrong weight packing

`harness_xpurt/CMakeLists.txt` links `weights.c` **once per model**, from
the primary (first) backend's `generated/<bs>/` dir, on the assumption
recorded in its own comment that "all copies are identical". That is no
longer true: `generate_skeleton` now resolves weight layout per op and
per backend, so `gemmini_q31` packs conv2d_s8 weights **HWIO** while
`rvv` packs them **IHWOC** (grep `backend-packed layout:` at the top of
each `weights.c`). `backend_rename.py` deliberately does not rename
weight symbols, for the same stale reason.

Consequence: any op the schedule routes to the NON-primary backend reads
mis-permuted weights and produces wrong numbers, while single-network
runs of the same models verify bit-exact. Observed on the F2 3-net run:
dronet emitted `[0, 127]` against a golden of `[-56, 127]`.

**Timing is unaffected** — a conv kernel does the same MAC count
whatever the weight permutation, and the harness's per-op cycles match
the standalone single-network profiles to within ~4% on the affected
path. Treat mixed-backend *numerics* as untrusted until weight symbols
are renamed per backend and one `weights.c` is linked per
(model, backend).

## 6. Compare against the microros baseline (optional)

Same workload run through the fixed-pinning micro-ROS harness gives
a reference point that isolates "what does the scheduler buy you over
naive per-net pinning". The flow for the baseline lives in
`zephyr-chipyard-sw/modelblaster/examples/microros_demo/ROS_FLOW.md`. Once
you have a microros uartlog too:

```bash
python3 scripts/plot_microros_vs_xpurt.py \
  --microros "microros 3-net Config B=data/microros_3net_q31_firesim_configB.txt" \
  --xpurt    "xpurt 3-net Q31=data/xpurt_<name>_q31_firesim.txt" \
  --out plots/microros_vs_xpurt_<name>.png
```

## Pre-reqs (one time per shell)

```bash
cd /scratch2/dima/misc_sw/FreshScheduler

# 1. Conda env + Zephyr SDK
cd zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
cd ..

# 2. API keys (only if running BACKEND=llm or --optimize anywhere)
source set_api_keys.sh
```

## Where each detail lives

| concern | doc |
|---|---|
| Workload spec schema (`networks_*.json`) | This doc §1; example files under `data/toplevel/` |
| Single-network PyTorch → ELF | `zephyr-chipyard-sw/modelblaster/notes/pipeline_overview.md` |
| Schedule JSON format + walker | `zephyr-chipyard-sw/modelblaster/notes/xpurt_walker_semantics.md` |
| Scheduler invocation flags | `scripts/run_xpurt_schedule.py --help`; module docs in `xpu-rt/` |
| FireSim harness env knobs | `modelblaster/examples/xpurt_demo/run.sh` header |
| Co-execution baseline matrix | `modelblaster/notes/firesim_co_execution_baseline_plan.md` |
| micro-ROS reference flow | `modelblaster/examples/microros_demo/ROS_FLOW.md` |
| RVV / Gemmini / Q31 gotchas | `modelblaster/notes/gemmini_extension_plan.md`, `modelblaster/notes/zephyr_rvv_fix_summary.md`, `modelblaster/notes/saturn_strided_memop_bug.md` |
| Top-level Merlin/SpacemiT path | `FreshScheduler/README.md` |
