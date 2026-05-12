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
        │  agents/examples/xpurt_demo/run.sh
        ▼
zephyr-chipyard-sw/agents/examples/xpurt_demo/<quant>/build/<targets>_firesim/zephyr/zephyr.elf
        │
        │  firesim runworkload (called by run.sh)
        ▼
/scratch2/dima/chipyard-fsim/.../uartlog  (per-dispatch trace CSV)
        │
        │  agents/scripts/plot_ros_trace.py / scripts/plot_scheduled_json.py
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
* From the **agents** pipeline (FireSim/Zephyr): output under
  `zephyr-chipyard-sw/gen/qnn_vmfb/<net>/.../<net>_dispatch_graph.json`
  (after running `agents/examples/<net>/run.sh`).

The dispatch graph format is the same in both; pick whichever flow has
already generated graphs for the networks in your spec.

## 2. Profile data (per network × backend × target)

The scheduler needs per-op execution time on the target hardware. The
profile data is keyed by `(network, backend, target, topo_tag)` and
lives under `zephyr-chipyard-sw/gen/profile/...` for FireSim runs.

**Run** (per network, once per backend you'll allow the scheduler to
pick):

```bash
cd zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh

TARGET=gemmini_q31 BACKEND=reference QUANT=int8 \
  RUNNER=firesim FIRESIM_TIMEOUT=600 \
  bash agents/examples/<net>/run.sh

TARGET=rvv BACKEND=reference QUANT=int8 \
  RUNNER=firesim FIRESIM_TIMEOUT=600 \
  bash agents/examples/<net>/run.sh
```

Each invocation:
1. Generates the network's IR (`extract_graph`) and codegen for that
   backend (`generate_skeleton`, `generate_kernels`).
2. Builds the harness ELF, runs on FireSim, captures per-op wall
   cycles into `gen/profile/<backend>/<target>/<net>/.../results.csv`.

See `agents/notes/pipeline_overview.md` for the single-model flow
breakdown and `agents/README.md` for the env-var reference (TARGET,
BACKEND, QUANT, OPTIMIZE, RUNNER).

## 3. Schedule

**Run** (from `FreshScheduler` root):

```bash
python scripts/run_xpurt_schedule.py \
  --networks data/toplevel/networks_<name>.json \
  --solver decomposed \
  --profiled
```

Solver choices:
* `greedy` — fast, list-scheduling heuristic
* `greedy_periodic` — greedy with explicit periodic-priority handling
* `decomposed` — pruned MILP per periodic-window slice (best quality)

**Output:**
* `schedules/scheduled_networks_<name>_<solver>_profiled.json` — the
  ready-to-execute schedule
* `plots/networks_<name>_<solver>_profiled.png` — Gantt of the
  predicted schedule
* (optional) `plots/<name>_predicted_vs_actual.png` once you've also
  recorded an actual trace from step 5 below

The schedule JSON format is documented in
`zephyr-chipyard-sw/agents/notes/scheduler_investigation.md`.
xpurt-walker semantics (how an entry's `hardware_target`,
`time_dependency`, etc. become C dispatch records) live in
`agents/notes/xpurt_walker_semantics.md`.

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
  REGISTRY=$(pwd)/agents/cores/chipyard_dual_rocket_gemmini_q31.json \
  CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv \
  RUNNER=firesim FIRESIM_TIMEOUT=900 \
  bash agents/examples/xpurt_demo/run.sh
```

What this does (per `agents/examples/xpurt_demo/run.sh`):

1. Re-runs each constituent `agents/examples/<net>/run.sh` (unless
   `FORCE_REGEN=0`) so per-network IR + per-backend kernel TUs exist.
2. `ingest_xpurt_schedule.py` reads the schedule.json, validates each
   `(network, dispatch_id)` against that network's IR, resolves
   `hardware_target → (core_name, core_kind, hart)`, and emits a flat
   `dispatch_table.c`. Critical detail: it builds an
   `ir_dispatch_id → codegen_idx` remap because zero-cost ops
   (`view`, `chunk*`) are filtered from the codegen table —
   see `agents/notes/xpurt_walker_semantics.md` for the trap.
3. `generate_xpurt_main.py` emits the `main.c` that initializes
   `agents_pool` and walks the dispatch table.
4. `west build` links every model × every backend you listed.
5. Boots on FireSim via `firesim_runner.py` (which now calls
   `firesim infrasetup` before `runworkload` — see
   `agents/validation/firesim_runner.py`).
6. Captures `AGENTS_XPURT_TRACE` rows from the uartlog if
   `XPURT_TRACE=1`.

## 5. Inspect the FireSim run

The runner streams uartlog and prints `AGENTS_WALL_CYCLES` per network
when each finishes. The raw uartlog lands at
`/scratch2/dima/chipyard-fsim/sims/firesim/firesim_rundir/sim_slot_0/uartlog`.

Snapshot it (helps with comparison and re-plotting later):

```bash
cp /scratch2/dima/chipyard-fsim/sims/firesim/firesim_rundir/sim_slot_0/uartlog \
   data/xpurt_<name>_q31_firesim.txt
```

Per-dispatch trace plotting (Gantt):

```bash
python3 agents/scripts/plot_ros_trace.py \
  --uartlog data/xpurt_<name>_q31_firesim.txt \
  --clock-mhz 1 \
  --out plots/xpurt_<name>_q31_firesim.png \
  --title "xpurt schedule, <name>, Q31 firesim"
```

> mtime ticks are 1 µs at the modeled 1 GHz SoC frequency on FireSim,
> so `--clock-mhz 1`. See `agents/examples/microros_demo/ROS_FLOW.md`
> §4 for the same convention applied to microros runs.

Predicted-vs-actual schedule overlay (matches predicted bars against
the actual per-dispatch start/end times pulled from the uartlog):

```bash
python scripts/plot_scheduled_json.py \
  --schedule schedules/scheduled_networks_<name>_<solver>_profiled.json \
  --actual data/xpurt_<name>_q31_firesim.txt \
  --out plots/<name>_<solver>_predicted_vs_actual.png
```

## 6. Compare against the microros baseline (optional)

Same workload run through the fixed-pinning micro-ROS harness gives
a reference point that isolates "what does the scheduler buy you over
naive per-net pinning". The flow for the baseline lives in
`zephyr-chipyard-sw/agents/examples/microros_demo/ROS_FLOW.md`. Once
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
| Single-network PyTorch → ELF | `zephyr-chipyard-sw/agents/notes/pipeline_overview.md` |
| Schedule JSON format + walker | `zephyr-chipyard-sw/agents/notes/xpurt_walker_semantics.md` |
| Scheduler invocation flags | `scripts/run_xpurt_schedule.py --help`; module docs in `xpu-rt/` |
| FireSim harness env knobs | `agents/examples/xpurt_demo/run.sh` header |
| Co-execution baseline matrix | `agents/notes/firesim_co_execution_baseline_plan.md` |
| micro-ROS reference flow | `agents/examples/microros_demo/ROS_FLOW.md` |
| RVV / Gemmini / Q31 gotchas | `agents/notes/gemmini_extension_plan.md`, `agents/notes/zephyr_rvv_fix_summary.md`, `agents/notes/saturn_strided_memop_bug.md` |
| Top-level Merlin/SpacemiT path | `FreshScheduler/README.md` |
