# Reproducing the mlp_control + dronet + yolov8_nano XPU-RT schedule on spike

Spike-only (no FireSim) reproduction of the schedule-generation + combined-binary
flow for `networks_periodic_mlp10ms_dronet20ms_yolov8_firesim.json`'s topology:
profile each model on spike, generate an XPU-RT schedule from that profile
data, build+run the combined multi-network binary via modelblaster's
`xpurt_demo`, and (optionally) render the real spike execution as a timeline
via `XPURT_TRACE=1` + `plot_xpurt_trace.py`.

**Expected result:** `OVERALL: PASS (3 models)` — `mlp_control` within fp32
tolerance, `dronet`/`yolov8_nano` bit-exact — with predicted-vs-actual
makespan matching to within ~0.05% (`dronet`/`yolov8_nano`, >95% of total
time, each within 0.1%). Known residual: `mlp_control`'s own ops run
~15-17% faster than predicted (see "Known limitations" below); this is a
small (~4% of schedule) and long-standing unexplained gap, not a
regression to chase during a normal repro run.

Repo layout reminder: `zephyr-chipyard-sw/` is a submodule of this repo, and
`zephyr-chipyard-sw/modelblaster/` is itself a nested submodule
(`ucb-bar/ModelBlaster.git`). A fix that touches modelblaster needs its own
3-commit chain to land: commit in `modelblaster` → bump the `modelblaster`
pointer inside `zephyr-chipyard-sw` → bump the `zephyr-chipyard-sw` pointer
in this repo.

## Reproduce from a fresh clone: exact steps

`scripts/repro_workload.sh` runs the entire flow below (sections
0/1/2/3/5/6, and optionally 7) as a single command, against whatever
workload spec you hand it:

```bash
bash scripts/repro_workload.sh data/toplevel/networks_mlp_dronet_yolo_spike.json --trace
```

(`scripts/repro_mlp_dronet_yolo_spike.sh` still exists as a shim that
forwards to it with that spec as the default, so older commands keep
working.)

It resolves the `zephyr-chipyard-sw`/`modelblaster` submodule checkouts from
`.gitmodules` (not a hardcoded path), activates the conda/Zephyr-SDK env,
profiles all 3 models on spike, generates dispatch graphs (with the
basename-rename fixup), bridges profile data via symlinks, generates the
schedule, builds+runs the combined `xpurt_demo` binary, and (with `--trace`)
renders the real-execution timeline. Expect `OVERALL: PASS (3 models)` and a
predicted/actual makespan within ~0.05% (see §10.3's caveat about
`mlp_control`'s own residual deviation). It also installs
`scripts/install_xpurt_deps.sh`'s dependencies into the active env before
doing any of that (see §0) — pass `--skip-deps` once that's already been
done for this env. Flags to skip already-fresh stages (`--skip-deps`,
`--skip-profile`, `--skip-dispatch`, `--skip-schedule`, `--skip-build`) are
documented in the script's `--help`.

The script is **driven entirely by the workload spec** — there is no
hardcoded model list, backend list or target anywhere in it. The JSON says
what to run and the script does that:

| spec field | drives |
| --- | --- |
| `hardware.profile.target` | `spike` → this doc's no-RTL flow; anything else (`firesim_rocket_saturn`, …) → `RUNNER=firesim`, i.e. `docs/Firesim/end_to_end_xpurt_firesim.md` |
| `hardware.profile.topo_tag` | `PROFILE_CORES` for the capture (`topo_0` → `0`, `topo_0_1` → `0,1`) |
| `hardware.profile_hw` | which HW backends get profiled, get dispatch graphs, and are linked into the binary. Each value is both the on-disk profile dir and (mapped: `RVV`/`<cfg>_rvv` → `rvv`, plus `scalar`, `gemmini`, `gemmini_q31`) a modelblaster `TARGET=`. `cpu_p`/`cpu_e` also pick `CPU_P_KIND`/`CPU_E_KIND` |
| `networks[*].identifier` | which models get profiled and built (deduped, in spec order) |
| `networks[*].quant` | each model's modelblaster quant tree (§4). Absent → inferred from the `<model>.<quant>/` basename in `dispatch_deps_path`, else `fp32` |
| `networks[*].dispatch_deps_path` | that network's IREE target and the dispatch-graph basename step 2 renames its output to — per network, so a spec may mix a spike-emitted graph with a FireSim one |
| `scheduler.*` | `run_xpurt_schedule.py`, as always |
| `flow.*` | optional, read by this script only (the scheduler ignores it): `flow.solver`, `flow.trace`, `flow.profile.{out_root,clock_mhz}`, `flow.build.{registry,backends,firesim_conf,firesim_timeout,force_regen}` |

So the single-network repros of §9 are just a different JSON:

```bash
bash scripts/repro_workload.sh data/toplevel/networks_dronet_only_spike.json
```

and a FireSim workload is the same command again:

```bash
bash scripts/repro_workload.sh data/toplevel/networks_periodic_dronet_yolov8_firesim.json
```

`--dry-run` prints the fully resolved plan (models, quants, backends, core
kinds, registry, schedule path) and exits without running anything — worth
doing first on a spec you haven't run before. `--quants` overrides the
spec's per-model quants (one comma-separated entry per model, in spec
order) when you want the same topology at a different precision without
editing the spec, and `--solver`/`--trace`/`--no-trace` override
`flow.solver`/`flow.trace`.

The one thing the spec can't express is the **bitstream**, which is what
the core registry encodes. `flow.build.registry` names it; without one the
script takes `xpurt_demo`'s default on spike (nothing to get wrong there),
otherwise the single `modelblaster/cores/*.json` whose core kinds are
exactly the spec's `cpu_p`/`cpu_e` pair — and refuses to guess when
several match, listing them.

The sections below (including the from-scratch environment setup in
"Prerequisites") are the manual, step-by-step walkthrough the script
automates — read them for the "why" behind each stage and the bugs each
workaround is standing in for. Useful on its own for debugging a stage in
isolation, or if the script's assumptions (model list, quants, backends)
don't fit a variant you're trying.

## Prerequisites: building the `zephyr` conda env + Zephyr SDK from scratch

Everything below §0 assumes the `zephyr` conda env, the Zephyr SDK, and the
`west` workspace already exist. Two earlier passes through this doc
(including a from-scratch *repo* clone) explicitly skipped this by
symlinking in an already-installed toolchain — this section closes that
gap, verified via a genuine from-scratch install (conda + SDK + full
pipeline run) on 2026-08-27. Good news: unlike the unrelated `xpurt`/Isaac
Sim env elsewhere in this repo (see `docs/xpurt_env_setup.md`), **this
environment already has a real, committed setup script** —
`zephyr-chipyard-sw/README.md`'s "Standalone Installation" section plus
`scripts/install_conda.sh` / `install_submodules.sh` / `install_toolchain_sdk.sh`.
This section is just that flow, run for real, with the gaps it hit and the
extra (undocumented) packages this specific pipeline needs on top of it.

```bash
# 1. Clone WITHOUT --recursive. This repo's top-level .gitmodules
#    registers 4 submodules (merlin, zephyr-chipyard-sw, sims/IsaacLab,
#    hw/chipyard) and zephyr-chipyard-sw itself registers 11 more
#    (zephyr_ws/zephyr, modelblaster, XNNPACK, executorch, tinympc,
#    riscv-gnu-toolchain, ...) -- `--recursive` would clone ALL of them,
#    including hw/chipyard (the full Chipyard RTL/toolchain checkout,
#    SSH-only, very large) and sims/IsaacLab (a large robotics sim
#    framework), neither of which this spike-only flow touches. Init only
#    the two submodules this flow actually needs, explicitly:
git clone git@github.com:ucb-bar/XPU-RT.git
cd XPU-RT
git submodule update --init zephyr-chipyard-sw
git -C zephyr-chipyard-sw submodule update --init modelblaster

# 2. One-time toolchain bootstrap: conda env + west workspace + Zephyr SDK.
#    Only needed once per machine (or whenever tools/miniforge3 /
#    tools-manual/zephyr-sdk-1.0.0-beta1 don't already exist).
cd zephyr-chipyard-sw
source scripts/install_conda.sh
bash scripts/install_submodules.sh
bash scripts/install_toolchain_sdk.sh
cd ..

# 3. Run the full repro: installs xpu-rt's + modelblaster's own deps into
#    the zephyr env, profiles all 3 models on spike, generates dispatch
#    graphs + schedule, builds + runs the combined xpurt_demo binary, and
#    renders the real-execution timeline.
bash scripts/repro_mlp_dronet_yolo_spike.sh --trace
```

Expect `OVERALL: PASS (3 models)` (see "Expected result" above). Re-running
later on the same machine: skip step 2, and pass `--skip-deps` in step 3
once `install_xpurt_deps.sh` has already run for this env.

`scripts/repro_mlp_dronet_yolo_spike.sh` resolves the
`zephyr-chipyard-sw`/`modelblaster` submodule checkouts from `.gitmodules`
(not a hardcoded path) and activates the conda/Zephyr-SDK env itself. Flags
to skip already-fresh stages (`--skip-deps`, `--skip-profile`,
`--skip-dispatch`, `--skip-schedule`, `--skip-build`) are documented in the
script's `--help`.

The sections below are the manual, step-by-step walkthrough the script
automates — useful for debugging a single stage in isolation, or for a
variant (different model list, quants, backends) the script doesn't cover.

## Prerequisites

Two host-specific gaps seen on shared Linux hosts that step 2 above can hit
(skip if `install_toolchain_sdk.sh`/`west build` just work for you):

- **Missing `libidn.so.11`** during SDK CMake package registration
  (`cmake: error while loading shared libraries: libidn.so.11`). Fix:
  ```bash
  mkdir -p tools-manual/compat_lib
  ln -sf /lib/x86_64-linux-gnu/libidn.so.12 tools-manual/compat_lib/libidn.so.11
  cd tools-manual/zephyr-sdk-1.0.0-beta1
  LD_LIBRARY_PATH="$PWD/../compat_lib:${LD_LIBRARY_PATH:-}" ./setup.sh -c
  cd -
  cp tools/patches/generic.cmake tools/patches/target.cmake \
     tools-manual/zephyr-sdk-1.0.0-beta1/cmake/zephyr/
  ```
  This only affects `find_package(Zephyr-sdk)` auto-discovery; `west build`
  finds the SDK via `set_envvars_sdk.sh`'s `ZEPHYR_SDK_INSTALL_DIR` either way.
- **An ancient Xilinx-Vitis-bundled `cmake` shadows the real one on PATH**,
  causing `west build` to fail with `cannot get cmake version: ... exit
  status 127`. Check `which cmake`; this is why `export PATH="/usr/bin:...`"
  appears in §0 below.

Sanity check:
```bash
pip install spike==0.0.5.dev20   # public PyPI wheel, despite the git-hash-looking version string
west build -p -b spike_riscv64 samples/hello_world/ -d /tmp/hello_build
spike /tmp/hello_build/zephyr/zephyr.elf   # expect "Hello World! spike_riscv64/rocketchip_virt_riscv64"
```

Beyond the base Zephyr install, this pipeline needs modelblaster's own deps
(torch, ultralytics, pillow), xpu-rt's own scheduler deps
(numpy/scipy/matplotlib/pandas), the pinned `spike` wheel above, and the
`libgl1` system package (`opencv-python`, pulled in by `ultralytics`, links
`libGL.so.1` at import time — absent on a bare headless host). All of this
is installed by one script, run from the top-level repo with the `zephyr`
env active:

```bash
pip install -r zephyr_ws/zephyr/scripts/requirements-base.txt
bash scripts/install_xpurt_deps.sh
```

`zephyr-chipyard-sw` itself stays fully standalone — none of the above is
installed by its own install scripts, so a consumer that only wants
Zephyr/RISC-V dev (e.g. a project embedding it as a submodule on an
unrelated branch) never needs any of this.

## 0. Environment setup (run before every command below)

```bash
export XPURT_ROOT=/path/to/your/XPU-RT   # <-- set this first
cd "${XPURT_ROOT}/zephyr-chipyard-sw"
source tools/miniforge3/etc/profile.d/conda.sh
conda activate zephyr
source scripts/set_envvars_sdk.sh
export PATH="/usr/bin:${PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"   # re-promote conda's python/pip ahead of /usr/bin's
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
```

The `zephyr` env above is produced entirely by zephyr-chipyard-sw's own
standalone install (`install_conda.sh`/`install_submodules.sh`/
`install_toolchain_sdk.sh`) — nothing xpurt-specific. Everything this
specific reproduction flow needs *on top of* that (modelblaster's own deps
— torch, `ultralytics`, pillow, ...; xpu-rt's own scheduler deps; the
pinned `spike` wheel; the `libgl1` system package for `ultralytics`'
`opencv-python`, see Bug 16) is declared and installed from ONE place in
*this* repo, into the same env:

```bash
cd "${FRESHSCHEDULER_ROOT}"
bash scripts/install_xpurt_deps.sh
```

`scripts/repro_workload.sh` (below) runs this automatically —
`--skip-deps` skips it once you've already run it for this env.

## 1. Profile each model on spike

All commands run from `zephyr-chipyard-sw/` with the env above active.
`PROFILE_CORES=0` + `PROFILE_CLOCK_MHZ=1000.0` match the FireSim-original
spec's assumptions; `TARGET=scalar`/`rvv` select the backend.

```bash
# mlp_control (fp32 only — this model has no int8 quant)
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
QUANT=fp32 TARGET=scalar BACKEND=reference bash modelblaster/examples/mlp_control/run.sh
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
QUANT=fp32 TARGET=rvv    BACKEND=reference bash modelblaster/examples/mlp_control/run.sh

# dronet (int8)
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=scalar BACKEND=reference bash modelblaster/examples/dronet/run.sh
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=rvv    BACKEND=reference bash modelblaster/examples/dronet/run.sh

# yolov8_nano (int8)
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=scalar BACKEND=reference bash modelblaster/examples/yolov8_nano/run.sh
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=rvv    BACKEND=reference bash modelblaster/examples/yolov8_nano/run.sh
```

Each run also verifies bit-exact (int8) or near-fp32-tolerance (fp32)
correctness standalone against the PyTorch golden. Output lands at
`zephyr-chipyard-sw/modelblaster/gen/profile/{scalar,RVV}/spike/<model>/<basename>/<run_tag>/topo_0/results.csv`.

**Quirk:** the profile CSV's `<basename>` directory is always tagged
`.fp32` regardless of the actual `QUANT` used. If you profile the same
model at both `int8` and `fp32`, the second run overwrites the first at
that path — don't run both quants back to back for the same model unless
you actually want that.

## 2. Generate/fix dispatch graphs

`emit_dispatch_graph.py` isn't covered in the top-level docs; invoke it
directly from `zephyr-chipyard-sw/modelblaster/`:

```bash
cd zephyr-chipyard-sw/modelblaster
for hw in scalar RVV; do
  python3 pipeline/emit_dispatch_graph.py \
    --ir examples/<model>/<quant>/generated/graph.json \
    --out-root ../gen/vmfb --target generic_riscv64 --hw "$hw"
done
```

Writes `../gen/vmfb/<model>/generic_riscv64/<hw>/<model>.<quant>/<model>.<quant>_dispatch_graph.json`.

**Basename mismatch:** the profile CSVs are always tagged `.fp32` (see
quirk above) but `emit_dispatch_graph.py` tags its output with the model's
*actual* quant (e.g. `.int8`). The scheduler matches dispatch graphs to
profile CSVs by basename, so an int8-sourced dispatch graph must be renamed
to `.fp32` after generation:

```bash
for hw in scalar RVV; do
  d="../gen/vmfb/<model>/generic_riscv64/$hw"
  mv "$d/<model>.int8" "$d/<model>.fp32"
  mv "$d/<model>.fp32/<model>.int8_dispatch_graph.json" "$d/<model>.fp32/<model>.fp32_dispatch_graph.json"
done
```

## 3. Bridge profile data into the top-level repo (`gen_root` workaround)

The workload spec's `hardware.profile.gen_root` field is **ignored** by
`xpu-rt/profile_loader.py`'s `find_profile_csv()`, which hardcodes
`<repo>/gen/profile/...` relative to the top-level repo. Since the real
data lives under the nested `zephyr-chipyard-sw/modelblaster/gen/`, bridge
it with symlinks (not fixed at the source — redo this if lost, e.g. after
`git clean`):

```bash
cd "${XPURT_ROOT}"
mkdir -p gen/profile/RVV/spike gen/profile/scalar/spike
for model in mlp_control dronet yolov8_nano; do
  ln -s "../../../../zephyr-chipyard-sw/modelblaster/gen/profile/RVV/spike/$model"    "gen/profile/RVV/spike/$model"
  ln -s "../../../../zephyr-chipyard-sw/modelblaster/gen/profile/scalar/spike/$model" "gen/profile/scalar/spike/$model"
done
```

## 4. Workload spec

`data/toplevel/networks_mlp_dronet_yolo_spike.json` (hand-written, not
generated):

```json
{
  "hardware": {
    "machines": { "cpu_p": 1, "cpu_e": 1 },
    "profile_hw": { "cpu_p": "RVV", "cpu_e": "scalar" },
    "profile": {
      "target": "spike",
      "topo_tag": "topo_0",
      "topo_tag_override": true,
      "gen_root": "zephyr-chipyard-sw/modelblaster/gen"
    },
    "p_core_speedup": 1.0
  },
  "scheduler": {
    "random_seed": 42,
    "solver_verbosity": 2,
    "time_limit": 60,
    "use_profiled": true,
    "prune_periodic": false,
    "restrict_makespan_to_nonperiodic": true
  },
  "networks": {
    "mlp_control": {
      "id": 0, "identifier": "mlp_control",
      "quant": "fp32",
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/mlp_control/generic_riscv64/RVV/mlp_control.fp32/mlp_control.fp32_dispatch_graph.json",
      "period": 10, "window_duration": 10
    },
    "dronet": {
      "id": 1, "identifier": "dronet",
      "quant": "int8",
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/dronet/generic_riscv64/RVV/dronet.fp32/dronet.fp32_dispatch_graph.json",
      "period": 1000, "window_duration": 1000
    },
    "yolov8_nano": {
      "id": 2, "identifier": "yolov8_nano",
      "quant": "int8",
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/yolov8_nano/generic_riscv64/RVV/yolov8_nano.fp32/yolov8_nano.fp32_dispatch_graph.json"
    }
  },
  "edges": []
}
```

`quant` is read by `scripts/repro_workload.sh` (not by the
scheduler, which ignores unknown keys) to pick each model's
`modelblaster/examples/<model>/<quant>/` tree for profiling, dispatch-graph
emission, and the combined build. It has to be declared explicitly because
the dispatch graphs are renamed to `.fp32` in §2 for the profile-CSV
basename match, so `dispatch_deps_path` no longer records the real
precision. Omitting it means `fp32`.

Notes on settings that changed during debugging (see Bugs 5/6/7 below):
- `period`/`window_duration` for mlp_control/dronet: 10ms/20ms (original,
  mirroring the FireSim spec) → 100ms/150ms (loosened, per-user request, for
  a fast test) → 1000ms/1000ms (isolation test to rule out "many periodic
  instances" as the cause of the dependency-cycle bug) → **mlp_control back
  to 10ms/10ms, dronet left at 1000ms/1000ms** (current — validated that the
  1-instance result was purely a consequence of period vs. makespan, and
  stress-tested the Bug 7 fix at a much higher periodic-instance count; see
  §5.1 below).
- `scheduler.prune_periodic`: `true` → **`false`** (works around Bug 5).
- `scheduler.time_limit: 60` only applies to the `milp` solver — it is
  **silently ignored** by `greedy_periodic` (used here). The only bound on
  `greedy_periodic`'s convergence loop is `--max-periodic-iters` (CLI flag,
  default 4) — this caps loop *iterations*, not wall-clock time, and setting
  it to `1` produces an invalid (non-converged) schedule, not just a fast
  one. Don't use it as a time-limit substitute.

`yolov8_nano`'s `dispatch_deps_path` currently points at the RVV
dispatch-graph file, which was **just regenerated** (mtime 14:45, see §2) —
newer than the schedule below (mtime 14:24). The schedule on disk was built
against the file's *previous* content, not this latest regeneration.

## 5. Generate the schedule

```bash
cd "${XPURT_ROOT}"
python3 scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_mlp_dronet_yolo_spike.json \
  --solver greedy_periodic --profiled
```

Outputs (both untracked, regenerated in place each run):
- `schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json`
- `plots/networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.png`

You'll see an informational `WARN: granularity advisor -- ...` line about
dispatch granularity — non-fatal, the schedule JSON is still written
correctly; not something to act on for this repro.

Expected result with the spec above: 67 `mlp_control` instances, 1 each of
`dronet`/`yolov8_nano`, 705 total dispatches, ~665ms predicted makespan
(non-periodic; 694ms across all operations), converging after 3 refinement
iterations.

**`--solver auto` gives a better schedule here**, and the repro script
takes the same flag (`bash scripts/repro_mlp_dronet_yolo_spike.sh
--solver auto`). It runs all four heuristics and keeps the best; on this
workload `greedy_reserved` wins by leaving dronet on the RVV lane and
letting yolov8_nano start earlier: **63 `mlp_control` instances, 677 total
dispatches, 628.94ms predicted non-periodic makespan** (657.35ms across
all operations), still `OVERALL: PASS (3 models)` on spike. Both solvers
land every `mlp_control` instance inside its 10ms window. The script's
default stays `greedy_periodic` so the numbers above remain the recorded
baseline; see the solver table in the top-level README for the full
comparison.

## 6. Build + run the combined binary (`xpurt_demo`)

```bash
cd "${XPURT_ROOT}/zephyr-chipyard-sw"
# (env setup from §0)
SCHEDULE_JSON="${XPURT_ROOT}/schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json" \
MODELS=mlp_control,dronet,yolov8_nano \
QUANTS=fp32,int8,int8 \
BACKENDS=scalar,rvv \
FORCE_REGEN=1 \
RUNNER=spike \
XPURT_TRACE=1 \
  bash modelblaster/examples/xpurt_demo/run.sh
```

**`FORCE_REGEN=1` is required** whenever any model's generator code
(`generate_skeleton.py`/`generate_kernels.py`) is newer than its already-generated
`generated/` output — with `FORCE_REGEN=0` the stale generated `model.c` is
reused as-is.

**Use `QUANTS` (plural), not `QUANT`**, for this mixed-quant build — a
blanket `QUANT=int8` with `FORCE_REGEN=0` will silently regenerate any
(model, quant) directory that doesn't already exist, including models that
should stay at a different quant (`mlp_control` has no int8 variant).

Expect `OVERALL: PASS (3 models)` — `mlp_control` (`max_abs_err` ~1e-7),
`dronet`/`yolov8_nano` bit-exact (`max_abs_err=0`).

## 7. Instrumented spike run → real execution timeline (optional)

`XPURT_TRACE=1` (already set in §6) makes the schedule-driven runtime emit a
per-dispatch CSV block (`=== MODELBLASTER_XPURT_TRACE_BEGIN ===` / `_END`)
into the run log, with both the scheduler's *predicted* timing and the
*actual* measured spike cycles per dispatch. Render it with the existing
`plot_xpurt_trace.py` tool:

```bash
cd "${XPURT_ROOT}/zephyr-chipyard-sw/modelblaster"
source ../tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
export PYTHONPATH="$PWD/..${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m modelblaster.scripts.plot_xpurt_trace \
  <path-to-the-run-log-you-captured-in-§6> \
  --clock-mhz 10 \
  --out "${XPURT_ROOT}/plots/xpurt_trace_mlp_dronet_yolo_spike.png" \
  --csv "${XPURT_ROOT}/schedules/xpurt_trace_mlp_dronet_yolo_spike.csv"
```
