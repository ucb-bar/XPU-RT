# Reproducing the mlp_control + dronet + yolov8_nano XPU-RT schedule on spike

Spike-only (no FireSim) reproduction of the schedule-generation + combined-binary
flow for `networks_periodic_mlp10ms_dronet20ms_yolov8_firesim.json`'s topology.
Goal: profile each model on spike, generate an XPU-RT schedule from that
profile data, build+run the combined multi-network binary via modelblaster's
`xpurt_demo`, and (optionally) render the real spike execution as a timeline
via `XPURT_TRACE=1` + `plot_xpurt_trace.py`.

**Current status (final):** the full 3-network schedule builds and runs
correctly. **All three models PASS** (`mlp_control` `max_abs_err=6.52e-08`;
`dronet` and `yolov8_nano` both bit-exact, `max_abs_err=0`). Overall
schedule timing matches predicted to within **0.05%** (704.997ms predicted
vs. 704.675ms actual), with `dronet`/`yolov8_nano` (>95% of total time) each
within 0.1%. One known, low-impact, not-fully-root-caused residual remains:
`mlp_control`'s own operators (only ~4% of total schedule time) run
~15-17% faster than predicted — see §10.3. Full root-cause chain: Bug 11
(runtime buffer overflow) → Bug 12 (weight-symbol collision across
backends, the main correctness+timing fix) → stale `yolov8_nano` profiling
data (§10.1) → final rebuild (§10).

Repo layout reminder: `zephyr-chipyard-sw/` is a submodule of this repo, and
`zephyr-chipyard-sw/modelblaster/` is itself a nested submodule (pointing at
`ucb-bar/ModelBlaster.git`). Two of the fixes below live inside that nested
submodule and need their own commit chain (modelblaster → zephyr-chipyard-sw
pointer bump → this repo's pointer bump) if kept.

## Quick start: end-to-end script

`scripts/repro_mlp_dronet_yolo_spike.sh` runs the entire flow below (sections
0/1/2/3/5/6, and optionally 7) as a single command:

```bash
bash scripts/repro_mlp_dronet_yolo_spike.sh --trace
```

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

The sections below are the manual, step-by-step walkthrough the script
automates — read them for the "why" behind each stage and the bugs each
workaround is standing in for. Useful on its own for debugging a stage in
isolation, or if the script's assumptions (model list, quants, backends)
don't fit a variant you're trying.

## 0. Environment setup (run before every command below)

Every absolute path in this doc is written relative to `FRESHSCHEDULER_ROOT` —
**set this to wherever you actually cloned this repo** (it is NOT
`/scratch2/dima/misc_sw/FreshScheduler` unless that happens to be your own
checkout; hardcoding one user's path here is exactly what broke this flow
for a second user on this same shared machine — every command below reads
`${FRESHSCHEDULER_ROOT}`, none of it should be copy-pasted with someone
else's literal path):

```bash
export FRESHSCHEDULER_ROOT=/path/to/your/FreshScheduler   # <-- set this first
cd "${FRESHSCHEDULER_ROOT}/zephyr-chipyard-sw"
source tools/miniforge3/etc/profile.d/conda.sh
conda activate zephyr
source scripts/set_envvars_sdk.sh
export PATH="/usr/bin:${PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"   # see Bug 13 -- re-promote conda's python/pip
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"   # see Bug 1
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

`scripts/repro_mlp_dronet_yolo_spike.sh` (below) runs this automatically —
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

# yolov8_nano (fp32 — added for the open-issue diagnostic, see below)
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=fp32 TARGET=scalar BACKEND=reference bash modelblaster/examples/yolov8_nano/run.sh
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=fp32 TARGET=rvv    BACKEND=reference bash modelblaster/examples/yolov8_nano/run.sh
```

Each run also verifies bit-exact (int8) or near-fp32-tolerance (fp32)
correctness standalone against the PyTorch golden — confirmed PASS for all
six combinations above (mlp_control fp32, dronet int8, yolov8_nano int8,
yolov8_nano fp32; scalar + RVV each).

Output lands at:
```
zephyr-chipyard-sw/modelblaster/gen/profile/{scalar,RVV}/spike/<model>/<basename>/<run_tag>/topo_0/results.csv
```

**Quirk:** the profile CSV's `<basename>` directory is always tagged
`.fp32` regardless of the actual `QUANT` used (observed behavior, not traced
to its exact source line). This means `yolov8_nano`'s int8 profile and its
true-fp32 profile land at the **same path** —
`gen/profile/{scalar,RVV}/spike/yolov8_nano/yolov8_nano.fp32/...` — so
running the fp32 profiling command after the int8 one **overwrites** the
int8 profile data. Confirmed on disk right now: only the true-fp32 CSVs
exist at that path (mtimes 2026-08-26 14:32/14:35); the int8 profile would
need to be regenerated if needed again.

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

**Bug (basename mismatch):** the profile CSVs are always tagged `.fp32`
(see quirk above) but `emit_dispatch_graph.py` tags its output with the
*actual* quant of the `--ir` file passed in (e.g. `.int8`). The scheduler
matches dispatch graphs to profile CSVs by basename, so an int8-sourced
dispatch graph must be renamed to `.fp32` after generation:

```bash
for hw in scalar RVV; do
  d="../gen/vmfb/<model>/generic_riscv64/$hw"
  mv "$d/<model>.int8" "$d/<model>.fp32"
  mv "$d/<model>.fp32/<model>.int8_dispatch_graph.json" "$d/<model>.fp32/<model>.fp32_dispatch_graph.json"
done
```

`dronet` additionally had a **stale** pre-existing dispatch graph (30
dispatches, from an older model definition) that had to be regenerated from
the current IR (`examples/dronet/int8/generated/graph.json`, 24 dispatches)
using the same procedure — a stale file, not a tagging issue.

Current on-disk state (verified):
| model | hw | dispatches | mtime |
|---|---|---|---|
| mlp_control | scalar/RVV | (fp32 IR) | — |
| dronet | scalar/RVV | 24 | 2026-08-26 14:23 |
| yolov8_nano | scalar/RVV | 212 | 2026-08-26 14:45 (regenerated from the **genuine fp32 IR**, for the open-issue diagnostic — see below; not yet used in a schedule regeneration) |

## 3. Bridge profile data into the top-level repo (`gen_root` workaround)

The workload spec's `hardware.profile.gen_root` field is **ignored** by
`xpu-rt/profile_loader.py`'s `find_profile_csv()`, which hardcodes
`<repo>/gen/profile/...` relative to the top-level FreshScheduler repo. Since
the real data lives under the nested `zephyr-chipyard-sw/modelblaster/gen/`,
bridge it with symlinks (already in place, confirmed present):

```bash
cd "${FRESHSCHEDULER_ROOT}"
mkdir -p gen/profile/RVV/spike gen/profile/scalar/spike
for model in mlp_control dronet yolov8_nano; do
  ln -s "../../../../zephyr-chipyard-sw/modelblaster/gen/profile/RVV/spike/$model"    "gen/profile/RVV/spike/$model"
  ln -s "../../../../zephyr-chipyard-sw/modelblaster/gen/profile/scalar/spike/$model" "gen/profile/scalar/spike/$model"
done
```

Not fixed at the source — redo this if the symlinks are ever lost (e.g.
after `git clean`).

## 4. Workload spec

`data/toplevel/networks_mlp_dronet_yolo_spike.json` (untracked, hand-written
this session). Current content:

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
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/mlp_control/generic_riscv64/RVV/mlp_control.fp32/mlp_control.fp32_dispatch_graph.json",
      "period": 1000, "window_duration": 1000
    },
    "dronet": {
      "id": 1, "identifier": "dronet",
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/dronet/generic_riscv64/RVV/dronet.fp32/dronet.fp32_dispatch_graph.json",
      "period": 1000, "window_duration": 1000
    },
    "yolov8_nano": {
      "id": 2, "identifier": "yolov8_nano",
      "dispatch_deps_path": "zephyr-chipyard-sw/gen/vmfb/yolov8_nano/generic_riscv64/RVV/yolov8_nano.fp32/yolov8_nano.fp32_dispatch_graph.json"
    }
  },
  "edges": []
}
```

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
cd "${FRESHSCHEDULER_ROOT}"
python3 scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_mlp_dronet_yolo_spike.json \
  --solver greedy_periodic --profiled
```

Outputs (both untracked, regenerated in place each run):
- `schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json`
- `plots/networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.png`

First confirmed-good result (both periods at 1000ms): 249 schedule entries,
one instance each of `mlp_control0`, `dronet0`, `yolov8_nano`, makespan
628.94ms, no dependency cycle. Gantt plot shows two rows — `scalar
(CPU_E#0)`: mlp_control0 then dronet0, finishing ~455ms; `RVV (CPU_P#0)`:
yolov8_nano, 0–629ms, dominates the makespan. (Known cosmetic bug, not yet
root-caused: a legend-color mismatch in this plot.)

### 5.1 Validating the 1-instance result + a periodic-instance stress test

The 1-instance-each result above was suspected to be purely a consequence of
period vs. makespan, not a bug — confirmed by reading
`scripts/run_xpurt_schedule.py:431`:

```python
needed = max(1, int(np.ceil(iter_makespan / T)))
```

`T` is the network's `period`; `iter_makespan` is yolov8_nano's non-periodic
makespan (because `restrict_makespan_to_nonperiodic: true`), re-measured each
convergence-loop iteration. With both periods at 1000ms and yolov8_nano
taking ~629ms: `ceil(629/1000) = 1` for each network — exactly what was
observed. Each network's count is computed independently off the same
shared makespan, not off each other's instance counts.

To both validate this and stress-test the Bug 7 tie-break fix at a much
higher periodic-instance density, `mlp_control`'s period was reverted to its
original 10ms while leaving `dronet` at 1000ms (§4 spec updated
accordingly), then the schedule regenerated (§5 command, unchanged). Result:
**58 `mlp_control` instances**, `dronet` still 1, `yolov8_nano` still 1 (642
total dispatches), makespan 574.997ms, `ceil(575/10) = 58` — exact match, and
`dronet`'s count is unaffected as predicted. **No dependency cycle** — the
Bug 7 fix generalizes to this denser schedule. (The makespan itself shifted
slightly from 628.94ms to 574.997ms between the two runs despite
`mlp_control`/`dronet` living on the scalar core and `yolov8_nano` on RVV;
not investigated further — plausible cause is the greedy picker's
tie-breaking shifting for any dispatches with multi-core `machine_combinations`
options as queue contention changes, but this wasn't confirmed.)

This run also surfaced two plotting bugs (Bug 10) that initially blocked the
JSON from being written at all, and a runtime buffer-overflow bug (Bug 11)
that only manifested once a single network had more than one periodic
instance's worth of cumulative dispatches in one combined-binary run — both
now fixed. With both fixed, `plots/networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.png`
renders cleanly at the full `dpi=500` (no fallback needed): 58 `mlp_control`
hairlines spanning the scalar core the whole 575ms, `yolov8_nano` dominating
the RVV core through ~530ms, `dronet0` placed at the end on RVV as well (a
heuristic placement difference from the earlier 1000ms/1000ms run, not a
bug).

## 6. Build + run the combined binary (`xpurt_demo`)

```bash
cd "${FRESHSCHEDULER_ROOT}/zephyr-chipyard-sw"
# (env setup from §0)
SCHEDULE_JSON="${FRESHSCHEDULER_ROOT}/schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json" \
MODELS=mlp_control,dronet,yolov8_nano \
QUANTS=fp32,int8,int8 \
BACKENDS=scalar,rvv \
FORCE_REGEN=1 \
RUNNER=spike \
XPURT_TRACE=1 \
  bash modelblaster/examples/xpurt_demo/run.sh
```

**`FORCE_REGEN=1` is required** (it's actually the script's own default —
an earlier pass through this doc used `FORCE_REGEN=0` for speed, which is
fine once every model's `generated/` dir already reflects the latest
generator code, but is wrong any time `generate_skeleton.py`/
`generate_kernels.py` themselves changed, e.g. after Bug 11 below — with
`FORCE_REGEN=0` the stale, already-generated `model.c` files are reused as-is
and the fix never takes effect).

**Result against the 58-instance schedule, with Bugs 7/9/11 applied:**
`mlp_control` **PASS** (`max_abs_err=6.68e-06`, runs correctly across all 58
periodic instances). `dronet` **FAIL** (`max_abs_err=51`, atol=0) — a
**regression** from the earlier 1-instance-per-network schedule, where it
passed bit-exact. `yolov8_nano` **FAIL** (`max_abs_err=30`) — the
pre-existing open issue, unchanged. See "Open issue" below — `dronet`
starting to fail once `mlp_control`'s periodic density increased 58x is
itself evidence for the buffer-aliasing theory. Log saved at
`/tmp/claude-1172/.../scratchpad/xpurt_demo_mlp58_v3.log`.

**Wall-clock time for the full rebuild+rerun cycle:** ~3.5 minutes end to
end (measured via file mtimes bounding the background command: generator
edit at 16:07:31 to log completion at 16:11:05, 2026-08-26). Breakdown:
~2 min re-generating all 3 models standalone (`FORCE_REGEN=1` re-runs each
model's own `run.sh` first), then the `xpurt_demo` pipeline itself is fast
(`generate_kernels` 0.067s + `build` 8.647s per its own stage timers), with
the remainder (~1.5 min) being the actual spike simulation + per-network
verification passes for the combined binary. This is orders of magnitude
faster than an equivalent FireSim run would be, consistent with why this
whole reproduction stays on spike.

**Use `QUANTS` (plural), not `QUANT`** for mixed-quant builds like this one —
a blanket `QUANT=int8` with `FORCE_REGEN=0` will silently regenerate any
(model, quant) directory that doesn't already exist, including models that
should stay at a different quant (mlp_control has no int8 variant and must
stay fp32). This bit me once this session; cleaned up the stray
`examples/mlp_control/int8/` directory it created.

## 7. Instrumented spike run → real execution timeline

`XPURT_TRACE=1` (already set in §6) makes the schedule-driven runtime emit a
per-dispatch CSV block (`=== MODELBLASTER_XPURT_TRACE_BEGIN ===` /
`_END`) into the run log, one row per dispatch with both the scheduler's
*predicted* timing and the *actual* measured spike cycles:
`entry_id,network,instance,dispatch_id,op,name,core_kind,hart,predicted_start_ms,predicted_duration_ms,worker_kind_idx,actual_start_cycles,actual_end_cycles`.
This is independent of the per-network correctness verification — the trace
captures real timing regardless of PASS/FAIL, though whether the *values*
computed along the way are trustworthy for a FAILing network is a separate
question this trace doesn't answer.

A dedicated tool already exists to turn this into a timeline plot —
`zephyr-chipyard-sw/modelblaster/scripts/plot_xpurt_trace.py` (pre-existing,
not written this session) — renders two stacked Gantt charts (predicted
schedule vs. actual execution) and flags any entry that ran later than
predicted:

```bash
cd "${FRESHSCHEDULER_ROOT}/zephyr-chipyard-sw/modelblaster"
source ../tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
export PYTHONPATH="$PWD/..${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m modelblaster.scripts.plot_xpurt_trace \
  <path-to-the-run-log-you-captured-in-§6> \
  --clock-mhz 10 \
  --out "${FRESHSCHEDULER_ROOT}/plots/xpurt_trace_mlp_dronet_yolo_spike_clk10.png" \
  --csv "${FRESHSCHEDULER_ROOT}/schedules/xpurt_trace_mlp_dronet_yolo_spike_clk10.csv"
```

**IMPORTANT — use `--clock-mhz 10`, not `1000`.** `actual_start/end_cycles`
in the trace come from `k_cycle_get_64()`, which is `mtime`-based (RISC-V
CLINT), not `rdcycle()`-based — these are two genuinely different clocks
(see §8). An earlier pass through this doc used `--clock-mhz 1000` (matching
the *predicted*-side profiling assumption) to interpret the *actual* side
too, which is a unit error, not a real finding — it made real execution look
~5x faster than predicted and periodic instances look bunched at the start
of the run. Re-run with `--clock-mhz 10` (matching `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=10000000`,
confirmed in §8) and the picture flips: actual makespan came out to
**11,006.9ms vs. 575ms predicted (19x slower)**, with periodic instances
correctly spread across roughly the right window. That 19x gap doesn't come
from a clock mismatch either (§8 verifies the two clocks agree) — it was the
weight-symbol-collision bug, §9/Bug 12 below.

## 8. Clock/timing sanity check: rdcycle vs. mtime

Two genuinely different clocks are in play in this pipeline, and confusing
them (as an earlier pass through this doc did) produces misleading
conclusions. Both checks below are reproducible from data already collected
in §6/§7's logs — no new runs needed.

**8.1 — Confirmed NOT using `--real-time-clint`.** Spike has an opt-in flag
(`spike --help` → `--real-time-clint  Increment clint time at real-time
rate`) that ties `mtime` to host wall-clock time — which would make timing
non-deterministic and host-speed-dependent. Grepped the whole modelblaster
pipeline: this flag is never passed anywhere. Spike's default, deterministic
CLINT is in use, as it should be.

**8.2 — `k_cycle_get_64()` is `mtime`, not `rdcycle()`.**
`zephyr_ws/zephyr/drivers/timer/riscv_machine_timer.c`: `sys_clock_cycle_get_64()`
returns `mtime() << CONFIG_RISCV_MACHINE_TIMER_SYSTEM_CLOCK_DIVIDER`. Our
build's `.config` has `CONFIG_RISCV_MACHINE_TIMER_SYSTEM_CLOCK_DIVIDER=0` —
no scaling — so this is raw, real `mtime`, a genuine RISC-V real-time
reference clock, hart-independent (unaffected by any hart being idle/WFI —
this is *why* the periodic-start gate correctly uses it instead of
per-hart `rdcycle()`/`mcycle`, which does NOT advance during `wfi` and would
desync across harts). `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=10000000` (10MHz)
describes its real tick rate.

Per-op profiling (`records_[]` in `generate_skeleton.py`, feeding
`results.csv` → the scheduler's `predicted_*_ms`) uses `rdcycle()` instead —
a separate, real CPU instruction/cycle counter. `profile_writer.py:36-39`
documents that for spike these are **retired-instruction counts on a flat
memory model**, converted to "ms" via `PROFILE_CLOCK_MHZ` (default 1000.0),
an explicitly-arbitrary placeholder, not a calibrated physical value.

**8.3 — Empirically verified the two clocks ARE consistently related.** For
a clean, uncontended `mlp_control` instance (any of instances 2–53 in the
58-instance run — all report an identical `WALL_CYCLES_INST` of exactly
3,950 mtime ticks), summing that same instance's `rdcycle()`-based per-op
records (`MODELBLASTER_PROFILE_BEGIN [mlp_control]` block) gives 393,312.
Ratio: 393,312 / 3,950 ≈ **99.6**, matching the **100** implied by
`PROFILE_CLOCK_MHZ=1000` ÷ `CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=10MHz`.
Reconfirmed even more cleanly on the minimal `mlp_control`-only build (§9.1):
standalone rdcycle-sum 393,465 vs. xpurt mtime×100 405,000 → within 2.5%.
**Conclusion: there is no clock-calibration bug.** The two clocks agree; any
predicted-vs-actual gap has a different cause.

## 9. Isolating the real root cause: minimal single-network builds

Given the 3-network combined binary showed `dronet`/`yolov8_nano` failing
correctness (`max_abs_err=51`/`30`) *and* retiring far more instructions
than predicted (~13–16x, computed via the verified 100:1 ratio in §8.3),
while `mlp_control` was clean on both counts, two hypotheses were live: (a)
cross-network buffer aliasing (the original theory, before this section),
or (b) something specific to how `dronet`/`yolov8_nano`'s own kernels behave
under the xpurt mechanism regardless of what else is scheduled. Minimal
single-network builds settle this.

### 9.1 `mlp_control`-only — rules out the xpurt mechanism itself

Workload spec: `data/toplevel/networks_mlp_only_spike.json` (single network,
no `period`/`window_duration` — non-periodic, single pass). Schedule
generated the same way as §5 (7 dispatches, 0.39ms predicted makespan, no
periodic replication). Built via §6's command with `MODELS=mlp_control
QUANTS=fp32`.

| | rdcycle total (compute) | mtime wall_cycles | ratio |
|---|---|---|---|
| standalone `run.sh` | 393,465 | 3,950 | 99.6 |
| xpurt-generated (single-network) | 393,653 | 4,050 | 97.2 |

Essentially identical (rdcycle +0.05% noise, mtime +2.5% — plausibly the
`fence rw,rw` + semaphore overhead around each dispatch). **Rules out the
xpurt dispatch mechanism (fences, semaphores, worker threads) as a source of
any large regression** — applied to `mlp_control` alone, the overhead is
negligible.

### 9.2 `dronet`-only — rules out cross-network buffer aliasing

Workload spec: `data/toplevel/networks_dronet_only_spike.json` (same
pattern, single network, non-periodic). Built via §6's command with
`MODELS=dronet QUANTS=int8`. Result, **before** the fix below: `max_abs_err=51`
(`actual=[-5.0, 127.0]` vs `golden=[-56.0, 127.0]`) — **identical** failure
to the 3-network case, reproduced with `dronet` completely alone. This
**rules out cross-network buffer aliasing** — there is no other network
present to alias with.

Total rdcycle also confirmed the same ~16x blowup in isolation
(452,989,905 actual vs. 28,413,832 predicted from a matching standalone
run — see Bug 12 below for how this was root-caused precisely).

### Bug 12 (the real root cause) — weight-symbol collision across backends

See the full write-up under "Bugs found and fixed this session" below.
Summary of the isolation process: standalone `dronet` at `TARGET=rvv` with
curated kernels enabled passed bit-exact (28,413,832 rdcycle, matching
predicted); standalone at `TARGET=scalar` ALSO passed bit-exact — so neither
individual backend's kernels were at fault. Diffing the actual generated
`weights.c` between `generated/scalar/` and `generated/rvv/` showed the
`conv2d_s8` weight tensors are packed in genuinely different physical
layouts per backend (`OIHW` for scalar vs. `IHWOC` for RVV,
`backends.py:95-97`'s `MODELBLASTER_RVV_IHWOC_WEIGHTS=1`), but under the
**identical, non-static C symbol name** in both files. The xpurt combined
build compiled only one backend's `weights.c` (whichever is first in
`BACKENDS=`, i.e. `scalar`) as the single "shared" weights object
(`harness_xpurt/CMakeLists.txt`'s old `_weights_tgt`), so any dispatch that
actually ran on RVV read scalar-packed (wrong-layout) weight data — a
deterministic wrong-answer bug independent of which kernel algorithm
executed, and independent of any other network being present.

### 9.3 Post-fix verification

Rebuilt `dronet`-only against the Bug 12 fix (`FORCE_REGEN=1`, same
command): **PASS**, `max_abs_err=0`, `actual=[-56.0, 127.0]` ==
`golden=[-56.0, 127.0]`. Total rdcycle 28,415,376 vs. standalone's
28,413,832 (0.005% off); mtime wall_cycles 284,400 vs. standalone's 284,200
(0.07% off) — both well under the 5% target. `yolov8_nano`-only and the
full 3-network rebuild against this same fix are the next steps (§10).

## Bugs found and fixed this session

1. **PYTHONPATH not exported by `_run_lib.sh`** (workaround only, not fixed
   at the source). `examples/<model>/run.sh` computes
   `REPO_ROOT=.../modelblaster` (two dirs up from itself), but the shared
   `_run_lib.sh` submodule-adaptation block checks for
   `${REPO_ROOT}/modelblaster/pipeline` — a doubly-nested path that never
   exists — so the condition that would set `PYTHONPATH` is always false.
   Always `export PYTHONPATH="$PWD"` from `zephyr-chipyard-sw/` first.
2. `emit_dispatch_graph.py` isn't documented in the top-level flow; must be
   invoked directly (§2).
3. `gen_root` in the workload spec is ignored by `profile_loader.py`;
   bridged with symlinks (§3), not fixed at the source.
4. Dispatch-graph/profile-CSV basename mismatch (`.int8` vs. always-`.fp32`)
   requires a manual rename after generation (§2). Also caught a genuinely
   **stale** dronet dispatch graph (30 vs. 24 dispatches) needing outright
   regeneration.
5. `prune_periodic: true`'s post-loop trim
   (`xpu-rt/postprocessing.py::trim_periodic_after_nonperiodic_makespan()`)
   determines which periodic instances to drop using an attribute-based
   check (`min_start_t`/`max_end_t`) that disagrees with the convergence
   loop's own string-based periodic-instance detection, producing an
   internally-inconsistent schedule for this workload shape. Not fixed at
   the source — worked around via `"prune_periodic": false` in the spec.
6. `QUANT` vs `QUANTS` footgun in `xpurt_demo/run.sh` — see §6 above. Existing
   behavior, not modified, just a mistake to avoid.
7. **Fixed in code** — `xpu-rt/postprocessing.py`, the `time_dependency`
   assignment loop (tracked file, top-level repo, **uncommitted**). Original
   tie-break used `completion_time <= start_time` when picking a dispatch's
   same-core predecessor; two independent dispatches genuinely tying on
   completion/start time (e.g. parallel branches after a fork, sharing one
   core) could have the tie broken by incidental list-insertion order rather
   than true precedence, letting a dispatch's own successor be picked as its
   "predecessor" — producing a real 2-cycle (`dispatch_28` ↔ `dispatch_29`)
   that `ingest_xpurt_schedule.py`'s topological sort correctly rejected.
   Fix: `<=`/`>` → `<`/`>=`, so exact ties are dropped rather than guessed at
   (real data dependencies already order anything that must be ordered).
8. Covered under Bug 4 (stale dronet dispatch graph).
10. **Fixed in code** — `xpu-rt/plot.py` (tracked file, top-level repo,
   **uncommitted**), `plot_optimization_schedule()`'s final `plt.savefig()`
   call. The Gantt legend has one entry per job *instance*
   (`legend_labels`/`legend_handles` built per `job_name`, not deduplicated
   by base network name — see `xpu-rt/plot.py` around line 390), so 58
   `mlp_control` instances produced ~60 legend rows; at the hardcoded
   `dpi=500` with `bbox_inches='tight'`, this triggered a matplotlib/freetype
   `RuntimeError: ... raster overflow`. Because `run_xpurt_schedule.py` calls
   `output_scheduled_json()` *after* the plot save (§5, no ordering
   swap made), this exception was blocking the schedule JSON from being
   written at all, not just the picture. Fix: wrapped the `savefig` call in
   `try/except` (+ `plt.close()` in a `finally`) so a render failure prints a
   warning and lets JSON output proceed — the plot is a visualization aid,
   not the deliverable. **Not fixed:** the legend-blowup itself (one entry
   per instance rather than per base network) — still produces an unreadable
   or unrenderable legend at high periodic-instance counts; would need
   dedup-by-base-name in the legend-building loop to actually fix the plot
   output, not just make failure non-fatal. (A second companion fix, Bug
   10b below, was also needed before this plot actually rendered at full
   quality — the legend fix alone wasn't sufficient.)
10b. **Fixed in code** — `scripts/run_xpurt_schedule.py` (tracked, top-level
   repo, **uncommitted**), the plot title construction (~line 523). Same root
   cause as Bug 10: `title_networks` was built from every unique `job_id`'s
   name, not deduplicated by base model kind, so a title listing all 58
   `mlp_control0..57` names forced `bbox_inches='tight'` to blow the canvas
   out to 26113×1768px (almost entirely title whitespace) — this is what was
   actually still triggering the dpi=500 raster-overflow retry even after the
   Bug 10 legend fix landed. Fixed by deduping title network names through
   `plot._kind_from_job_name()` (same helper the legend fix uses). With both
   10 and 10b applied, the plot renders correctly at dpi=500 with no
   fallback needed.
11. **Fixed in code** — `zephyr-chipyard-sw/modelblaster/pipeline/generate_skeleton.py`
   (tracked, nested `modelblaster` submodule, **uncommitted**) — a genuine
   runtime buffer overflow, unrelated to Q31/gemmini. Each model's per-op
   profiling record buffer (`records_[MODEL_..._OP_COUNT]`, line ~1918) is
   sized for exactly one forward pass (7 slots for `mlp_control`'s 7 ops).
   The write side (`int slot = n_++`, line ~1868) had no bounds check, and
   `n_` only resets via `run_model_{mid}()`'s straight-line path or an
   explicit `reset_profile()` call — but the xpurt schedule-driven runtime
   invokes dispatches directly through the per-dispatch table, using neither
   reset point, so `n_` climbs across every periodic instance in a run with
   nothing to stop it. Every earlier schedule had ≤1 instance per network, so
   cumulative dispatches never exceeded `OP_COUNT` and this never surfaced.
   With 58 `mlp_control` instances, the buffer filled after instance 0 and
   instance 1's dispatches wrote straight past the array end, corrupting
   memory — manifested as a spike `mcause: 5, Load access fault` a few
   dispatches into the post-run per-model profile dump. Fixed by wrapping
   the write (`slot = (n_++) % MODEL_..._OP_COUNT`) and capping the read
   side's reported count to the same bound — safe regardless of instance
   count, and now reports the most recent instance's trace instead of
   corrupting memory. Verified: rebuilding with `FORCE_REGEN=1` and rerunning
   no longer crashes; `mlp_control` passes cleanly across all 58 instances.
9. **Fixed in code** — `zephyr-chipyard-sw/modelblaster/examples/xpurt_demo/run.sh`
   (tracked file, nested `modelblaster` submodule, **uncommitted**). The
   `spike_runner` invocation branch only ever passed a single blanket
   `--quant`/`--io` (first model's golden), unlike the `firesim_runner`
   branch which already builds a per-model `--quants` flag. For a
   mixed-quant build (`QUANTS=fp32,int8,int8`), any model whose actual quant
   differs from the first model's in `MODELS` failed verification with
   "golden not found" even though it ran correctly. Fix: build a
   `--io-paths net=path,...` argument from the existing `MODEL_LIST`/
   `QUANT_LIST` arrays (mirroring `spike_runner.py`'s already-supported
   `--io-paths` flag), and fix the `--io` fallback to use
   `${QUANT_LIST[0]}` instead of the blanket `${QUANT}`.
12. **Fixed in code** — two files, both required together:
    `zephyr-chipyard-sw/modelblaster/pipeline/generate_skeleton.py` (tracked,
    nested `modelblaster` submodule, **uncommitted**) and
    `zephyr-chipyard-sw/modelblaster/harness_xpurt/CMakeLists.txt` (tracked,
    nested submodule, **uncommitted**). This was the real cause of both the
    "cross-network numeric corruption" symptom AND the 13-19x
    predicted-vs-actual timing gap — see §9 for the full isolation process.
    Root cause: `generate_skeleton.py`'s `_backend_pack_weight()` genuinely
    repacks conv weight tensors into a different physical layout per backend
    (`OIHW` for scalar, `IHWOC` for RVV — `backends.py:95-97`'s
    `MODELBLASTER_RVV_IHWOC_WEIGHTS=1`), but `_weight_name()` (which names
    the C symbol holding that data, used both for `weights.c`'s definition
    and every dispatch function's reference to it) never accounted for
    backend, so `generated/scalar/weights.c` and `generated/rvv/weights.c`
    both define the **identical, non-static** symbol name
    (e.g. `dronet_conv_modules_0_weight_q`) with **different data**.
    `harness_xpurt/CMakeLists.txt` compiled only one backend's `weights.c`
    per model into a single "shared" object (whichever is first in
    `BACKENDS=`, i.e. `scalar`) on the explicit (and wrong) assumption that
    "all copies are identical" — so any dispatch that actually executed on
    RVV silently read scalar-packed weight data through RVV's IHWOC
    indexing, producing a deterministic wrong answer (not a crash, since
    only one symbol definition ever got linked) that had nothing to do with
    which kernel algorithm ran or which other networks were scheduled
    alongside it. Fix: `_weight_name()` now takes an optional `backend`
    argument and appends it as a suffix whenever provided (e.g.
    `dronet_conv_modules_0_weight_q_rvv` vs. `..._scalar`) — applied
    unconditionally (not just to layout-sensitive conv weights) for
    simplicity and to guarantee no collision; `emit_model()` was given a
    `backend` parameter and its ~46 call sites into `_weight_name()` updated
    to pass it through (mechanical, via a scripted regex substitution);
    `generate()`'s existing call to `emit_model()` now forwards its own
    `backend` argument. `harness_xpurt/CMakeLists.txt` was updated to build
    one `_weights_tgt` OBJECT library **per (model, backend)** instead of
    one per model, and to link all of them into the final binary instead of
    just the "primary" backend's. Verified: see §9.3 (dronet-only rebuild:
    `max_abs_err=51`→`0`, timing within 0.1% of standalone) and §10 (full
    3-network rebuild).
13. **Fixed in code** — this doc's own §0, `export PATH="/usr/bin:${PATH}"`.
    Found via a genuinely sandboxed (Docker, `ubuntu:24.04`, no host mounts
    except `.ssh`) from-scratch validation run. That line exists to fix a
    `cmake`-shadowing issue (an old Vitis-bundled `cmake` earlier in PATH);
    it works, but it also demotes the conda `zephyr` env's own `python3`/
    `pip` behind `/usr/bin`'s system ones. On any host enforcing PEP 668
    (stock Ubuntu 24.04, but apparently not the original host this doc was
    written on — hence this going unnoticed), every subsequent `pip install`
    in this doc then fails with `error: externally-managed-environment`,
    blocking `pip install spike`/`torch`/`-r requirements-base.txt`/
    `-e modelblaster/`. Fix: immediately re-prepend `${CONDA_PREFIX}/bin`
    after the `/usr/bin` line, so the real `cmake` from `/usr/bin` is still
    found (ahead of whatever shadowed it before), but the conda env's own
    `python3`/`pip` win over `/usr/bin`'s.
14. **Fixed in code** — `zephyr-chipyard-sw/modelblaster/models/mlp_control.py`
    (tracked, nested `modelblaster` submodule). Also found by the same
    sandboxed validation run: `_DEFAULT_CKPT` hardcoded an absolute path,
    `/scratch2/dima/misc_sw/FreshScheduler/logs/rsl_rl/crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt`
    — a 1.1MB trained-policy checkpoint that existed only on one user's
    disk and was **not git-tracked anywhere** (confirmed via `git ls-files`
    and `git check-ignore` — genuinely untracked local state, not merely
    misconfigured `.gitignore`). Unlike every other hardcoded-path bug this
    session, no `FRESHSCHEDULER_ROOT`-style env var fixes this for a fresh
    clone: the file itself doesn't exist anywhere reproducible, so the very
    first `mlp_control` profiling command fails with `FileNotFoundError`
    before any of this doc's own workarounds are even reached. Fix:
    committed the checkpoint into modelblaster itself
    (`models/checkpoints/mlp_control/model_6998.pt`, small enough to be a
    normal git blob) and changed `_DEFAULT_CKPT` to derive from `__file__`
    instead of a hardcoded absolute string — self-contained regardless of
    where or how deep this repo is checked out.
    `MODELBLASTER_MLP_CONTROL_CKPT` still overrides it if needed.
15. **Fixed in code** — `zephyr-chipyard-sw/modelblaster/models/dronet.py`
    (tracked, nested `modelblaster` submodule). Found by a second sandboxed
    Docker validation run made *after* Bugs 13/14 landed on `dev` — both of
    those were confirmed fixed (pip installs succeeded, `mlp_control`'s
    checkpoint resolved correctly), but `dronet`'s own profiling step then
    hit the exact same class of bug, one level worse: `_DRONET_SRC` used
    `importlib.util.spec_from_file_location` to dynamically load the
    `DronetTorch` architecture class from an absolute path,
    `/scratch2/dima/misc_sw/FreshScheduler/qnn_models/dronet.py` — a file in
    the *top-level* repo, entirely outside `modelblaster`, never committed
    to it. Since this loads a `.py` file by literal path rather than
    importing a module, no `PYTHONPATH`/`FRESHSCHEDULER_ROOT` fix could
    reach it. `_DEFAULT_CKPT` had the identical untracked-absolute-path
    problem as Bug 14 (`logs/dronet/2026-04-27_17-10-41/best.pt`, 1.27MB,
    confirmed untracked via `git ls-files`/`git check-ignore`), just not
    reached yet since the architecture load fails first. Fix: vendored the
    architecture class into `modelblaster` itself as `models/dronet_arch.py`
    (a copy of `qnn_models/dronet.py`, which is itself already a documented
    copy of `merlin/models/dronet/dronet.py` — vendoring small,
    self-contained model-definition files like this is an established
    pattern in this codebase, not a new one), replaced the dynamic
    `importlib`-based load with a normal `from . import dronet_arch`, and
    committed the checkpoint to `models/checkpoints/dronet/best.pt` with
    `_DEFAULT_CKPT` now `__file__`-relative (identical fix shape to Bug 14).
    `MODELBLASTER_DRONET_CKPT` still overrides the checkpoint if needed.
    Verified: `get_model()` loads and runs a forward pass correctly from a
    fresh checkout with no `FileNotFoundError`.
16. **Fixed in code (partially) + doc fix** — a third sandboxed Docker
    validation run (after Bugs 13/14/15 all landed on `dev` and were
    confirmed non-recurring — `mlp_control`/`dronet` both profiled
    successfully) hit a *different class* of gap on `yolov8_nano`, the one
    model never previously reached: `pip install ultralytics` succeeds, but
    `ultralytics`'s own `opencv-python` dependency fails to import at
    runtime — `ImportError: libGL.so.1: cannot open shared object file` — a
    standard headless-Linux gap (`libgl1` isn't installed on a bare
    `ubuntu:24.04`; presumably already present on a real desktop/dev
    workstation, which is why this went unnoticed until a truly minimal
    container tested it). Two things needed fixing: (a) **doc** — `libgl1`
    noted as a required system package (§0 above); and (b) **code** —
    `zephyr-chipyard-sw/modelblaster/models/yolov8_nano.py`'s
    `_load_ultralytics_weights()` caught `ImportError` broadly and reported
    a fixed `"ultralytics not installed"` message regardless of the actual
    cause, which is misleading here (ultralytics *is* installed; one of
    *its* dependencies failed to import) and sent the previous debugging
    pass looking in the wrong place. Fixed to include the real underlying
    exception's message in the raised `RuntimeError` rather than a generic
    string. Not independently re-verified end-to-end after the `libgl1` fix
    — flagged here rather than silently assumed fixed.
17. **Fixed in code** — `scripts/repro_mlp_dronet_yolo_spike.sh`. A fourth
    sandboxed run (after Bugs 13-16 all landed and were confirmed
    non-recurring — all 3 models profiled successfully inside the actual
    pipeline run) found that the script had the **exact same PATH-ordering
    defect as the original Bug 13**, never actually fixed in the script
    itself — only the doc's copy-pasteable §0 block was fixed. This didn't
    break step 1 (model profiling) because `modelblaster/examples/
    _run_lib.sh` invokes the interpreter as bare `python` (no apt-installed
    `/usr/bin/python` exists on a bare `ubuntu:24.04` to shadow it), but the
    script's own schedule-generation step calls `python3
    scripts/run_xpurt_schedule.py` directly — and `/usr/bin/python3` *does*
    exist there (the apt `python3` package), so it silently resolved to the
    system interpreter instead of the conda env's, failing with
    `ModuleNotFoundError: No module named 'numpy'`. Same fix as Bug 13:
    re-prepend `${CONDA_PREFIX}/bin` after the `/usr/bin:${PATH}` line.
    This is also the occasion for the broader dependency-management
    cleanup below (§ dependency management) — the script now also calls
    `scripts/install_xpurt_deps.sh` before running anything, rather than
    assuming the active env already has everything it needs.

## Dependency management: zephyr-chipyard-sw vs. xpu-rt

Cleaned up after the bugs above kept surfacing the same underlying
confusion — which env needs what, and where is it declared:

- **`zephyr-chipyard-sw` stays fully standalone.** Its own
  `install_conda.sh`/`install_submodules.sh`/`install_toolchain_sdk.sh`
  are untouched and still produce a complete, usable `zephyr` env for
  Zephyr/RISC-V development on their own — no xpurt-specific package
  (torch, ultralytics, cvxpy, spike) is installed by them. A consumer that
  only wants the Zephyr/RISC-V dev environment (e.g. the separate RoSE
  project, which embeds this same zephyr-chipyard-sw as its own submodule
  on an unrelated branch) never needs to know xpurt exists.
- **Everything xpurt-specific is declared in *this* repo, in one place:**
  `pyproject.toml` (xpu-rt's own scheduler deps — numpy/scipy/matplotlib/
  pandas, plus an optional `milp` extra for cvxpy) and
  `scripts/install_xpurt_deps.sh` (installs xpu-rt's own deps, then
  modelblaster's own deps via its own `pyproject.toml`, then the pinned
  `spike` wheel, then the `libgl1` system package). One conda env — the
  same `zephyr` env produced by zephyr-chipyard-sw's standalone install —
  covers both `west build` and the scheduling/reproduction flow; nothing
  here creates a second env.
- **Removed as part of this cleanup** (superseded, not referenced by
  anything, confirmed via `git grep`): the top-level `setup.py` (replaced
  by `pyproject.toml`), the top-level `env.yml` (an orphaned, fully-pinned
  MOSEK+cvxpy conda env that no install script ever actually used — the
  README's own documented setup command uses a *different* file,
  `merlin/env_linux.yml`, entirely unrelated to this flow), and
  `zephyr-chipyard-sw/modelblaster/requirements.txt` (a stale, incomplete
  subset of modelblaster's own `pyproject.toml`, missing `pyyaml`/
  `pillow`/`ultralytics`).

## Resolved: cross-network numeric corruption in the combined binary

**Originally filed as an open issue; root-caused and fixed as Bug 12
above.** Standalone (§1), all three models pass bit-exact (int8) or within
tolerance (fp32), on both backends. In the pre-fix **combined**
multi-network binary built via `xpurt_demo`, `mlp_control` passed but
`dronet`/`yolov8_nano` both failed (`max_abs_err=51`/`30`).

The original working theory here was cross-network buffer
aliasing/corruption (`buffers.c`'s `extern`-shared storage across
networks), motivated by `dronet` having passed bit-exact in an earlier,
sparser (1-instance-per-network) schedule and only starting to fail once
`mlp_control` was reverted to 58 periodic instances (§5.1) — suggestive of
corruption that needed dense interleaving to manifest. **This theory was
wrong.** §9.2's minimal `dronet`-only build (no other network present at
all) reproduced the identical failure, conclusively ruling out cross-network
aliasing. The real cause (Bug 12) was the weight-symbol collision described
above — its manifestation just happens to correlate with schedule density
because denser schedules are more likely to actually place a dispatch on
the RVV core kind (reading the wrong-layout weights) versus a sparser
schedule that might route everything through whichever backend happened to
be "primary." The apparent 1-instance-schedule "pass" for `dronet` was
very likely a case where the sparser schedule didn't happen to schedule
dronet's ops in a way that exposed the mismatch, not evidence the bug
wasn't present.

## 10. Full 3-network rebuild — final result

### 10.1 Reprofiling `yolov8_nano`

Before rebuilding the full schedule, checked whether `yolov8_nano`'s
existing profile data (captured early this session, mtime 14:35, well
before Bugs 11/12) was stale relative to the now-fixed code. Direct
same-op comparison: the original `results.csv` recorded 4,552,998 cycles
for `l0.conv`; a fresh standalone run of the *current* code measures
6,815,996 cycles for the identical op/shape — a 49.7% increase, confirming
staleness (some kernel/weight change earlier in this session's fixes
genuinely changed `yolov8_nano`'s real compute cost). `dronet` and
`mlp_control`'s existing profile data were separately confirmed NOT stale
(§9.3's dronet numbers already matched fresh standalone to within 0.005%;
mlp_control's numbers were re-verified in §10.3 below and came back
byte-identical to before, ruling out staleness for it too).

Reprofiled `yolov8_nano` only (both backends), reusing the exact §1
commands:

```bash
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=scalar BACKEND=reference RUNNER=spike bash modelblaster/examples/yolov8_nano/run.sh
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 PROFILE_CLOCK_MHZ=1000.0 \
GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
QUANT=int8 TARGET=rvv BACKEND=reference RUNNER=spike bash modelblaster/examples/yolov8_nano/run.sh
```

### 10.2 Regenerate the schedule + rebuild

Same commands as §5/§6, run against the refreshed profile data (schedule
regeneration picks up the new `results.csv` automatically — no code
change needed):

```bash
python3 scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_mlp_dronet_yolo_spike.json \
  --solver greedy_periodic --profiled
```

Result: **71** `mlp_control` instances (up from 58 — `yolov8_nano`'s
now-accurate, larger makespan needs more periodic coverage), 733 total
dispatches, **704.997ms** predicted makespan (up from 574.997ms). Then
rebuilt via §6's exact command (`FORCE_REGEN=1`, `QUANTS=fp32,int8,int8`,
`BACKENDS=scalar,rvv`, `XPURT_TRACE=1`).

**Result: `OVERALL: PASS (3 models)`** — `mlp_control`
(`max_abs_err=6.52e-08`), `dronet` (`max_abs_err=0`, bit-exact), `yolov8_nano`
(`max_abs_err=0`, bit-exact, sampled). No crashes.

### 10.3 Timing variation — target achieved for the dominant networks

Ran `plot_xpurt_trace.py --clock-mhz 10` (§7/§8) against the final run's
log and cross-checked with a direct per-network aggregate calculation from
the resulting CSV (`schedules/xpurt_trace_mlp_dronet_yolo_spike_final.csv`):

```
predicted makespan: 704.997 ms
actual    makespan: 704.675 ms   (1.00x predicted)
entries finishing later than predicted: 375/733

Per-network (sum of predicted_duration_ms vs. sum of actual per-dispatch duration):
  yolov8_nano: predicted=628.939ms actual=628.975ms   deviation = +0.01%
  dronet:      predicted= 28.414ms actual= 28.390ms   deviation = -0.08%
  mlp_control: predicted= 27.928ms actual= 23.275ms   deviation = -16.66%
  OVERALL:     predicted=685.281ms actual=680.640ms   deviation = -0.68%
```

The two substantial, hardware-relevant networks (`dronet`, `yolov8_nano` —
together >95% of the schedule's total time) are within **0.1%** of
predicted, and the schedule-level makespan match is within **0.05%**. Both
comfortably clear the 5% target.

**`mlp_control`'s own operators do not** — a genuine, reproducible ~15-17%
deviation (always faster than predicted, never slower), confirmed via two
independent measurement paths (the `XPURT_TRACE` per-dispatch actual
timing, and the `MODELBLASTER_PROFILE_BEGIN` rdcycle profile dump for the
last-captured periodic instance, e.g. `mlp.2`'s `linear` op: 17,087 actual
rdcycle-equivalent vs. 264,229 predicted). Investigated and ruled out as
causes:
- **Stale profiling** — reprofiled `mlp_control` on both backends;
  cycle counts came back byte-identical to the existing data.
- **Cross-backend weight mismatch (Bug 12)** — not applicable; `mlp_control`
  has no curated kernels or backend-specific weight packing at all.
- **Intra-op parallelism / `modelblaster_pool` interference** — ruled out;
  the build's own log confirms `pool_sizes: [0, 0]` (both kinds), i.e. NULL
  pools, no helper threads, for this hardware spec (`machines: {cpu_p:1,
  cpu_e:1}`).
- **An obvious code-level RVV vector-length state bug** — disassembled the
  compiled `kernel_linear_mlp_control_rvv` (`llvm-objdump` on
  `.../mlp_control/fp32/generated/rvv/kernels.c.obj`): the loop's `vsetvli`
  is called fresh every iteration against locally-initialized,
  freshly-zeroed loop counters (`li t2,0 / li t0,0 / li t6,0` at function
  entry) — nothing here should behave differently based on whatever ran
  on the hart immediately before it.

**Not fully root-caused.** `mlp_control`'s absolute contribution is small
(27.9ms of 704.997ms predicted, ~4% of the schedule), so this residual
doesn't materially affect overall schedule accuracy, but the mechanism
itself remains unexplained at the instruction level — plausible remaining
candidate (not verified, spike's own source isn't vendored in this repo to
check directly): some interaction between spike's default CLINT/cycle
accounting and genuinely concurrent multi-hart execution, which wasn't
exercised the same way in either the standalone harness (§9.1's isolated
`mlp_control`-only test, which matched predicted almost exactly) or in
`dronet`/`yolov8_nano`'s own isolated single-network tests (§9.2, §10.1).
Flagged here as a known, scoped, low-impact open item rather than
papered over.

## 11. Commits, pushes, and from-scratch verification

All code fixes above are committed and pushed (working-tree-only,
uncommitted state described in earlier sections is now stale — check `git
log` in each repo for the actual commits):

- `modelblaster` (`git@github.com:ucb-bar/ModelBlaster.git`,
  branch `feat/harness-shared-input`): `6369fd6` (Bugs 9/11/12) and
  `650558a` (a new bug found only by this from-scratch verification, see
  below).
- `zephyr-chipyard-sw` (`git@github.com:ucb-bar/zephyr-chipyard-sw.git`,
  branch `et-sizing-knobs`): two pointer-bump commits following the two
  modelblaster commits above.
- Top-level (pushed to the `ucb-bar` remote, `git@github.com:ucb-bar/XPU-RT.git`,
  branch `feature/fp-precision-stripping` — **note:** this repo has three
  configured remotes pointing at three different repo names (`origin`→
  `Scheduler.git`, `new`→`XPURT.git`, `ucb-bar`→`XPU-RT.git`); confirm
  which is the intended target before pushing again): Bugs 7/10/10b, the
  workload specs, this doc, and the `zephyr-chipyard-sw` pointer bumps.

**From-scratch verification.** Cloned the pushed branch fresh (`git clone
--branch feature/fp-precision-stripping ... FreshScheduler_fresh`, then
`git submodule update --init` for `zephyr-chipyard-sw`, `modelblaster`, and
`zephyr_ws/zephyr` specifically — `hw/chipyard` is not needed for this flow
and isn't a plain git submodule anyway). The only pieces NOT pulled by the
clone are the untracked, multi-GB toolchain (`tools/miniforge3` — the conda
env, and `tools-manual/zephyr-sdk-1.0.0-beta1` — the Zephyr SDK), for which
there's no committed setup recipe (same situation as this repo's unrelated
`docs/xpurt_env_setup.md` case) — re-provisioning those from scratch is a
separate, much larger undertaking than validating the fixes, so they were
symlinked in from the existing installation instead, plus a one-time copy
of the small `zephyr_ws/.west/config` file (also untracked).

This surfaced one real, previously-invisible bug: `west build` failed at
CMake configure with `File not found: .../harness/backends/spike_single_core.conf`
for every model — that file (and its FireSim sibling,
`firesim_chipyard_singlecore.conf`) existed in the original working tree
but had never actually been committed to modelblaster, despite being a
required `EXTRA_CONF_FILE` for the default `RUNNER=spike` path. This is
unrelated to this session's other fixes (long-standing gap on the
`feat/harness-shared-input` branch) but would have silently blocked
*anyone* cloning fresh. Fixed by committing both files (`650558a`).

With that fixed, the fresh clone reproduced the full pipeline exactly:
same 71/1/1 instance counts, same 733 dispatches, same 704.997ms predicted
makespan, `OVERALL: PASS (3 models)`, and the same timing-accuracy result
(**704.675ms actual, 1.00x predicted** — identical to the original run to
0.03ms). `yolov8_nano`'s golden values differed numerically from the first
run (test-input generation isn't seeded), but `actual == golden` bit-exact
in both runs regardless — expected, not a bug.

## Artifact inventory

| Path | Repo | Status |
|---|---|---|
| `data/toplevel/networks_mlp_dronet_yolo_spike.json` | top-level | untracked |
| `schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json` | top-level | untracked, regenerated by §5 |
| `plots/networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.png` | top-level | untracked, regenerated by §5 |
| `gen/profile/{RVV,scalar}/spike/{mlp_control,dronet,yolov8_nano}` | top-level | untracked symlinks (§3) |
| `xpu-rt/postprocessing.py` | top-level | **tracked, modified, uncommitted** (Bug 7 fix) |
| `xpu-rt/plot.py` | top-level | **tracked, modified, uncommitted** (Bug 10 fix) |
| `scripts/run_xpurt_schedule.py` | top-level | **tracked, modified, uncommitted** (Bug 10b fix) |
| `plots/xpurt_trace_mlp_dronet_yolo_spike.png` | top-level | untracked, real spike execution timeline (§7) |
| `schedules/xpurt_trace_mlp_dronet_yolo_spike.csv` | top-level | untracked, parsed trace data backing the plot above (§7) |
| `zephyr-chipyard-sw/modelblaster/gen/profile/{scalar,RVV}/spike/<model>/...` | nested submodule | untracked, real profile CSVs (§1) |
| `zephyr-chipyard-sw/gen/vmfb/{mlp_control,dronet,yolov8_nano}/generic_riscv64/{scalar,RVV}/<model>.fp32/...` | `zephyr-chipyard-sw` submodule | untracked, dispatch graphs (§2) |
| `zephyr-chipyard-sw/modelblaster/examples/{mlp_control,dronet,yolov8_nano}/{fp32,int8}/generated/` | nested submodule | untracked, per-model extraction/kernel output |
| `zephyr-chipyard-sw/modelblaster/examples/xpurt_demo/run.sh` | nested submodule | **tracked, modified, uncommitted** (Bug 9 fix) |
| `zephyr-chipyard-sw/modelblaster/pipeline/generate_skeleton.py` | nested submodule | **tracked, modified, uncommitted** (Bug 11 + Bug 12 fixes) |
| `zephyr-chipyard-sw/modelblaster/harness_xpurt/CMakeLists.txt` | nested submodule | **tracked, modified, uncommitted** (Bug 12 fix) |
| `data/toplevel/networks_mlp_only_spike.json` | top-level | untracked, minimal single-network repro (§9.1) |
| `data/toplevel/networks_dronet_only_spike.json` | top-level | untracked, minimal single-network repro (§9.2) |
| `data/toplevel/networks_yolov8_only_spike.json` | top-level | untracked, minimal single-network repro (§9, yolov8 leg) |
| `schedules/scheduled_networks_{mlp,dronet,yolov8}_only_spike_greedy_periodic_profiled.json` | top-level | untracked, regenerated by the §9 commands |
| `plots/networks_{mlp,dronet,yolov8}_only_spike_greedy_periodic_profiled.png` | top-level | untracked, regenerated by the §9 commands |
| `plots/xpurt_trace_mlp_dronet_yolo_spike_clk10.png` | top-level | untracked, corrected (10MHz) real execution timeline (§7) |
| `schedules/xpurt_trace_mlp_dronet_yolo_spike_clk10.csv` | top-level | untracked, parsed trace data backing the plot above (§7) |
| `schedules/scheduled_networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.json` | top-level | untracked, **final** regeneration (§10.2 — 71 mlp_control instances, 733 dispatches, superseded the 58-instance version from §5.1) |
| `plots/networks_mlp_dronet_yolo_spike_greedy_periodic_profiled.png` | top-level | untracked, **final** regeneration (§10.2) |
| `plots/xpurt_trace_mlp_dronet_yolo_spike_final.png` | top-level | untracked, **final** real execution timeline, all 3 models PASS (§10.3) |
| `schedules/xpurt_trace_mlp_dronet_yolo_spike_final.csv` | top-level | untracked, parsed trace data backing the plot above (§10.3) |
| `scripts/repro_mlp_dronet_yolo_spike.sh` | top-level | **tracked**, end-to-end automation of sections 0/1/2/3/5/6/7 (see "Quick start" above) |
| `zephyr-chipyard-sw/modelblaster/models/checkpoints/mlp_control/model_6998.pt` | nested submodule | **tracked** (Bug 14 fix) — trained MLP policy checkpoint, previously only on one user's disk |
| `zephyr-chipyard-sw/modelblaster/models/mlp_control.py` | nested submodule | **tracked, modified** (Bug 14 fix — `_DEFAULT_CKPT` now `__file__`-relative) |
| `zephyr-chipyard-sw/modelblaster/models/checkpoints/dronet/best.pt` | nested submodule | **tracked** (Bug 15 fix) — trained DroNet checkpoint, previously only on one user's disk |
| `zephyr-chipyard-sw/modelblaster/models/dronet_arch.py` | nested submodule | **tracked, new** (Bug 15 fix) — vendored copy of `qnn_models/dronet.py`'s `DronetTorch` class |
| `zephyr-chipyard-sw/modelblaster/models/dronet.py` | nested submodule | **tracked, modified** (Bug 15 fix — imports the vendored arch, `_DEFAULT_CKPT` now `__file__`-relative) |
| `zephyr-chipyard-sw/modelblaster/models/yolov8_nano.py` | nested submodule | **tracked, modified** (Bug 16 fix — error message no longer masks the real `ImportError` cause) |
| `scripts/repro_mlp_dronet_yolo_spike.sh` | top-level | **tracked, modified** (Bug 17 fix — same PATH-ordering defect as Bug 13, never fixed in the script itself; also now calls `install_xpurt_deps.sh`) |
| `pyproject.toml` | top-level | **tracked, new** — xpu-rt's own deps, replaces `setup.py` (dependency-management cleanup) |
| `scripts/install_xpurt_deps.sh` | top-level | **tracked, new** — the one place all xpurt-specific deps (on top of a standalone zephyr-chipyard-sw install) are installed from (dependency-management cleanup) |
| `setup.py` | top-level | **removed** — superseded by `pyproject.toml` |
| `env.yml` | top-level | **removed** — orphaned, never wired into any install script (dependency-management cleanup) |
| `zephyr-chipyard-sw/modelblaster/requirements.txt` | nested submodule | **removed** — stale, incomplete subset of modelblaster's own `pyproject.toml` |

The code fixes (`postprocessing.py`, `plot.py`, `run_xpurt_schedule.py`,
`xpurt_demo/run.sh`, `generate_skeleton.py`, `harness_xpurt/CMakeLists.txt`)
are real bug fixes worth keeping; everything else here is either
regenerable output or a workaround (symlinks, `PYTHONPATH` export,
`prune_periodic: false`) that isn't fixed at its actual source. Two
Q31-Gemmini-specific fixes from a parallel investigation this session
(`modelblaster/scripts/validate_q31_matrix.sh`,
`modelblaster/pipeline/generate_kernels.py`'s algorithm-filter wiring) are
out of scope for this doc — they're unrelated to the mlp/dronet/yolo
scheduling flow.
