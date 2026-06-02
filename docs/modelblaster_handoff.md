# ModelBlaster ⇄ XPU-RT iterative scheduling loop — full handoff brief

> Self-contained brief for the ModelBlaster-familiar Claude Code session. Assumes
> **no prior context** about the XPU-RT scheduler-advisor work. Read top to bottom;
> the in-repo copy of the contracts/commands lives at
> `/scratch2/agustin/XPU-RT/docs/iterative_firesim_loop.md`, but THIS file is the
> complete version.

---

## 0. TL;DR — what we're doing and your role

We built an **agentic scheduler advisor** in XPU-RT that, given a schedule, says whether
it meets its deadline, where the bottleneck is, whether dispatch granularity is too
fine/coarse, and proposes concrete changes across three axes:

- **A. scheduler/placement** — swap solver/scheduler (done on the XPU-RT side, predicted).
- **B. profiler/backend** — schedule under different backends (gemmini_q31 vs rvv) (XPU-RT side, predicted).
- **C. granularity** — **merge dispatches (fuse) or break them down (split)** to trade
  per-dispatch/transition overhead against parallelism.

The **inner loop is predicted** (XPU-RT, seconds). The **authoritative timing is FireSim**
(your side), and **axis C requires ModelBlaster** to physically realize fused/split kernels.

**Your role:** (1) realize axis-C granularity changes (fuse/split ops → regenerate kernels,
verify on **spike**, re-profile on **FireSim**), (2) build + run the candidate **bundle** on
FireSim in one batched session, (3) produce **actual (measured) Gantts**. The XPU-RT side
hands you a `firesim_batch.json` (Contract 1) and per-network fusion hints (Contract 2);
you hand back new profiles + dispatch graphs + traces.

Workload for the demo: **1 yolov8_nano + 4 mlp_control + 2 dronet** on `firesim_rocket_saturn`.

---

## 1. Repos, branch, access

- **XPU-RT** (scheduler + advisor + drivers): `/scratch2/agustin/XPU-RT`
  - Active branch: **`xpurt-scheduler-advisor`** (off `sched-feedback-exp`; ~38 commits, **not pushed**).
    `cd /scratch2/agustin/XPU-RT && git checkout xpurt-scheduler-advisor`
  - The flat python package lives in `xpu-rt/` (modules imported flat, e.g. `from advisor import ...`).
- **ModelBlaster**: `/scratch2/agustin/ModelBlaster` — also vendored as a submodule at
  `/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/modelblaster` (init with
  `git submodule update --init zephyr-chipyard-sw` from the XPU-RT root if empty).
- Profiles + dispatch graphs the scheduler reads:
  - profiles: `gen/profile/<hw>/firesim_rocket_saturn/<model>/.../results.csv`
    (present for `gemmini_q31` and `V256D128_rvv`; **`scalar` is NOT profiled**)
  - dispatch graphs: `zephyr-chipyard-sw/gen/vmfb/<model>/.../<model>.<quant>_dispatch_graph.json`

---

## 2. Environment

- **XPU-RT tools run with system `python3`** (has numpy/matplotlib; no cvxpy needed for
  advisor/greedy/decomposed). Tests: `cd xpu-rt && python3 -m unittest discover -s tests`.
- **ModelBlaster / FireSim** (your side — verify against your own setup; this is the
  documented flow):
  ```bash
  cd /scratch2/agustin/XPU-RT/zephyr-chipyard-sw   # (or your FreshScheduler checkout)
  source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
  source scripts/set_envvars_sdk.sh
  ```
- **spike** = fast functional sim (kernel-gen / correctness inner loop).
  **FireSim** = cycle-accurate timing (slow: infrasetup ~15–30 min, runworkload ~1–2 min →
  **batch many candidates per FireSim session**).

---

## 3. The loop

```
baseline schedule ──► advisor diagnoses ──► propose a BUNDLE of candidates
   (deadline? bottleneck? granularity?)        (axes A/B/C)
        ▲                                            │
        │                                            ▼
   compare + pick winner ◄── run candidates ──┬─ A/B: XPU-RT predicted (seconds)
   before/after Gantt                         └─ C : ModelBlaster (spike re-gen → FireSim)
```

- XPU-RT runs A/B predicted and **evaluates C predicted too** (via `rewrite.py`
  fuse/split + re-schedule), but **fusion's real payoff (launch/transition overhead) only
  shows on FireSim** — the predicted model has no per-dispatch overhead term. So the
  *decision* of what to fuse/split is made predicted (+ an overhead-aware estimate); the
  *measured* benefit comes from your FireSim run.

---

## 4. The workload spec

`/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json`
- `hardware.machines = {cpu_p:1, cpu_e:1}`; `hardware.profile_hw = {cpu_p: gemmini_q31, cpu_e: V256D128_rvv}` (heterogeneous baseline).
- `hardware.profile = {target: firesim_rocket_saturn, topo_tag: topo_0, topo_tag_override: true, gen_root: zephyr-chipyard-sw/gen}`.
- networks: `mlp_control` (period 10, window_duration 10, num_instances 4), `dronet`
  (period 20, window 20, num_instances 2), `yolov8_nano` (one-shot). `dispatch_deps_path`
  points at each model's `_dispatch_graph.json`.
- Note: periodic `window_duration` is **not** propagated to `op.deadline_us`, so the advisor
  takes the frame deadline from `--deadline-us` (the drivers compute `auto`).

---

## 5. XPU-RT side — commands (what we produce; you can run these too)

```bash
cd /scratch2/agustin/XPU-RT

# schedule (emits schedule fixture + _metrics.json + _report.json (schema v2, per-dispatch list))
python3 scripts/run_xpurt_schedule.py --networks-json <spec> --solver decomposed
#   --solver {milp,greedy,greedy_periodic,decomposed}; --solver milp picks a registry
#   algorithm via --scheduler {mosek,heft,peft,edf,cpsat,milp_*,...}

# diagnose a schedule (deadline / bottleneck / granularity / rebalance|coarsen|finer + projected makespan)
python3 xpu-rt/advisor.py --report schedules/scheduled_<...>_report.json --deadline-us 65 --gantt

# terminal ASCII Gantt (deadline marker, shaded late dispatches)
python3 xpu-rt/plot_gantt.py --report schedules/scheduled_<...>_report.json --deadline-us 65

# sweep many schedulers on one workload + advisor per scheduler
python3 scripts/profile_schedulers.py --networks-json <spec> --schedulers decomposed,heft,peft,edf --deadline-us 65

# axis B: compare backends (gemmini vs rvv; missing backends skipped)
python3 scripts/compare_backends.py --networks-json <spec> --solver decomposed --deadline-us 65

# the iterative driver: baseline -> advise -> bundle -> run A/B predicted -> winner
#   emits artifacts/iterate/{report.md, iteration_result.json, firesim_batch.json, before_after_gantt.png}
python3 scripts/iterate_firesim.py --networks-json <spec> --baseline-solver decomposed --deadline-us auto --gantt

# one-command demo (predicted-only)
bash scripts/demo_iterate_firesim.sh

# axis C — EVALUATED merge-vs-split decision (reuses rewrite.py fuse/split + re-schedules each);
#   emits the chosen transform as a Contract-2 hint (fusion_hints or split_hints)
python3 scripts/granularity_loop.py --networks-json <spec> --baseline-solver decomposed \
  --emit-hint artifacts/iterate/granularity_hint.json
#   On the demo workload it decides MERGE: fuse mlp_control dispatches [0..5] (removes 5
#   dispatches, predicted makespan -5us via removed cross-device transitions). Fusing's full
#   payoff (launch overhead) is NOT in the predicted model -> confirm on FireSim. Splitting
#   yolo ops shows 0 predicted gain (correctly not chosen).
```

Demo result you can reproduce: baseline `decomposed` 75.6µs **misses** a 65µs budget →
`heft` 54.4µs **meets** (−28%); axis B: gemmini-on-P-core beats rvv-on-P-core; axis C emits
fusion hints for all 3 networks.

---

## 6. Contracts

### Contract 1 — candidate bundle (`xpurt.candidate_bundle/v1`) → `artifacts/iterate/firesim_batch.json`
```json
{
  "contract": "xpurt.candidate_bundle/v1",
  "deadline_us": 65.0,
  "baseline": {"solver": "decomposed", "scheduler": null,
               "profile_hw": {"cpu_p": "gemmini_q31", "cpu_e": "V256D128_rvv"}},
  "candidates": [
    {"id":"A1","axis":"scheduler","realizable_by":"xpurt","solver":"greedy","scheduler":null,"profile_hw":{...}},
    {"id":"B1","axis":"backend","realizable_by":"xpurt","solver":"decomposed","scheduler":null,
     "profile_hw":{"cpu_p":"V256D128_rvv","cpu_e":"V256D128_rvv"}},
    {"id":"C1","axis":"fusion","realizable_by":"modelblaster","solver":"...","scheduler":null,
     "profile_hw":{...}, "hints": { /* Contract 2 */ }}
  ]
}
```
- `realizable_by:"xpurt"` → reproduce by running `run_xpurt_schedule.py` with that
  solver/scheduler and a spec whose `hardware.profile_hw` = `profile_hw`. (We provide the
  schedule fixture already; you build `harness_xpurt` for it.)
- `realizable_by:"modelblaster"` → **your job**: apply `hints`, re-profile, re-schedule.

### Contract 2 — fusion hints (`modelblaster.fusion_hints/v1`)
`fuse_groups` are **local dispatch ids in that model's own dispatch graph** (shared across
periodic instances), derived from sub-1k-µs dependency chains the advisor flagged.
```json
{
  "contract": "modelblaster.fusion_hints/v1",
  "reason": "granularity verdict 'too_fine': fuse adjacent sub-1k-us dispatch chains ...",
  "networks": [
    {"network": "mlp_control", "fuse_groups": [[0,1,2]], "n_tiny": 9},
    {"network": "dronet",      "fuse_groups": [[3,4],[7,8]], "n_tiny": 12},
    {"network": "yolov8_nano", "fuse_groups": [[...]], "n_tiny": ...}
  ]
}
```
`granularity_loop.py` also emits a **split** variant when splitting wins:
```json
{"contract": "modelblaster.split_hints/v1", "reason": "...",
 "networks": [{"network": "dronet", "split_ops": [{"op": 4, "n_splits": 2}]}]}
```

---

## 7. ModelBlaster side — what to implement / run (your tasks)

ModelBlaster is a PyTorch→Zephyr/RISC-V inference compiler (trace → IR `graph.json` → C
kernels → Zephyr harness → spike/FireSim profile). The 5-stage pipeline:
`pipeline/extract_graph.py` → `generate_skeleton.py` → `generate_kernels.py` →
`emit_dispatch_graph.py` → (schedule) → `ingest_xpurt_schedule.py` + `generate_xpurt_main.py`.
Authoritative docs: `modelblaster/notes/xpurt_walker_semantics.md`,
`modelblaster/notes/scheduler_investigation.md`, and `modelblaster/examples/*/run.sh`.
**Verify the exact commands below against your run.sh / notes — they're the documented flow.**

### 7a. Realize axis-C granularity (Contract 2) — THE NEW WORK
ModelBlaster has **no fusion/granularity knob today** (fusions are hardcoded pattern
detections in `extract_graph.py`). To realize a hint you need to:
1. For each `fuse_group` in a network, **merge those ops in the IR** (extend
   `pipeline/extract_graph.py` post-trace: merge the listed `dispatch_id`s into one op, fold
   their `depends_on`, reassign ids) and add a **fused kernel** in
   `pipeline/reference_kernels.py` (or via the LLM codegen path).
   - For a `split` hint: decompose one heavy op into N sub-ops (chain them).
2. **Verify on spike** (fast): `RUNNER=spike QUANT=int8 TARGET=rvv bash modelblaster/examples/<model>/run.sh`
3. **Emit the coarser dispatch graph**:
   `python -m modelblaster.pipeline.emit_dispatch_graph --ir <graph.json> --out-root gen/vmfb --target generic_riscv64 --hw RVV`
4. **Re-profile on FireSim** (see 7c) → new `results.csv`.
5. Hand back the new `*_dispatch_graph.json` + `results.csv` so XPU-RT re-schedules and
   re-diagnoses (closing the loop). The advisor's predicted Δ ≈ overhead saved; FireSim gives
   the measured Δ.

### 7b. Build + run the bundle (Contract 1) on FireSim — batched
For each candidate, the XPU-RT schedule fixture is at
`schedules/scheduled_<spec>_<tag>_profiled.json`. Build the schedule-driven multi-model ELF:
```bash
SCHEDULE_JSON=schedules/scheduled_<...>_profiled.json \
MODELS=yolov8_nano,mlp_control,mlp_control,mlp_control,mlp_control,dronet,dronet \
BACKENDS=rvv,gemmini_q31,scalar RUNNER=firesim XPURT_TRACE=1 \
bash modelblaster/examples/xpurt_demo/run.sh
```
`ingest_xpurt_schedule.py` validates `(network, dispatch_id)` against the IR and remaps
ir_dispatch_id → codegen_idx (zero-cost ops are filtered — see
`notes/xpurt_walker_semantics.md §2`). Run all candidates in **one** FireSim session.

### 7c. (Re)profile a (model, backend) on FireSim
```bash
PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=firesim_rocket_saturn PROFILE_CORES=0,1,2,3 \
PROFILE_CLOCK_MHZ=1000.0 RUNNER=firesim QUANT=int8 TARGET=<rvv|gemmini_q31> \
bash modelblaster/examples/<model>/run.sh
# -> gen/profile/<hw>/firesim_rocket_saturn/<model>/.../results.csv
```

### 7d. Produce ACTUAL Gantts (for the PR before/after)
The harness emits a trace; parse the uartlog to `xpurt_trace.csv`, then:
```bash
python3 /scratch2/agustin/XPU-RT/xpu-rt/plot_gantt.py --trace <xpurt_trace.csv> --out actual.png
```
`render_gantt(trace_csv)` draws predicted-vs-actual side by side. Compare against the
XPU-RT predicted `artifacts/iterate/before_after_gantt.png`.

---

## 8. Data flow (paths)

```
ModelBlaster                         XPU-RT
extract_graph -> graph.json
emit_dispatch_graph -> gen/vmfb/<model>/.../_dispatch_graph.json  ──►  networks_*.json dispatch_deps_path
firesim profile -> gen/profile/<hw>/firesim_rocket_saturn/<model>/.../results.csv  ──►  profile_loader (run_xpurt_schedule --profiled)
                                                                  run_xpurt_schedule -> schedules/scheduled_*_profiled.json (+ _report.json)
ingest_xpurt_schedule <── schedules/scheduled_*_profiled.json
harness_xpurt -> zephyr.elf -> FireSim -> xpurt_trace.csv  ──►  plot_gantt --trace (actual Gantt)
```

---

## 9. What to hand back to XPU-RT

1. New `*_dispatch_graph.json` + `results.csv` for any fused/split (axis-C) variant.
2. Measured makespan per candidate (from FireSim) — so the advisor can re-rank with real numbers.
3. `xpurt_trace.csv` per run → actual Gantts.
Then XPU-RT re-runs `advisor.py` / `iterate_firesim.py` on the measured reports to propose
the next bundle, closing the loop.

## 10. First concrete task

1. `git submodule update --init zephyr-chipyard-sw` (if empty) and `conda activate zephyr`.
2. Reproduce the predicted demo: `bash scripts/demo_iterate_firesim.sh`; open
   `artifacts/iterate/report.md` + `firesim_batch.json`.
3. Build + run the `firesim_batch.json` candidates on FireSim (one batched session); collect
   measured makespans + `xpurt_trace.csv`.
4. For the `C1` fusion candidate, apply Contract-2 hints (extend `extract_graph.py` + fused
   kernels), spike-verify, re-profile, hand back the new graph/profile.
5. Render actual before/after Gantts and report measured numbers back.
```
