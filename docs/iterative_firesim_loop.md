# Iterative FireSim scheduling loop — runbook & contracts

This is the handoff for driving the **iterative scheduling-improvement loop** on the
FireSim 3-model-family workload (**1 yolov8_nano + 4 mlp_control + 2 dronet**). The
xpurt side (advisor, bundle proposer, drivers, Gantt) is built and tested; the
**ModelBlaster session leads the real spike/FireSim end-to-end** using the contracts
below.

## The loop

```
baseline schedule ──► advisor diagnoses ──► propose a BUNDLE of candidates
   (deadline? bottleneck? granularity?)        (axes A/B/C)
        ▲                                            │
        │                                            ▼
   compare + pick winner ◄── run candidates ──┬─ A/B: xpurt, predicted (seconds)
   before/after Gantt                         └─ C : ModelBlaster (spike re-gen → FireSim)
```

- **Inner loop = fast & predicted**: scheduler/placement (A) and profiler/backend (B)
  candidates are scheduled from the existing profiles in seconds — no FireSim.
- **Authoritative timing = batched FireSim**: build the whole candidate bundle and run
  it in **one** FireSim session (infrasetup is the ~15–30 min cost; amortize it).
- **Kernel-gen / fusion iteration = spike**: realize axis-C fusion on spike (fast,
  functional) before committing a candidate to the FireSim batch.

## xpurt side (already built — predicted-only, no FireSim)

```bash
# one-command demo: baseline → iterate → winner → report + before/after Gantt
bash scripts/demo_iterate_firesim.sh

# the driver directly
python3 scripts/iterate_firesim.py \
  --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
  --baseline-solver decomposed --deadline-us auto --gantt --out-dir artifacts/iterate

# axis-B backend comparison on its own
python3 scripts/compare_backends.py \
  --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
  --solver decomposed --deadline-us 70

# diagnose any one schedule
python3 xpu-rt/advisor.py --report schedules/scheduled_<...>_report.json --deadline-us 70 --gantt
```

`iterate_firesim.py` emits to `artifacts/iterate/`:
- `report.md` — baseline vs winner table, advisor narrative, axis-B comparison, before/after Gantt.
- `iteration_result.json` — machine-readable rows + winner + deadline.
- `firesim_batch.json` — **the candidate set for the ModelBlaster session to build + run** (below).
- `before_after_gantt.png` — stacked predicted Gantt for the PR.

Note: periodic `window_duration` is not propagated to `op.deadline_us`, so the advisor's
deadline comes from `--deadline-us` (`auto` = midpoint between the best candidate and the
baseline, so the baseline misses and the winner meets — pass a real frame budget for
production targets). `scalar` has no profiles, so axis B compares `gemmini_q31` vs
`V256D128_rvv` (missing backends are skipped, not errors).

## Contract 1 — candidate bundle (`xpurt.candidate_bundle/v1`)

Produced by `xpu-rt/bundle.py:propose_bundle(...)`; serialized in `firesim_batch.json`.

```json
{
  "contract": "xpurt.candidate_bundle/v1",
  "deadline_us": 70.0,
  "baseline": {"solver": "decomposed", "scheduler": null,
               "profile_hw": {"cpu_p": "gemmini_q31", "cpu_e": "V256D128_rvv"}},
  "candidates": [
    {"id": "A1", "axis": "scheduler", "realizable_by": "xpurt",
     "solver": "greedy", "scheduler": null, "profile_hw": {...}, "rationale": "..."},
    {"id": "B1", "axis": "backend", "realizable_by": "xpurt",
     "solver": "decomposed", "scheduler": null,
     "profile_hw": {"cpu_p": "V256D128_rvv", "cpu_e": "V256D128_rvv"}, "rationale": "..."},
    {"id": "C1", "axis": "fusion", "realizable_by": "modelblaster",
     "solver": "...", "scheduler": null, "profile_hw": {...},
     "hints": { /* Contract 2 */ }, "rationale": "..."}
  ]
}
```

- `realizable_by: "xpurt"` → reproduce by running `scripts/run_xpurt_schedule.py` with that
  `--solver`/`--scheduler` and a networks JSON whose `hardware.profile_hw` = `profile_hw`.
- `realizable_by: "modelblaster"` → apply `hints` (Contract 2), then re-profile + re-schedule.

## Contract 2 — fusion hints (`modelblaster.fusion_hints/v1`)

Produced by `xpu-rt/bundle.py:fusion_hints_from_diagnosis(...)` from the advisor's coarsen
verdict (only when granularity is `too_fine`). `fuse_groups` are op-id chains in that
network's dispatch graph that should collapse into one coarser dispatch.

```json
{
  "contract": "modelblaster.fusion_hints/v1",
  "reason": "granularity verdict 'too_fine': fuse adjacent sub-1k-us dispatch chains ...",
  "networks": [
    {"network": "mlp_control", "fuse_groups": [[0, 1, 2], [7, 8]], "n_tiny": 9}
  ]
}
```

## ModelBlaster session — what to implement / run

For each candidate in `firesim_batch.json`:

1. **axis A/B (`realizable_by: xpurt`)** — build `harness_xpurt` for that `(solver, scheduler,
   profile_hw)` schedule and add it to the FireSim batch. The schedule JSON is already produced
   by the xpurt driver (`schedules/scheduled_*_profiled.json`); ingest via
   `pipeline/ingest_xpurt_schedule.py` + `generate_xpurt_main.py`.
2. **axis C (`realizable_by: modelblaster`)** — realize the fusion hints:
   - rewrite the model graph to merge each `fuse_group` (extend `pipeline/extract_graph.py`;
     add fused kernels in `pipeline/reference_kernels.py` as needed),
   - **verify on spike** (`validation/spike_runner.py`) — fast inner iteration,
   - emit the coarser dispatch graph + (re)profile on FireSim,
   - feed the new profile/`*_dispatch_graph.json` back so xpurt can re-schedule.
3. **Run the batch on FireSim once** (`validation/firesim_runner.py`) — amortized infrasetup.
4. **Produce actual Gantts**: parse the `xpurt_trace.csv` from the uartlog and render
   predicted-vs-actual with `python3 xpu-rt/plot_gantt.py --trace <trace.csv> --out actual.png`.
   Compare against the predicted before/after composite from the xpurt driver.

The xpurt advisor (`xpu-rt/advisor.py`) and the comparison (`scripts/compare_backends.py`,
`scripts/profile_schedulers.py`) can be called on any emitted `*_report.json` to diagnose the
measured runs and propose the next bundle — closing the loop.
