# Merlin Integration

Merlin (`third_party/merlin/`, submodule) is the SpacemiT / QRB5165
compiler toolchain that XPU-RT's scheduling stack drives end-to-end. The
heterogeneous compile-schedule-profile loop combines

- Merlin's `compile_dispatch_matrix` (per-target dispatch dumps),
- Merlin's `profile_dispatch_matrix` (on-board mean / median / p99
  latencies per (dispatch, target)),
- XPU-RT's MOSEK scheduler (`xpu_rt.scheduler.solve_makespan`),
- XPU-RT's re-quantisation pass (Phase A2, applied via Merlin's
  iree-compile plugin flag).

A full reference walkthrough lives in
[merlin_integration (legacy notes)](../merlin_integration.md). This page
is a quick map; the legacy doc is authoritative for the loop's exact
arguments and convergence criteria.

## Where the toolchain lives

| Artifact | Path |
|---|---|
| Merlin submodule | `third_party/merlin/` |
| Compile-matrix tool | `third_party/merlin/tools/compile_dispatch_matrix.py` |
| Profile-matrix tool | `third_party/merlin/tools/profile_dispatch_matrix.py` |
| Board roundtrip | `third_party/merlin/tools/board_roundtrip.py` |
| Heterogeneous loop driver | `scripts/heterogeneous_loop.py` |
| XPU-RT scheduler entry | `scripts/run_xpurt_schedule.py` |

The submodule used to live at the repo root as `merlin/`; it was
consolidated under `third_party/` so every external code dependency lives
in one place. All scripts and CMake paths updated accordingly.

## One iteration

```bash
python scripts/heterogeneous_loop.py \
  --merlin-root third_party/merlin \
  --source model.mlir \
  --out-dir build/het_loop_model \
  --targets cpu,qnn_gpu,qnn_hta \
  --diversity-weight 100 \
  --max-rounds 3 \
  --use-cost-table   # Phase D: per-edge transfer costs from qrb5165_costs.json
```

Round structure:

1. `compile_dispatch_matrix.py` → per-target dispatch dumps + matrix.json
2. `profile_dispatch_matrix.py` → profiled_manifest.json (mean_us per
   (dispatch, target))
3. Build Workload + processing-times + transfer-times JSON, with per-edge
   transfer costs derived from
   `xpu_rt.targets.backends.qnn.qrb5165_costs.json`.
4. `solve_makespan(workload, target_diversity_weight=...)` → schedule.json
5. `apply_placement_requantization` (Phase A2, when enabled) — Merlin
   iree-compile plugin flag `--merlin-placement-requant-json`.
6. Terminate when the placement set is stable across two rounds, or when
   `k == max-rounds`.

## Runtime targets

The merged `runtime/CMakeLists.txt` builds two dispatch runners against
the Merlin standalone archive:

```bash
cmake -B runtime/build -S runtime \
      -DCMAKE_BUILD_TYPE=Release \
      -DXPURT_STANDALONE_LIB_PATH=third_party/merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_standalone.a
cmake --build runtime/build --target json_dispatch_runner xpurt_scheduler_runner
```

When `XPURT_STANDALONE_LIB_PATH` is unset, only the standalone
`libxpu_rt` static library + its test binaries are built (no Merlin
dependency required).

## See also

- [merlin_integration (legacy notes)](../merlin_integration.md) — full
  reference for the loop's arguments, IO contracts, and convergence
  criteria.
- [Telemetry feedback](telemetry-feedback.md) — feeding on-board
  measurements back into the recipe cache.
