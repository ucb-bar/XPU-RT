# QRB5165 Hardware Spec — Provenance Analysis

This document accompanies `xpu-rt/data/hardware/qrb5165.yaml`. It records
what was found, what's measured vs inferred vs public-spec, where the
gaps are, who consumes the spec, and what data to collect next.

## 1. What was found

The codebase already had QRB5165 facts spread across six categories of
files. The new spec consolidates them in declaration order:

| Source | Role | Schema / shape |
|---|---|---|
| `xpu-rt/python/xpu_rt/targets/backends/qnn/qrb5165_costs.json` | Per-op cost table (567 entries; 273 measured, 294 infeasible). Sections: `execute`, `init`, `memcpy`, `rescale`, `dequant_quant`. | Custom v1, loaded by `cost_table.CostTable.load`. |
| `xpu-rt/data/profiled/qnn_cost_matrix.json` | Per-op profiled latencies for yolov8n + dronet across CPU/GPU/DSP. | `qnn_cost_matrix_v1`. |
| `xpu-rt/data/profiled/qnn_e2e/measurements.json` | Whole-network solo E2E per backend. | Plain JSON. |
| `xpu-rt/data/calibration/qrb5165.json` | Per-(workload, backend) overhead and contention factors. | `calibration_model_v2` on disk (module wants v3). |
| `xpu-rt/data/profiled/qnn_closed_loop/contention_state.json` | EMA-converged contention factors over 4 closed-loop rounds. | Plain JSON. |
| `xpu-rt/python/xpu_rt/targets/backends/qnn/{board,cost_table,scheduler,profile_lookup}.py` | Lane definitions, board-config dataclass, fallback constants. | Python. |
| `xpu-rt/python/xpu_rt/targets/cards/hexagon_npu.yaml` | Hexagon-only `target_card_v1` stub (8 MiB VTCM). | `target_card_v1`. |
| Public Qualcomm specs | Snapdragon 865 / Adreno 650 / Hexagon 698 architectural facts. | Datasheet. |

## 2. Schema choice

There are three schemas for "describe a target" in the repo:

1. **`target_resource v1.0`** — `xpu-rt/python/xpu_rt/schemas/v1/target_resource.schema.yaml`. Used by the production profiles in `xpu-rt/python/xpu_rt/targets/profiles/blackwell_b200.yaml`, `blackwell_rtx_pro_6000.yaml`, and the `examples/target_profiles/*.yaml` examples (cuda_a100, multi_device, riscv_soc, saturn_opu, trainium1). Loaded by `xpu_rt.targets.schema.TargetProfile.from_dict`. Supports multi-device SoCs, kernel families, host_sync_cost_us per device, free-form `cost_model` and `calibration_data` blocks, and a required `metadata.assumptions` provenance ledger.
2. **`graphcomp_target_config_v1`** — `xpu-rt/configs/targets/*.yaml`. Single-device, Region Dossier V2 shape. Used by the graph-compilation pipeline (`graph_compilation/capture.py`). Doesn't support multi-device.
3. **`target_card_v1`** — `xpu-rt/python/xpu_rt/targets/cards/*.yaml`. Tiny dispatch-mode + memory-tier stub. Used by `xpu_rt.targets.backends.merlin.target_spec`. Too thin for QRB5165's multi-lane geometry.

**We chose `target_resource v1.0`** because (a) it's the only one of the
three that can model a multi-domain SoC in one file, (b) it has a
required provenance block which is essential for an SoC where most
fields are inferred or datasheet-sourced, and (c) it's used by the
most-production profile (Blackwell B200, the paper hardware).

## 3. Measured vs inferred vs public-spec

| Field | Source basis | Confidence |
|---|---|---|
| Lane topology (CPU+GPU+DSP) | datasheet | high |
| Kryo 585 1+3+4 layout | datasheet | high |
| Adreno 650 identity | datasheet | high |
| Hexagon 698 identity + HVX/HMX | datasheet | high |
| VTCM size (8 MiB) | datasheet (matches `hexagon_npu.yaml`) | high |
| L1D / L2 / L3 sizes | datasheet (ARM/Snapdragon ref) | high |
| LPDDR5 capacity (8 GiB) | datasheet | high |
| LPDDR5 bandwidth | **estimated** (no microbenchmark) | low |
| Adreno 650 peak FLOPS | **estimated** (cost table is load-bearing; left 0/unknown) | low |
| Hexagon 698 peak FLOPS | **estimated** (same as above) | low |
| Cross-lane interconnect bandwidth | **estimated** (shared LPDDR5 fabric) | low |
| `host_sync_cost_us` per lane | measured (calibration v2 overhead_us) | high |
| Per-op `execute` rows (273 entries) | measured (`qnn-net-run` / `board_roundtrip`) | high |
| `init.HTA` (20.7 ms), `init.GPU` (6.98 ms) | measured (`qnn-net-run` init stats) | high |
| `memcpy.CPU__CPU` fit | measured (`profile_transfers_on_board`, n=4) | high |
| `memcpy.cross_lane` (CPU↔GPU, GPU↔DSP, ...) | **estimated** (50µs + 8 GB/s fallback in `cost_table.py`) | low |
| `rescale.uint8`, `dequant_quant.{uint8,fp32,fp16}` | measured (n=4 fit) | high |
| Calibration `overhead_us[w][b]` (yolov8n+dronet) | measured (solo E2E − chain sum) | high |
| Calibration `contention_factor.GPU` | **default_no_data** | low |
| Solver policy thresholds | inferred (cross-link to `SolverPolicy` defaults; not QRB5165-specific) | high |
| Anomaly: `strided_slice_0` DSP 46× slower | measured (in `qnn_cost_matrix.json`) | high |
| 294/567 infeasible markers | measured-as-explicit-infeasibility (audit) | high |

Field-count breakdown: of ~50 distinct fields populated in the spec,
roughly **45 % measured, 35 % datasheet/public-spec, 15 % inferred
from code, 5 % estimated/unknown**.

## 4. Coverage gaps (top-3 "unknown" / "would close")

1. **LPDDR5 sustained bandwidth (per-lane and aggregate).** Today
   marked `0` (unknown) on every memory level except `CPU__CPU`
   memcpy. To close: a STREAM-style memcpy microbenchmark on board,
   one run per lane (CPU memset, OpenCL `clEnqueueCopyBuffer` from
   GPU, Hexagon scalar loop). The fallback in `cost_table.py`
   currently assumes 8 GB/s — a measurement would replace that
   constant.

2. **Cross-lane memcpy bridge fits.** The cost table only has
   `CPU__CPU`; cross-lane transfers (`CPU__GPU`, `CPU__DSP`,
   `GPU__DSP`) silently use the 50 µs + 8 GB/s fallback. To close:
   extend `profile_transfers_on_board` to issue actual cross-lane
   buffer copies and add `CPU__GPU`, `CPU__DSP`, `GPU__DSP` rows.
   This is the highest-leverage gap because cross-lane transfers
   appear in every multi-lane schedule.

3. **Calibration file schema drift (v2 → v3).** The on-disk file is
   `calibration_model_v2`; the consuming module
   (`xpu_rt.runtime.calibration`) refuses to load v1/v2. To close:
   re-run `bootstrap_from_solo_measurements` +
   `bootstrap_contention_from_closed_loop` against the existing E2E
   and closed-loop data and overwrite `qrb5165.json`. No new on-board
   data needed — pure re-fit.

Honourable mentions (not in the top 3 but worth tracking):
- `contention_factor.GPU` is `default_no_data`. Closed-loop runs to
  date were CPU+DSP only.
- Per-compute-unit peak FLOPS for all three lanes are `0` (unknown).
  Only matters when roofline modeling lands.

## 5. Cross-reference matrix — who consumes what

The spec is not orphaned: every cross-link points at a file already
loaded by production code.

| Spec field | Consumer module(s) |
|---|---|
| `cost_model.per_op_cost_table` | `targets/backends/qnn/cost_table.py`, `targets/backends/qnn/scheduler.py`, `targets/backends/qnn/profile_lookup.py`, `audit/cost_table_audit.py`, `cli.py` (`_DEFAULT_QNN_COST_TABLE`), `mcp/tools/qnn_flow.py`, `mcp/tools/feedback_loop_tools.py` |
| `cost_model.per_op_cost_matrix` | `scheduler/qnn_real_workload.py` (loads + builds `WorkloadOps`), `runtime/calibration.py` (chain-sum input), `audit/cost_table_audit.py` (`load_cost_matrix_rows`) |
| `cost_model.e2e_solo` | `runtime/calibration.py::bootstrap_from_solo_measurements` |
| `cost_model.bridge_table.memcpy` | `targets/backends/qnn/cost_table.py::CostTable.memcpy_us`, `targets/backends/qnn/transfer_model.py` |
| `cost_model.solver_policy` | `scheduling/policy.py::SolverPolicy` |
| `calibration_data.calibration_model` | `runtime/calibration.py`, `scheduling/feedback_loop.py`, `mcp/tools/feedback_loop_tools.py` |
| `calibration_data.contention_model` | `targets/backends/qnn/contention.py`, `scheduling/feedback_loop.py` |
| `devices[2].memory_hierarchy.vtcm` | `targets/cards/hexagon_npu.yaml` (mirrors), `kernels/envelope_bridge.py` (HVX vector-byte hint), `kernels/contract_translator.py` |
| `metadata.qrb5165_evaluation_models` / canonical workload | `configs/models/paper_yolov8n_dronet_x12.yaml` |

Five concrete production consumers that already touch QRB5165 data
and should/could route through this spec:

1. **`xpu_rt.targets.backends.qnn.cost_table.CostTable`** — currently
   loads the JSON directly. The spec's `cost_model.per_op_cost_table.path`
   is the same path. A future loader could read the spec first and
   forward to `CostTable.load` (one-line change).
2. **`xpu_rt.targets.backends.qnn.scheduler`** — consumes `CostTable`
   and the calibration model; both are now spec-cross-linked.
3. **`xpu_rt.runtime.calibration`** — `bootstrap_from_solo_measurements`
   needs the cost matrix + E2E paths; both are spec-cross-linked.
4. **`xpu_rt.scheduling.feedback_loop`** — multi-tenant 12× DroNet flow.
   Needs the contention state file path; spec-cross-linked.
5. **`xpu_rt.cli`** (`_DEFAULT_QNN_COST_TABLE`) — hardcodes the JSON
   path; could load the spec instead and pull the cross-link.

## 6. Quality flags from the experiments

- **`strided_slice_0` is 46× slower on DSP than GPU on yolov8n.**
  Single worst per-op pathology. The HMX accelerator can't pattern-match
  the slice; the DSP path falls back to a scalar HVX loop. The scheduler
  picks GPU correctly today, but the row stays in the matrix and skews
  any unfiltered family-level statistic. The audit's
  `pathological_ratios` surfaces it.
- **294 of 567 entries in `qrb5165_costs.json` are `infeasible: true`.**
  These are op×lane×dtype combinations that QNN refuses to compile (e.g.
  fp32-on-HTA). Any consumer that reads the table directly must check
  the flag; the typed `execute_us()` API already raises `KeyError`.
- **Calibration v3 sensitivity to per-workload overhead.** Single
  per-backend overhead (v1) overfits to yolov8n; per-(workload, backend)
  (v2) overshoots under contention. v3 splits the residual into a fixed
  overhead term + a multiplicative contention factor. The current
  on-disk file is still v2 — a re-bootstrap is required. The contention
  factor for GPU is `default_no_data` because closed-loop runs to date
  exercise only CPU+DSP.

## 7. Recommended next data-collection items (prioritized)

1. **Cross-lane memcpy fits** (`CPU__GPU`, `CPU__DSP`, `GPU__DSP`).
   Highest leverage — affects every multi-lane schedule. Estimated
   half-day on-board.
2. **Re-bootstrap calibration to v3.** No new data — just run the
   bootstrap helpers against existing E2E + closed-loop files. <1 hour.
3. **Closed-loop GPU rounds.** Add a workload mix that contends GPU
   (e.g. 4× yolov8n on GPU under a CPU co-tenant) so
   `contention_factor.GPU` becomes `measured` instead of `default_no_data`.
4. **LPDDR5 sustained bandwidth.** STREAM-style triad on each lane.
   Day on-board. Replaces the 8 GB/s fallback constant.
5. **Adreno 650 / Hexagon 698 peak FLOPS measurement** (matmul roofline).
   Only relevant once roofline modeling lands; deferrable.
6. **Extend `qnn_cost_matrix.json` beyond yolov8n + dronet** to a
   third workload (the `paper_yolov8n_dronet_x12.yaml` slot already
   anticipates this) so family_backend_specialty has more breadth.
