# Wave 1.9 — Target-specific code inventory + migration plan

**Status**: inventory complete · migrations not yet applied

## Goal

Catalog every place in `python/compgen/` where target-specific
assumptions leak into otherwise-universal code, classify each by
its true scope (any-target / class / vendor / arch), and map it
to its destination under the unified `targets/{class}/{vendor}/{arch}/`
hierarchy. Pre-requisite for Waves 1.10–1.15 (the actual moves).

This document is the source of truth for the migration. Every
mechanical move in Wave 1.14 traces back to a row here.

## The unified hierarchy (recap)

```
python/compgen/
  ── universal modules below stay vendor-blind ──
  api.py                     compile_to_megakernel — calls into adapters
  runtime/autotune/          BackendChoice dispatch — class/vendor/arch lookup
  runtime/lowering/          FX → MegakernelGraph — patterns only
  runtime/event_tensor.py    universal IR
  runtime/megakernel.py      universal IR
  kernels/cost/roofline.py   universal math
  kernels/cost/etc_predict.py   ETC vs eager — vendor-supplied TFLOPS tables
  transforms/emit_cuda_megakernel.py   ← THIS file is misnamed; has both
                                        universal (CgTask struct, kind switch)
                                        AND nvidia-specific (cuLaunch, __shared__)
                                        bits. Splits across the migration.

  ── per-target packages below: BOTH compile-side AND runtime-side ──
  targets/
    backend.py                   existing TargetBackendProtocol (universal)
    backends/{npu,saturn_opu}/   existing examples (already in this shape)
    ── new ──
    gpu/contracts.py             GpuBodyEmitter, GpuProbe, GpuRuntime, GpuCostModel
    gpu/nvidia/common/           CUDA driver, NVRTC, cooperative-launch glue
    gpu/nvidia/blackwell/        sm_100/sm_120 specifics
    gpu/nvidia/hopper/           sm_90 specifics
    gpu/nvidia/ampere/           sm_80/86 specifics
    gpu/amd/                     stub
    cpu/contracts.py             CpuBodyEmitter, CpuRuntime
    cpu/x86/                     stub
    cpu/arm/                     stub
    tpu/contracts.py             TpuBodyEmitter, TpuRuntime, TpuTopology
    custom/                      MCP-registered user targets at session scope
```

Every per-target package has the same internal layout:

```
targets/{class}/{vendor}/{arch}/
  __init__.py        public API; registers itself with the in-process registry
  probe.py           compile-time: detect arch, libraries
  body_emitter.py    compile-time: emit per-op kernel bodies
  cost.py            compile-time: TFLOPS/bandwidth tables, roofline overlay
  runtime.py         runtime: driver wrapper, kernel module, launcher
  README.md          per-leaf rationale + extension notes
```

## Inventory by source file

Format per row: **what's there** → **scope** → **destination**.

### `runtime/native/cuda.py` (entire file is NVIDIA-only, currently universal-shaped)

| Symbol | Scope | Destination |
|---|---|---|
| `CudaUnavailableError` | gpu.nvidia.common | `targets/gpu/nvidia/common/errors.py` |
| `CudaDeviceProbe` | gpu.nvidia.common | `targets/gpu/nvidia/common/probe.py` |
| `_CudaProbeStruct` (26 fields) | gpu.nvidia.common | `targets/gpu/nvidia/common/abi.py` |
| `CudaModule` (NVRTC compile + cuModuleLoadData) | gpu.nvidia.common | `targets/gpu/nvidia/common/module.py` |
| `CudaMegakernelLauncher` | gpu.nvidia.common | `targets/gpu/nvidia/common/launcher.py` |
| `CudaEventTensor`, `CudaDynamicQueue`, `CudaCommGroup` | gpu.nvidia.common | `targets/gpu/nvidia/common/primitives.py` |
| `discover_cublasdx_include` | gpu.nvidia (header-only lib applies to all NVIDIA) | `targets/gpu/nvidia/common/discovery.py` |
| `discover_libcudacxx_include` | gpu.nvidia.common | same |
| `discover_cutlass_include` | gpu.nvidia.common | same |
| `_resolve_cu13_nvrtc_lib_path` | gpu.nvidia.blackwell (cu13 NVRTC is required for sm_100+) | `targets/gpu/nvidia/blackwell/cu13_nvrtc.py` |
| `_compile_via_cu13_nvrtc` | gpu.nvidia.blackwell | same |
| `_load_cu13_nvrtc` | gpu.nvidia.blackwell | same |
| `cu13_nvrtc_available` | gpu.nvidia.blackwell | same |
| `_cu_check`, `_nvrtc_check` | gpu.nvidia.common | `targets/gpu/nvidia/common/_check.py` |
| `_ensure_cuda_driver_context` | gpu.nvidia.common | `targets/gpu/nvidia/common/context.py` |

### `runtime/native/device.py`, `library.py`, `__init__.py`

| Symbol | Scope | Destination |
|---|---|---|
| `Device.create("cuda:0")` | gpu (any-GPU concept; NVIDIA impl) | `targets/gpu/contracts.py::Device` Protocol + `targets/gpu/nvidia/common/device.py` impl |
| `load_library` (picks libcompgen_rt-cuda.so vs cpu.so) | any-target dispatch | becomes `targets.dispatch.load_native_library(target_class)` |
| `_NativeHalUnavailable` | universal | `runtime/native/errors.py` (stays — applies to any HAL backend) |

### `transforms/emit_cuda_megakernel.py` (mixed — splits across migration)

| Region | Scope | Destination |
|---|---|---|
| `DeviceFunctionSource` dataclass | universal IR | stays in `transforms/` (vendor-blind) |
| `CgTask` / `CgCell` structs | universal | `targets/gpu/contracts.py` (any-GPU schedule table format) |
| `CG_TASK_TABLE` / `CG_IN_CELLS` etc. emission | gpu (per-class schedule layout) | `targets/gpu/common/schedule_tables.py` |
| `__constant__` / `__device__` switchover (Wave 1.7) | gpu.nvidia.common (CUDA-specific storage classes) | `targets/gpu/nvidia/common/storage_class.py` |
| `cuLaunchCooperativeKernel` mention | gpu.nvidia.common | `targets/gpu/nvidia/common/launcher.py` |
| `__shared__` / `__syncthreads` / `atomicSub_system` in body templates | gpu.nvidia.common | move when bodies move |
| Megakernel wrapper template (the per-SM dispatch loop) | gpu (any-GPU concept; CUDA-specific syntax today) | `targets/gpu/nvidia/common/megakernel_wrapper.py` until other vendors arrive; abstract later |

### `runtime/lowering/fx_to_megakernel.py`

| Symbol | Scope | Destination |
|---|---|---|
| `_match_diamond`, `_match_ffn` | universal patterns | stay in `runtime/lowering/` |
| `_try_submodule_match` (Wave 1.8) | universal | stays |
| `_TILE_M / _TILE_N / _TILE_K` (32×32×32 fmaf path) | universal-default | stays as default; per-target may override |
| `_TILE_M_CUBLASDX = 64` / `_TILE_K_CUBLASDX = 16` | gpu.nvidia.blackwell | `targets/gpu/nvidia/blackwell/tile_shape.py` |
| `_select_tile_shape` | universal-with-vendor-override | becomes adapter call: `body_emitter.preferred_tile_shape()` |
| `_arch_to_cublasdx_sm` | gpu.nvidia | `targets/gpu/nvidia/common/sm_tag.py` |
| `_cublasdx_gemm_body` | gpu.nvidia.blackwell (cuBLASDx works on hopper+ but the bf16+fp32-acc + 64-tile choice is Blackwell) | `targets/gpu/nvidia/blackwell/body_emitter.py::cublasdx_gemm` |
| `_diamond_bodies` / `_ffn_bodies` (the fmaf path) | becomes adapter dispatch | universal lowering calls `body_emitter.gemm(...)`, `body_emitter.relu(...)` |
| `_emit_linear_full_body` etc. in `fx_generic.py` | universal-default fmaf | stays as fallback emitter; nvidia override exists |
| `_probe_backends` | gpu.nvidia | `targets/gpu/nvidia/common/probe.py` |

### `runtime/autotune/__init__.py`

| Symbol | Scope | Destination |
|---|---|---|
| `BackendChoice` dataclass | universal SHAPE | stays universal but `cublasdx_*`, `cublasdx_sm` move to `vendor_extras["gpu.nvidia.blackwell"]` blob |
| `probe_device(target)` | universal (dispatches by class) | stays; calls into `targets.{class}.probe()` |
| `_resolve_target_arch` | universal-with-vendor-fallback | stays; calls vendor probes |
| `_probe_libraries` (cuBLASDx + cu13 + libcudacxx + cutlass) | gpu.nvidia | `targets/gpu/nvidia/common/probe.py::probe_libraries` |
| `_arch_to_cublasdx_sm` (re-exported here) | gpu.nvidia | already moves with `fx_to_megakernel.py` row |
| Decision tree (`use_cublasdx` rules) | gpu.nvidia.blackwell | `targets/gpu/nvidia/blackwell/decision.py` |
| `_format_rationale` | gpu.nvidia (mentions cuBLASDx, cu13 by name) | becomes vendor-specific section appended to a universal rationale shell |

### `kernels/cost/etc_predict.py`

| Symbol | Scope | Destination |
|---|---|---|
| `EtcCostPrediction` dataclass | universal | stays |
| `WontWinError` | universal | stays |
| `predict_etc_dispatch` | universal-with-vendor-cost-table | stays; reads tables via adapter |
| `_BF16_TC_TFLOPS_PER_SM` | gpu.nvidia (per-arch entries) | `targets/gpu/nvidia/{hopper,blackwell,ampere}/cost.py` |
| `_FP32_SIMT_TFLOPS_PER_SM` | gpu.nvidia | same |
| `_SCHEDULING_OVERHEAD_US_PER_TASK = 1.0` | gpu (any-GPU empirical from cooperative-launch dispatch) | `targets/gpu/contracts.py::DEFAULT_SCHEDULING_OVERHEAD_US` |
| `_EAGER_LAUNCH_OVERHEAD_US = 10.0` | gpu.nvidia (cuBLAS launch overhead) | `targets/gpu/nvidia/common/cost.py` |
| Per-arch SM count table (`{"100": 132, "120": 188, ...}`) | gpu.nvidia (per-arch) | `targets/gpu/nvidia/{arch}/spec.py` |

### `mcp/tools/compile.py`

| Symbol | Scope | Destination |
|---|---|---|
| `compgen_compile_torch_model` | universal | stays |
| `compgen_run_compiled_bundle` | universal | stays; calls vendor runtime via adapter |
| `compgen_cublasdx_header_smoke` | gpu.nvidia | `targets/gpu/nvidia/common/mcp_tools.py` |
| `compgen_run_cuda_source` | gpu.nvidia | `targets/gpu/nvidia/common/mcp_tools.py` |
| `target_arch="sm_100"` default | gpu.nvidia.blackwell hardcode → comes from probe | already auto-detected; remove the default |

### `runtime/probe.py`

| Symbol | Scope | Destination |
|---|---|---|
| `probe_cuda_device` | gpu.nvidia | `targets/gpu/nvidia/common/probe.py` |
| `probe_via_native_hal` | gpu.nvidia.common | same |
| `probe_via_torch` | gpu.nvidia.common (uses torch.cuda) | same |
| `_NativeHalUnavailable` | universal | stays |

### `testing/etc_conformance.py` & `etc_dispatch.py`

| Symbol | Scope | Destination |
|---|---|---|
| Conformance harness | universal | stays |
| `_workload_buffers` (assumes torch.cuda for buffer marshalling) | gpu.nvidia | `targets/gpu/nvidia/common/dispatch.py::workload_buffers` |
| `etc_us` / `eager_us` measurement (uses `cudaEventRecord`) | gpu (any-GPU) | `targets/gpu/contracts.py::EventTimer` Protocol with vendor impl |

### `runtime/baremetal/chipyard.py`, `runtime/embedded/cnn_lowering.py`

These are already non-NVIDIA target-specific code that's outside `targets/`. Move them too:

| File | Scope | Destination |
|---|---|---|
| `runtime/baremetal/chipyard.py` | rocc.chipyard (RocketChip-class accelerator) | `targets/accel/chipyard/runtime.py` |
| `runtime/embedded/cnn_lowering.py` | embedded.npu (saturn-opu et al.) | likely already overlaps with `targets/backends/saturn_opu/` — consolidate |

## Universal modules that are already correctly placed (no migration needed)

- `runtime/event_tensor.py` — pure IR, vendor-blind
- `runtime/megakernel.py` — IR (DeviceCall, EventEdge, MegakernelGraph)
- `runtime/lowering/fx_to_megakernel.py` (matchers only — patterns are vendor-blind)
- `transforms/event_static_schedule.py` — universal scheduling math
- `transforms/event_dynamic_schedule.py` — universal
- `kernels/cost/roofline.py` — universal math (analytic, takes vendor-supplied peaks)
- `runtime/autotune/__init__.py` — entry point + dispatch (after the cuBLASDx-rule split)

## Summary by destination

| Destination | LoC budget | Primary contents |
|---|---|---|
| `targets/gpu/contracts.py` | ~200 | `GpuBodyEmitter`, `GpuProbe`, `GpuRuntime`, `GpuCostModel`, `Device` Protocols |
| `targets/gpu/nvidia/common/` | ~1200 | CUDA driver wrappers, NVRTC, cooperative launcher, generic discovery |
| `targets/gpu/nvidia/blackwell/` | ~800 | cuBLASDx body emitter, cu13 NVRTC, mma.sync, cluster-launch (Wave 1.6 lands here) |
| `targets/gpu/nvidia/hopper/` | ~150 | wgmma path, sm_90 spec |
| `targets/gpu/nvidia/ampere/` | ~100 | older mma atoms, sm_80/86 spec |
| `targets/cpu/contracts.py` | ~150 | `CpuBodyEmitter`, `CpuRuntime` (no event-tensor coordination needed) |
| `targets/cpu/x86/` | ~200 | AVX-512 fmaf body emitter, cdll dispatch |
| `targets/_template/` | ~200 | scaffold for new targets (Wave 1.17) |
| Universal modules (after migration) | unchanged | `api.py`, `runtime/lowering/`, `kernels/cost/`, `transforms/` |

## Migration ordering

The moves in Wave 1.14 happen in this order — each step compiles + tests stay green:

1. **Add scaffolding** (Waves 1.10, 1.11, 1.12): create empty `targets/{class}/{vendor}/{arch}/` directories with stub Protocol/registration. No imports yet.
2. **Move utilities first** (no public API): `_check.py`, `_resolve_cu13_*`, `discover_*`, `_arch_to_cublasdx_sm`. Old import paths kept as re-exports for one round.
3. **Move probes**: `CudaDeviceProbe`, `probe_libraries`. Old paths re-export.
4. **Move emitter helpers**: `_cublasdx_gemm_body`, `_diamond_bodies`, `_ffn_bodies`. Universal `fx_to_megakernel.py` calls into the adapter.
5. **Move runtime**: `CudaModule`, `CudaMegakernelLauncher`. Old paths re-export.
6. **Move BackendChoice cuBLASDx fields** to `vendor_extras` blob. The dispatch path reads adapters; old keys stay in `to_dict()` output for one round to avoid breaking bwell.
7. **Delete deprecated re-exports** after one round of compatibility.

## What stays out of scope for the inventory phase

- Cluster-launch wiring (Wave 1.6) — lands as `targets/gpu/nvidia/blackwell/cluster_launch.py` AFTER the migration is done.
- MHA/MoE patterns (Wave 2.1) — they're universal patterns; the matcher gets new entries but no per-target logic.
- AMD / Intel GPU support — placeholders only; real implementations are future work.

## Acceptance criteria for Wave 1.9

- [x] Every target-specific symbol catalogued with its destination
- [x] Universal modules confirmed vendor-blind (post-migration)
- [x] Migration ordering laid out so each step compiles
- [ ] Reviewed by user before any Wave 1.10+ moves
