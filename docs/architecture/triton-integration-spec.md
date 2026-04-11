# Triton Integration Specification

This document specifies how CompGen targets can become Triton backends, enabling developers to write `@triton.jit` kernels that compile to any CompGen-supported hardware.

## Background

Triton is a language and compiler for writing tile-level parallel programs. Its programming model (block loads, tile matmuls, reductions) maps naturally to any tiled accelerator — not just NVIDIA GPUs. Qualcomm's Hexagon-MLIR project proved this by making Hexagon NPU a Triton backend.

The integration gives two capabilities:
1. **Kernel developers** write Triton kernels that compile to custom hardware
2. **PyTorch users** get `torch.compile` integration via the Triton backend

## Architecture

### Two Integration Surfaces

```
┌──────────────┐        ┌──────────────┐
│ BaseBackend  │        │  DriverBase  │
│ (compilation)│        │  (runtime)   │
├──────────────┤        ├──────────────┤
│ supports_    │        │ is_active()  │
│   target()   │        │ get_current_ │
│ parse_       │        │   target()   │
│   options()  │        │ get_active_  │
│ make_ttir()  │        │   torch_     │
│ add_stages() │        │   device()   │
│ pack_        │        │ launcher_cls │
│   metadata() │        │              │
└──────────────┘        └──────────────┘
       │                       │
       └───── Used by ─────────┘
              Triton Runtime
```

### Compilation Flow

```
@triton.jit kernel
    ↓ Triton Frontend (parsing, type inference)
TTIR (Triton Tensor IR)
    ↓ make_ttir() [target-specific]
Optimized TTIR
    ↓ triton-to-linalg [shared infrastructure]
Linalg IR on Tensors
    ↓ target backend stages [fully target-specific]
Object Code / Shared Library
```

The key transition: `triton-to-linalg` converts Triton's block-level ops to structured linalg ops. This is the **universal bridge** — after this point, any linalg-consuming backend can process the IR.

## Triton Op Semantics

Triton operations have clear hardware-independent semantics that map to any tiled accelerator:

| Triton Op | Semantic | What It Does |
|-----------|----------|-------------|
| `tl.load(ptr, mask)` | Tile load | Load a tile of data from memory into registers |
| `tl.store(ptr, val, mask)` | Tile store | Store a tile from registers to memory |
| `tl.dot(a, b)` | Tile matmul | Matrix multiply two tiles, accumulate |
| `tl.exp(x)` | Elementwise | Apply exp to each element |
| `tl.sum(x, axis)` | Reduction | Reduce a tile along an axis |
| `tl.where(cond, a, b)` | Select | Conditional element selection |
| `tl.program_id(axis)` | Grid index | Current tile/block index |
| `tl.arange(start, end)` | Index gen | Generate sequential indices |
| `tl.zeros(shape, dtype)` | Init | Zero-initialized tile |
| `tl.maximum(a, b)` | Elementwise | Element-wise maximum |

### Target Mapping

Each target provides its own mapping from these semantics to hardware:

```python
# Example: NPU mapping
{
    "tile_load":     "dma.load.ch0 + vload",
    "tile_store":    "vstore + dma.store.ch0",
    "tile_matmul":   "vmatpush.weight + vmatmul.acc.mxu0",
    "elementwise":   "vexp.bf16 / vadd.bf16 / vmul.bf16",
    "reduction":     "vredsum.bf16",
    "grid_index":    "loop iteration variable",
}

# Example: GPU mapping (for reference)
{
    "tile_load":     "ld.global",
    "tile_store":    "st.global",
    "tile_matmul":   "wmma.mma.sync",
    "elementwise":   "CUDA math intrinsics",
    "reduction":     "warp shuffle reduce",
    "grid_index":    "blockIdx.x",
}

# Example: CPU mapping
{
    "tile_load":     "memref.load (vectorized)",
    "tile_store":    "memref.store (vectorized)",
    "tile_matmul":   "linalg.matmul (tiled)",
    "elementwise":   "math.exp / arith.addf",
    "reduction":     "vector.reduction",
    "grid_index":    "thread ID",
}
```

## What CompGen Provides Today

| Component | Status | Notes |
|-----------|--------|-------|
| PyTorch capture (Dynamo) | Done | `capture_dynamo_partitions()` |
| Linalg IR (xDSL) | Done | `fx_to_xdsl()` → Payload IR |
| Pattern detection | Done | 8 kernel patterns from FX graphs |
| Kernel contracts | Done | 91 per-shape + 8 pattern-level |
| Golden data | Done | Small + real variants per pattern |
| Ukernel registry | Done | Declaration/match/call ops |
| FP8 quantization | Done | torchAO integration |
| Target profiles | Done | Hardware spec YAML |

## What's Needed for Full Triton Integration

### Required (C++ component)

1. **`translate_linalg_to_obj()`** — C++ function exposed via pybind11 that takes linalg MLIR and produces object code for the target. This is the core compilation backend.

2. **`add_triton_plugin()` in CMakeLists.txt** — Register with Triton's build system.

3. **`triton-shared` dependency** — The `triton-to-linalg` pass lives in the `triton-shared` repo (shared across all non-CUDA backends).

### Required (Python component)

4. **`CompGenBackend(BaseBackend)`** — Implements `supports_target()`, `parse_options()`, `make_ttir()`, `add_stages()`, `pack_metadata()`.

5. **`CompGenDriver(DriverBase)`** — Implements `is_active()`, `get_current_target()`, `get_active_torch_device()`.

6. **`CompGenLauncher`** — Executes compiled kernels, handles I/O marshaling.

### Integration Path (No C++ Required)

CompGen can provide a **Python-based backend** that generates target code without the full C++ MLIR pipeline:

```python
class CompGenPythonBackend(BaseBackend):
    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        stages["ttsharedir"] = lambda src, meta: ttir_to_linalg(src)
        
        # Python-based lowering (no C++ needed)
        stages["o"] = lambda src, meta: self._python_lower(src, options, meta)
    
    def _python_lower(self, linalg_ir, options, metadata):
        # Parse linalg IR
        # Apply CompGen's xDSL passes (tiling, microop decomposition)
        # Emit target code (assembly, C, Python)
        return compiled_bytes
```

This trades compilation speed for implementation simplicity — suitable for development and kernel authoring, with a C++ backend added later for production.

## Comparison: Hexagon vs CompGen

| Aspect | Hexagon-MLIR | CompGen |
|--------|-------------|---------|
| Frontend (Triton) | `HexagonBackend(BaseBackend)` | `CompGenBackend(BaseBackend)` (to build) |
| Frontend (PyTorch) | torch-MLIR `fx.export_and_import()` | TorchDynamo `capture_dynamo_partitions()` (done) |
| IR convergence | linalg-on-tensors | xDSL Payload IR (linalg equivalent, done) |
| Custom dialect | HexKL (matmul microops) | Ukernel dialect (done) + Accel dialect (done) |
| Tiling | VTCM tiling pass (C++) | Target stages (Python, done) |
| DMA | hexagonmem.copy → memref.dma (C++) | To build per target |
| Vectorization | HVX 128-bit (C++) | VPU/SIMD per target (to build) |
| Microop decomp | hexkl-to-llvm (C++) | To build per target |
| Code generation | hexagon-clang (C++) | Python-based (to build) |
| Runtime | Device/simulator executor | Local executor (done) |
| Options | `HexagonOptions` (60+ fields) | `TargetOptions` (to build) |

## Next Steps

1. Implement `TargetBackendProtocol` and `TargetOptions` (scaffold)
2. Implement NPU-specific options and backend as first concrete target
3. Evaluate `triton-shared` for the `triton-to-linalg` pass integration
4. Build Python-based lowering as initial backend (fast iteration)
5. Add C++ pybind11 backend for production compilation speed
