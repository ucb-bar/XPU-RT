# Target Backend Model

This document describes CompGen's target backend architecture, inspired by Qualcomm's [Hexagon-MLIR](https://github.com/quic/hexagon-mlir) project which demonstrates how to make custom hardware a first-class citizen in both **Triton** and **PyTorch**.

## How It Works: The Hexagon Example

Hexagon-MLIR enables two powerful integration points:

1. Developers write `@triton.jit` kernels and they compile to Hexagon NPU
2. PyTorch models export via torch-MLIR and compile to Hexagon NPU

Both paths converge to **linalg IR**, which the Hexagon backend lowers to hardware code.

```
                    ┌─────────────────────┐
                    │  @triton.jit kernel  │
                    └─────────┬───────────┘
                              │ Triton compiler
                              │ TTIR → triton-to-linalg
                              ▼
┌─────────────┐         ┌──────────┐         ┌──────────────────────┐
│ PyTorch model│────────→│ Linalg IR│────────→│ Target Backend       │
└─────────────┘         └──────────┘         │ (tiling, DMA, HexKL, │
      │ torch-MLIR            ▲               │  vectorization, asm) │
      │ fx.export_and_import  │               └──────────┬───────────┘
      └───────────────────────┘                          │
                                                         ▼
                                                   Object Code / .so
```

## Triton Backend Plugin System

Triton has a pluggable backend architecture. Any hardware target registers as a backend plugin and receives `@triton.jit` kernels for compilation.

### Registration Mechanism

**Build-time registration:**
```bash
# Set TRITON_PLUGIN_DIRS to point to your backend
export TRITON_PLUGIN_DIRS="/path/to/my_backend"

# Triton's build system discovers the plugin via CMakeLists.txt
# which must call add_triton_plugin(TritonMyBackend ...)
```

The plugin exposes C++ lowering functions to Python via pybind11:
```cpp
// my_backend/python/triton_my_backend.cc
void init_triton_my_backend(py::module &&m) {
    m.def("translate_linalg_to_obj", &translate_linalg_to_obj);
    // ... other functions
}
```

### Backend Class (Compilation)

Every Triton backend implements `triton.backends.compiler.BaseBackend`:

```python
from triton.backends.compiler import BaseBackend
from triton.compiler import GPUTarget

class MyBackend(BaseBackend):
    """Compilation backend for my hardware target."""
    
    binary_ext = "o"  # Output file extension
    
    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        """Return True if this backend handles the given target."""
        return target.backend == "my_target"
    
    def parse_options(self, opts: dict) -> MyOptions:
        """Convert option dict to frozen dataclass."""
        args = {k: opts[k] for k in MyOptions.__dataclass_fields__ if k in opts}
        return MyOptions(**args)
    
    @staticmethod
    def make_ttir(mod, metadata, opt) -> ModuleOp:
        """Apply Triton IR optimization passes."""
        pm = ir.pass_manager(mod.context)
        pm.add_inliner()
        pm.add_rewrite_tensor_pointer()
        pm.add_canonicalizer()
        pm.add_cse()
        pm.add_triton_licm()
        pm.add_loop_unroll()
        pm.run(mod)
        return mod
    
    def add_stages(self, stages: dict, options: MyOptions, language: str):
        """Register the compilation pipeline as named stage lambdas."""
        # Stage 1: Triton IR optimization
        stages["ttir"] = lambda src, meta: self.make_ttir(src, meta, options)
        
        # Stage 2: Triton → Linalg conversion
        stages["ttsharedir"] = lambda src, meta: ttir_to_linalg(src)
        
        # Stage 3: Linalg → object code (calls into C++ backend)
        stages["o"] = lambda src, meta: linalg_to_obj(src, options, meta)
        
        # Stage 4 (optional): Object → shared library
        stages["so"] = lambda src, meta: obj_to_shared_lib(src, meta)
    
    def pack_metadata(self, metadata) -> tuple:
        """Pack kernel metadata for the runtime launcher."""
        return (metadata.num_warps, metadata.name, metadata.return_types, ...)
```

### Driver Class (Runtime)

Every Triton backend implements `triton.backends.driver.DriverBase`:

```python
from triton.backends.driver import DriverBase

class MyDriver(DriverBase):
    """Runtime driver for my hardware target."""
    
    def __init__(self):
        super().__init__()
        self.launcher_cls = MyLauncher()  # Kernel execution class
    
    def is_active(self) -> bool:
        """Return True if this driver should handle the current context."""
        return True
    
    def get_current_target(self) -> GPUTarget:
        """Return the current hardware target."""
        return GPUTarget("my_target", 0, 0)
    
    def get_active_torch_device(self) -> torch.device:
        """Return the PyTorch device for this target."""
        return torch.device("cpu")  # or custom device
```

### Activation

```python
import triton
from my_backend.driver import MyDriver

# Activate the backend
triton.runtime.driver.set_active(MyDriver())

# Now @triton.jit kernels compile to my_target
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr = 1024):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < n)
    y = tl.load(y_ptr + offsets, mask=offsets < n)
    tl.store(out_ptr + offsets, x + y, mask=offsets < n)

# This compiles to my_target, not CUDA
add_kernel[(n // 1024,)](x, y, out, n)
```

## Compilation Stage Pipeline

Hexagon's pipeline processes through 4 named stages. The key insight is that **Triton's shared IR converts to linalg**, which is the universal intermediate:

```
TTIR (Triton Tensor IR)
  │ make_ttir(): 11 passes
  │   inline, rewrite_tensor_pointer, canonicalize, combine,
  │   reorder_broadcast, cse, triton_licm, symbol_dce, loop_unroll
  ▼
TTSHAREDIR (→ Linalg on Tensors)
  │ ttir_to_ttsharedir(): triton-shared-opt --triton-to-linalg-experimental
  │   Converts Triton block-level ops to structured linalg ops
  ▼
OBJ (Object Code) — target-specific from here
  │ ttsharedir_to_obj(): C++ backend pipeline
  │   bufferization → tiling (VTCM) → DMA insertion →
  │   HexKL matmul decomposition → HVX vectorization →
  │   LLVM codegen → hexagon-clang
  ▼
SO (Shared Object)
  │ obj_to_so(): hexagon-clang++ linking
  │   Links with: qhmath_hvx, libhexkl_micro, async_runtime
  ▼
Executable on device/simulator
```

### CompGen's Equivalent Stages

CompGen already operates at the linalg level (xDSL Payload IR). The stages needed for any target:

```
CompGen Payload IR (xDSL linalg)
  │
  │ Stage: Tiling
  │   Tile linalg ops to target geometry (e.g., 32x32 for MXU, 64x64 for VTCM)
  ▼
  │ Stage: Memory Planning
  │   Allocate scratchpad memory, insert DMA transfers at tile boundaries
  ▼
  │ Stage: Microop Decomposition
  │   linalg.matmul → target.tiled_matmul → target.micro_* sequence
  │   (Like HexKL: matmul → setup_acc → copy_submatrix → mm → read_acc)
  ▼
  │ Stage: Code Generation
  │   Emit target ISA, assembly, C, or binary
  ▼
Compiled artifact (asm / .o / .so / Python)
```

## PyTorch Integration (torch-MLIR Path)

Hexagon's PyTorch path uses `torch-MLIR` to export models to linalg IR, then the same backend pipeline processes them:

```python
# 1. Export PyTorch model to MLIR linalg
from torch_mlir.fx import export_and_import

mlir_module = export_and_import(
    model, *sample_inputs,
    output_type="linalg-on-tensors"
)

# 2. Save MLIR bytecode
with open("model.mlirbc", "wb") as f:
    mlir_module.operation.write_bytecode(f)

# 3. Compile and run via launcher
launcher = TorchMLIRHexagonLauncher(executor)
outputs = launcher.run_torch_mlir(
    "model.mlirbc", inputs, function_name="forward"
)
```

### CompGen's Equivalent

CompGen already captures PyTorch models via TorchDynamo:

```python
# CompGen's existing path (already works)
from compgen.capture.torch_export import capture_dynamo_partitions

artifact = capture_dynamo_partitions(model, sample_inputs)
# artifact.graphs = list of FX GraphModule partitions
# → convert to xDSL Payload IR via fx_to_xdsl()
# → process through target backend stages
```

Both Hexagon and CompGen converge to linalg. The Triton path would add an alternative frontend.

## Custom Dialect for Hardware Microops

Hexagon defines the **HexKL** (Hexagon Kernel Library) dialect to bridge from abstract linalg to concrete hardware operations:

```mlir
// Level 1: Standard linalg
linalg.matmul ins(%A: f16, %B: f16) outs(%C: f32)

// Level 2: After matmul-to-hexkl pass
hexkl.matmul %A, %B, %C {tileM = 32, tileN = 128, tileK = 32}
    : memref<2048x8192xf16>, memref<8192x1024xf16> -> memref<2048x1024xf32>

// Level 3: After hexkl-to-llvm pass (microop decomposition)
hexkl.micro_hmx_setup_acc_read_f16()          // Setup HMX accumulator
hexkl.micro_hmx_acc_clear_f16()               // Clear accumulator
hexkl.micro_hmx_copy_submatrix_to_f16(%tile)  // DMA tile into HMX
hexkl.micro_hmx_rm_to_ah_f16()                // Row matrix → activation history
hexkl.micro_hmx_mm_f16()                      // Matrix multiply (hardware op)
hexkl.micro_hmx_ah_to_rm_f16()                // Activation history → row matrix
hexkl.micro_hmx_copy_f16_to_f32_submatrix()   // Convert + copy out
```

### Generalizable Pattern

Any target can follow the same three-level decomposition:

```
Level 1: linalg ops (target-independent)
    ↓ target.lower_to_tiled pass
Level 2: target.tiled_ops (parameterized by tile config)
    ↓ target.decompose_to_microops pass  
Level 3: target.micro_ops (direct hardware instructions)
    ↓ target.emit_code
Level 4: ISA / assembly / binary
```

## Memory Hierarchy and DMA

Hexagon's VTCM (Tightly Coupled Memory) tiling pass transforms computation to use the fast on-chip scratchpad:

```mlir
// Before VTCM tiling: operates on DDR
linalg.matmul ins(%A_ddr: memref<2048x8192xf16>,
                   %B_ddr: memref<8192x1024xf16>)
              outs(%C_ddr: memref<2048x1024xf32>)

// After VTCM tiling: operates on VTCM with DMA
scf.for %i = 0 to 2048 step 64 {
  scf.for %j = 0 to 8192 step 4096 {
    scf.for %k = 0 to 1024 step 64 {
      // DMA: DDR → VTCM (address space 1)
      memref.copy %A_ddr_slice → %A_vtcm : memref<64x4096xf16, 1>
      memref.copy %B_ddr_slice → %B_vtcm : memref<4096x64xf16, 1>
      
      // Compute on VTCM (fast)
      linalg.generic ... ins(%A_vtcm, %B_vtcm) outs(%C_vtcm)
      
      // DMA: VTCM → DDR
      memref.copy %C_vtcm → %C_ddr_slice
    }
  }
}
```

**Address spaces**: 0 = DDR (slow, large), 1 = VTCM/scratchpad (fast, small).

DMA operations can be further lowered to async hardware DMA:
```mlir
// hexagonmem.copy → memref.dma_start / dma_wait
%tag = memref.dma_start %src, %dst, %num_elements, %stride_src, %stride_dst
memref.dma_wait %tag
```

## Options-Driven Pipeline Configuration

Hexagon uses a frozen dataclass with 60+ fields controlling every compilation aspect:

```python
@dataclass(frozen=True)
class HexagonOptions:
    # Compute optimization
    vectorize: int = 1
    vector_length: int = 32
    num_threads: int = 4
    fusion: bool = True
    
    # Memory optimization
    enableVTCMTiling: bool = True
    enableConvertToHexagonmem: bool = True
    enableHexagonmemCopyToDMA: bool = False
    
    # Hardware features
    enableHexKL: bool = False              # Matrix acceleration
    enableMultiThreading: bool = False
    enableVectorization: bool = True       # HVX
    
    # Pipeline control
    htp_kernel_gen: bool = False
    target_artifact: str = "o"             # "o", "llir", "so"
    lowerConstantsInSeparateSharedObjects: bool = False
    
    # Profiling
    enableLWP: bool = False
    iterations: int = 10
```

Options propagate as stringified dict to the C++ backend:
```python
options_map = {str(k): str(v) for k, v in asdict(options).items()}
translate_linalg_to_obj(mlir_module, options_map)
```

## Execution Model

### Triton Kernel Execution

```python
# Triton grid → threads
struct Closure {
    int num_programs_X, num_programs_Y, num_programs_Z;
    int pid_X, pid_Y, pid_Z;  // Per-thread program ID
    // ... packed inputs
};

ThreadManager<Closure> tm(num_threads);
for (pid_X, pid_Y, pid_Z in grid):
    closures[flat_idx] = {inputs, grid_dims, program_ids};
tm.exec(kernel_helper, closures);
```

### PyTorch Model Execution

```cpp
// C++ wrapper generated by launcher
extern "C" void forward(void *result, MemRefDescriptor *input_0, ...);

FuncResult *r = new FuncResult;
forward(r, &input_desc_0, &input_desc_1, ...);
// Extract output tensors from result struct
```

## How CompGen Should Adopt This

### Already Done
- PyTorch capture via TorchDynamo → FX graphs (equivalent to torch-MLIR path)
- Linalg IR via xDSL (equivalent to Hexagon's linalg-on-tensors)
- Pattern detection and kernel contracts
- Golden data generation for correctness testing
- Ukernel dialect with declaration/match/call ops

### To Build (Scaffold)
1. **`TargetBackendProtocol`** — Generalized interface matching Hexagon's BaseBackend
2. **`TargetOptions`** — Base options dataclass matching HexagonOptions pattern
3. **Triton semantic mapping** — Table mapping Triton ops to target semantics
4. **Per-target options** — NPU, GPU, CPU options extending TargetOptions

### To Build (Per Target)
5. **Tiled microop decomposition** — Like HexKL's matmul decomposition
6. **Memory tiling pass** — Like VTCM tiling
7. **DMA insertion** — Like hexagonmem.copy → dma_start/wait
8. **C++ backend** (optional) — pybind11 module for Triton plugin registration

## Reference Files (Hexagon-MLIR)

| Component | File | Key Class/Function |
|-----------|------|-------------------|
| Backend | `qcom_hexagon_backend/backend/compiler.py` | `HexagonBackend(BaseBackend)` |
| Driver | `qcom_hexagon_backend/backend/driver.py` | `HexagonDriver(DriverBase)` |
| Options | `qcom_hexagon_backend/backend/hexagon_options.py` | `HexagonOptions` dataclass |
| torch-MLIR | `qcom_hexagon_backend/backend/torch_mlir_hexagon_launcher.py` | `TorchMLIRHexagonLauncher` |
| Triton launcher | `qcom_hexagon_backend/backend/triton_hexagon_launcher.py` | `TritonHexagonLauncher` |
| C++ bindings | `qcom_hexagon_backend/python/triton_qcom_hexagon_backend.cc` | `init_triton_qcom_hexagon_backend()` |
| HexKL dialect | `qcom_hexagon_backend/test/Conversion/HexKLToLLVM/` | MLIR test files |
| VTCM tiling | `qcom_hexagon_backend/test/Conversion/LinalgToLLVM/vtcm_tiling_static.mlir` | Tiling patterns |
| DMA lowering | `qcom_hexagon_backend/test/Conversion/DMAToLLVM/dma_copy.mlir` | DMA conversion |
| Math library | `qcom_hexagon_backend/backend/hexagon_extern/hexagon/libdevice.py` | qhmath_hvx wrappers |

The reference codebase is at `/scratch2/agustin/CompGen/tmp/hexagon-mlir/`.
