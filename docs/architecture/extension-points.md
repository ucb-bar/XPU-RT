# Extension Points Guide

This guide explains how to extend CompGen with custom kernel generators, quantization methods, target backends, MLIR dialects, and runtime adapters.

## Architecture Principles

CompGen separates **scaffold** (core infrastructure) from **extension points** (user-extensible surfaces):

- **Scaffold**: IR stack, compilation stages, agent framework, solvers, verification, storage. These define the pipeline and protocols. Modify with care.
- **Extension points**: Kernel providers, quantization methods, target backends, dialects, runtime adapters. These are designed for plugging in new implementations without touching scaffold code.

Every extension point follows the same pattern:
1. A **protocol** (Python Protocol class or ABC) defines the interface
2. A **registry** discovers and manages implementations
3. A **template** (`_template.py`) shows how to implement the protocol
4. An **example** (existing implementation) demonstrates a working extension

## Extension Point 1: Kernel Generators

**Purpose**: Generate optimized kernel implementations for compute patterns (matmul, softmax, attention, etc.).

**Where**: `python/compgen/kernels/providers/`

**Protocol**: `KernelProvider` in `python/compgen/kernels/provider.py`

```python
class KernelProvider(Protocol):
    """Protocol for kernel generation providers."""
    
    def accepts_contract(self, contract: KernelContract) -> bool:
        """Return True if this provider can handle the given contract."""
        ...
    
    def search(self, contract: KernelContract, budget: int = 50) -> ProviderResult:
        """Search for an optimized kernel implementation."""
        ...
    
    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export learned optimization knowledge."""
        ...
```

**Existing implementations**:
- `providers/autocomp.py` — LLM-driven kernel search via Autocomp
- `providers/triton_templates.py` — Parameterized Triton kernels
- `providers/ukernel_bridge.py` — Microkernel registry bridge

**How to add a new provider**:

```python
# python/compgen/kernels/providers/my_provider.py
from compgen.kernels.provider import KernelContract, KernelProvider, ProviderResult

class MyKernelProvider:
    """My custom kernel generator."""
    
    def accepts_contract(self, contract: KernelContract) -> bool:
        return contract.op_family in ("matmul", "softmax")
    
    def search(self, contract: KernelContract, budget: int = 50) -> ProviderResult:
        # Generate kernel code
        kernel_code = self._generate(contract)
        return ProviderResult(found=True, kernel_code=kernel_code, correct=True)
    
    def export_knowledge(self):
        return []

# Register:
from compgen.kernels.registry import ProviderRegistry
registry = ProviderRegistry()
registry.register(MyKernelProvider())
```

## Extension Point 2: Quantization Methods

**Purpose**: Define custom precision reduction schemes (FP8, INT4, mixed-precision, etc.).

**Where**: `python/compgen/quantization/methods/`

**Protocol**: torchAO's `AOBaseConfig` + `@register_quantize_module_handler`

```python
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import register_quantize_module_handler

@dataclass
class MyQuantConfig(AOBaseConfig):
    """My custom quantization configuration."""
    my_param: int = 8

@register_quantize_module_handler(MyQuantConfig)
def _my_quant_transform(module, config):
    # Replace module weights with quantized versions
    ...
    return module
```

**Existing implementation**: `quantization/fp8_config.py` (FP8 E4M3 with po2 scaling)

**Integration**: Add scheme name to `capture/torchao_pipeline.py:apply_quantization()`:
```python
if config.scheme == "my_quant":
    from my_package import MyQuantConfig
    quantize_(model, MyQuantConfig(**config.extra_args))
    return model
```

## Extension Point 3: Target Backends

**Purpose**: Compile and execute for a specific hardware target.

**Where**: `python/compgen/targets/backends/`

**Protocol**: `TargetBackendProtocol` in `python/compgen/targets/backend.py`

```python
class TargetBackendProtocol(Protocol):
    def supports_target(self, target_name: str) -> bool: ...
    def get_options(self) -> TargetOptions: ...
    def get_compilation_stages(self) -> list[str]: ...
    def compile_stage(self, stage_name, ir_text, options) -> CompilationStageResult: ...
    def compile(self, ir_text, options) -> CompiledArtifact: ...
    def validate(self, artifact, golden_inputs, golden_output) -> bool: ...
```

**Options**: Extend `TargetOptions` in `python/compgen/targets/options.py`:
```python
@dataclass(frozen=True)
class MyTargetOptions(TargetOptions):
    target_name: str = "my_chip"
    my_tile_size: int = 64
    my_memory_kb: int = 256
```

**Reference**: See `docs/architecture/target-backend-model.md` for the Hexagon-inspired full architecture.

## Extension Point 4: MLIR Dialects

**Purpose**: Define hardware-specific operations (like Hexagon's HexKL for matrix acceleration).

**Where**: `python/compgen/extensions/dialects/`

**Protocol**: `DialectSpec` in `python/compgen/extensions/xdsl_generate.py`

```python
from compgen.extensions.xdsl_generate import DialectSpec, DialectOpSpec

my_dialect = DialectSpec(
    name="my_accel",
    ops=[
        DialectOpSpec(name="tile_matmul", operands=["lhs", "rhs"], results=["result"],
                      attrs={"tile_m": "int", "tile_n": "int"}),
        DialectOpSpec(name="dma_load", operands=["src"], results=["dst"],
                      attrs={"channel": "int"}),
    ],
)
# Auto-generates xDSL Python dialect code
```

**Reference**: Hexagon's HexKL dialect (`hexkl.matmul`, `hexkl.micro_hmx_*`) is documented in `docs/architecture/target-backend-model.md`.

## Extension Point 5: Runtime Adapters

**Purpose**: Execute compiled artifacts on different runtimes (local CPU, IREE, PJRT, device simulator, etc.).

**Where**: `python/compgen/runtime/adapters/`

**Existing adapters**:
- `local_executor.py` — Local CPU/GPU benchmarking
- `iree_adapter.py` — IREE runtime
- `pjrt_adapter.py` — PJRT (JAX/XLA runtime)

## Extension Point 6: Transform Templates

**Purpose**: Parameterized MLIR Transform Dialect scripts for common optimization patterns.

**Where**: `python/compgen/transforms/templates/`

**Existing templates**: Triton-style fused kernels (matmul+bias+gelu, softmax, layer_norm, etc.)

## Extension Point 7: Graph Passes

**Purpose**: FX-graph-level analysis and transformation passes.

**Where**: `python/compgen/passes/`

**Pattern**: Python functions that take `torch.fx.GraphModule` and return modified graph or annotations.

**Existing passes**: `graph_decompose.py` — IREE-inspired fusion detection, transpose folding, composite op raising.

## Discovering Extension Points Programmatically

Extension point packages are marked with `__extension_point__ = True`:

```python
import compgen.kernels.providers as kp
print(kp.__extension_point__)     # True
print(kp.__extension_type__)      # "kernel_provider"
print(kp.__extension_protocol__)  # "compgen.kernels.provider.KernelProvider"
```

List all extension points:
```python
import pkgutil, importlib
for importer, modname, ispkg in pkgutil.walk_packages(compgen.__path__, compgen.__name__ + "."):
    if ispkg:
        try:
            mod = importlib.import_module(modname)
            if getattr(mod, "__extension_point__", False):
                print(f"{modname}: {mod.__extension_type__}")
        except ImportError:
            pass
```
