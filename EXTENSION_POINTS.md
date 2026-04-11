# CompGen Extension Points

This document maps every extensible surface in the CompGen codebase. Scaffold code (core infrastructure) is marked `[S]`. Extension points where users and LLMs add implementations are marked `[E]`.

## Quick Reference

| Extension Point | Where to Add | Protocol | Example |
|----------------|-------------|----------|---------|
| **Kernel generator** | `kernels/providers/` | `KernelProvider` | `autocomp.py` |
| **Quantization method** | `quantization/methods/` | `AOBaseConfig` + handler | `fp8_e4m3_po2/` |
| **Target backend** | `targets/backends/` | `TargetBackendProtocol` | `npu/` |
| **Runtime adapter** | `runtime/adapters/` | Executor interface | `local.py`, `iree.py` |
| **MLIR dialect** | `extensions/dialects/` | `DialectSpec` | HexKL-style |
| **Transform template** | `transforms/templates/` | Transform Dialect | Triton fused template |
| **Graph pass** | `passes/` | `FX graph → FX graph` | `graph_decompose.py` |

Each extension directory has a `_template.py` file showing exactly how to add your implementation.

## Package Map

```
python/compgen/
│
│  ─── SCAFFOLD (core infrastructure) ───
│
├── ir/                    [S] Three-layer IR stack
│   ├── payload/           [S]   Canonical computational IR (xDSL/linalg)
│   ├── recipe/            [S]   LLM-facing control IR
│   ├── semantic/          [S]   Verification/trust layer
│   ├── accel/             [S]   Accelerator dialect
│   ├── ukernel/           [S]   Microkernel call boundary
│   ├── tile/              [S]   Tile dialect
│   ├── layout/            [S]   Layout dialect
│   └── agent/             [S]   Agent dialect
│
├── stages/                [S] Compilation stage framework
│   ├── base.py            [S]   Stage contract, plugin base
│   ├── registry.py        [S]   Pipeline orchestration
│   ├── encoding/          [S]   Encoding stage
│   ├── dispatch/          [S]   Dispatch/fusion stage
│   ├── layout/            [S]   Layout resolution stage
│   └── bundle/            [S]   Bundle/package stage
│
├── capture/               [S] Model capture (torch.export, TorchDynamo)
├── agent/                 [S] LLM agent framework (CompilerEnv, loops)
├── llm/                   [S] LLM client adapters (Gemini, OpenAI, Anthropic)
├── eqsat/                 [S] Equality saturation engine
├── solve/                 [S] Mathematical solvers (CP-SAT, MILP, SMT)
├── verify/                [S] Verification ladder (structural→functional→performance→formal)
├── semantic/              [S] Formal verification (SMT backends)
├── rewrite/               [S] Rewrite rule infrastructure
├── promotion/             [S] LLM→verified→deterministic pipeline
├── memory/                [S] Artifact storage (content-addressed)
├── knowledge/             [S] Optimization knowledge base
├── search/                [S] Search infrastructure (frontiers, replay)
├── packs/                 [S] Extension pack system
├── models/                [S] Model catalog
├── benchmarks/            [S] Benchmark framework
├── analysis/              [S] Analysis infrastructure
├── synthesis/             [S] Guard synthesis
├── targetgen/             [S] Hardware spec → compiler generator
├── api.py                 [S] Top-level API
├── cli.py                 [S] CLI
│
│  ─── EXTENSION POINTS (user/LLM extensible) ───
│
├── kernels/               [E] Kernel generation ecosystem
│   ├── provider.py        [S]   KernelProvider protocol (implement this)
│   ├── registry.py        [S]   Provider registry (auto-discovers providers)
│   ├── selector.py        [S]   Strategy selection (NATIVE/LIBRARY/UKERNEL/AUTOCOMP)
│   ├── contracts.py       [S]   KernelSpec, KernelSearchPlan
│   ├── providers/         [E] ← ADD YOUR KERNEL GENERATOR HERE
│   │   ├── autocomp.py    [E]   Autocomp LLM-driven search
│   │   ├── triton_templates.py [E] Triton parameterized kernels
│   │   ├── ukernel_bridge.py   [E] Ukernel registry bridge
│   │   └── _template.py   [E]   Template for new providers
│   ├── patterns/          [E]   Kernel pattern detection + catalog
│   └── golden/            [E]   Golden test data generation
│
├── quantization/          [E] Quantization methods
│   ├── pipeline.py        [S]   QuantizedModelPipeline (orchestration)
│   ├── graph_analyzer.py  [S]   FX graph op coverage analysis
│   ├── verify.py          [S]   NPU alignment checks
│   ├── methods/           [E] ← ADD YOUR QUANTIZATION METHOD HERE
│   │   ├── fp8_e4m3_po2/  [E]   FP8 E4M3 with power-of-two scaling
│   │   └── _template.py   [E]   Template for new methods
│   ├── fp8_ops.py         [E]   FP8 quantization math (current location)
│   ├── fp8_tensor.py      [E]   FP8 tensor subclass
│   ├── fp8_config.py      [E]   torchAO config
│   ├── attention.py       [E]   FP8 attention module
│   ├── smolvla_recipe.py  [E]   SmolVLA-specific recipe
│   └── npu_op_map.py      [E]   NPU operator classification
│
├── targets/               [E] Hardware target backends
│   ├── backend.py         [S]   TargetBackendProtocol (implement this)
│   ├── options.py         [S]   Base TargetOptions
│   ├── schema.py          [S]   TargetProfile, YAML loading
│   ├── capability.py      [S]   CapabilitySpec, target classification
│   ├── triton_mapping.py  [S]   Triton op → target semantics
│   ├── backends/          [E] ← ADD YOUR TARGET BACKEND HERE
│   │   ├── npu/           [E]   NPU backend (in progress)
│   │   └── _template.py   [E]   Template for new backends
│   └── validate.py        [S]   Profile validation
│
├── runtime/               [E] Execution backends
│   ├── planner.py         [S]   Execution plan generation
│   ├── local_executor.py  [S]   Local benchmarking
│   ├── adapters/          [E] ← ADD YOUR RUNTIME ADAPTER HERE
│   │   └── _template.py   [E]   Template for new adapters
│   ├── iree_adapter.py    [E]   IREE runtime adapter
│   └── pjrt_adapter.py    [E]   PJRT runtime adapter
│
├── extensions/            [E] Custom dialects & patches
│   ├── xdsl_generate.py   [S]   Dialect generation framework
│   ├── llvm_patchgen.py   [S]   LLVM patch generation
│   ├── dialects/          [E] ← ADD YOUR MLIR DIALECT HERE
│   │   └── _template.py   [E]   Template for new dialects
│   └── (generated files)  [E]   Auto-generated dialect code
│
├── transforms/            [E] Transform scripts
│   ├── apply.py           [S]   Transform application
│   ├── synthesize.py      [S]   Transform synthesis
│   ├── verify.py          [S]   Transform verification
│   └── templates/         [E] ← ADD YOUR TRANSFORM TEMPLATES HERE
│
└── passes/                [E] Graph decomposition passes
    └── graph_decompose.py [E] ← ADD YOUR PASSES HERE

contrib/                   [E] Community-contributed extensions
├── README.md                  How to contribute
├── providers/                 Community kernel generators
├── quantization/              Community quantization methods
├── targets/                   Community target backends
└── dialects/                  Community MLIR dialects
```

## How to Extend Each Point

### Adding a Kernel Generator

1. Copy `python/compgen/kernels/providers/_template.py`
2. Implement `KernelProvider` protocol (see `provider.py`)
3. Register via `ProviderRegistry.register()`
4. Your provider competes alongside Autocomp and Triton templates

### Adding a Quantization Method

1. Copy `python/compgen/quantization/methods/_template.py`
2. Implement `AOBaseConfig` subclass + register handler with torchAO
3. Add scheme name to `capture/torchao_pipeline.py:apply_quantization()`
4. Your method is now usable via `QuantizationConfig(scheme="your_method")`

### Adding a Target Backend

1. Copy `python/compgen/targets/backends/_template.py`
2. Implement `TargetBackendProtocol` (see `backend.py`)
3. Define `TargetOptions` subclass with hardware-specific knobs
4. See `docs/architecture/target-backend-model.md` for the full Hexagon-inspired pattern

### Adding an MLIR Dialect

1. Copy `python/compgen/extensions/dialects/_template.py`
2. Define `DialectSpec` with operations and attributes
3. Use `xdsl_generate.py` to auto-generate the xDSL Python dialect
4. See Hexagon's HexKL dialect as reference

### Adding a Runtime Adapter

1. Copy `python/compgen/runtime/adapters/_template.py`
2. Implement the executor interface
3. Register in the runtime configuration

See `docs/architecture/extension-points.md` for the full guide with code examples.
