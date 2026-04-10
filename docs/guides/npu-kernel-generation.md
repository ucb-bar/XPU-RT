# NPU Kernel Generation Guide

This guide covers the end-to-end workflow for generating NPU kernels for the SmolVLA model using CompGen's FP8 quantization pipeline and autocomp kernel search. The pipeline produces **deduplicated kernel contracts** — the minimum set of unique kernel specifications that cover all compute operations in the model — ready for autocomp to generate NPU-native implementations.

## Overview

The workflow has three stages:

1. **Graph Analysis** — Load SmolVLA, apply FP8 E4M3 quantization, capture the computation graph, and analyze every operator for NPU hardware mapping.
2. **Contract Generation** — Extract concrete shapes/dtypes from the graph, deduplicate identical signatures, and produce kernel contracts in YAML and autocomp-ready format.
3. **Kernel Generation** — Feed contracts into autocomp to generate NPU kernels, validate correctness, and plug results back into CompGen.

## Prerequisites

### Clone and set up

```bash
git clone <compgen-repo-url>
cd CompGen

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install CompGen + quantization dependencies
uv pip install -e ".[quantization]"

# Install model dependencies (lerobot, transformers, etc.)
uv pip install lerobot transformers diffusers torchvision datasets \
    accelerate einops huggingface_hub draccus imageio pyserial deepdiff num2words
```

### External dependencies

- **Understanding-PI0** — SmolVLA model wrapper. Expected at `/scratch2/agustin/merlin/third_party/Understanding-PI0`. If located elsewhere, set path in `python/compgen/models/robotics.py:_resolve_existing_root()`.
- **autocomp** — Kernel search engine. Located at `third_party/autocomp/`. Install with `uv pip install -e third_party/autocomp`.

## Step 1: Run the SmolVLA FP8 Pipeline

The pipeline loads the real SmolVLA model (450M parameters), applies per-tensor FP8 E4M3 quantization with power-of-two scaling (matching the NPU's E8M0 scale registers), captures the computation graph via TorchDynamo, and generates kernel contracts.

```bash
python -m compgen.quantization.smolvla_e2e --output-dir artifacts/smolvla_fp8_npu
```

This produces:

```
artifacts/smolvla_fp8_npu/
    golden_inputs.pt              # Reference inputs for correctness testing
    golden_outputs.pt             # Reference outputs (unquantized)
    graph_analysis.json           # Full op coverage report
    alignment_report.json         # NPU hardware alignment verification
    payload.mlir                  # Payload IR
    verification_report.json      # Combined verification results
    manifest.json                 # Bundle metadata
    kernel_contracts/             # Deduplicated kernel contracts (YAML)
        summary.yaml              # Index of all contracts
        000_matmul_fp8_*.yaml     # One file per unique kernel signature
        001_matmul_fp8_*.yaml
        ...
    autocomp_problems/            # Autocomp-ready problem packages
        matmul_fp8_*/             # One directory per contract
            reference.py          # PyTorch reference implementation
            test.py               # Test harness (inputs + correctness check)
            contract.yaml         # Full contract metadata
        index.yaml                # Contract index
```

### What the pipeline does

| Step | Action | Output |
|------|--------|--------|
| Load | Load SmolVLA from HuggingFace (`lerobot/smolvla_base`) | 450M param model |
| Quantize | Apply `FP8E4M3Po2Config` via torchAO `quantize_()` | 77 FP8 linears, 1 FP8 conv2d |
| Attention | Replace SDPA with `ExportableFP8Attention` | 44 FP8 attention modules |
| Alignment | Verify all scales are po2, softmax stays BF16 | `alignment_report.json` |
| Rewrite | Replace `FP8E4M3Po2Tensor` modules with `ExportableFP8Linear` | Export-ready modules |
| Capture | TorchDynamo partitioned capture | ~46 graph partitions |
| Analyze | Classify every op against NPU execution units | ~98.7% coverage |
| Contracts | Extract shapes, deduplicate, generate YAML | ~30-50 unique contracts |
| IR | Convert first partition to xDSL Payload IR | `payload.mlir` |

## Step 2: Understand the Kernel Contracts

### Contract format

Each YAML contract specifies exactly what a kernel must implement:

```yaml
contract_id: matmul_fp8_1x48x960x960
op_family: matmul
npu_unit: mxu
input_shapes:
  - [1, 48, 960]
  - [960, 960]
output_shapes:
  - [1, 48, 960]
input_dtypes:
  - fp8_e4m3
  - fp8_e4m3
output_dtype: bf16
accumulation_dtype: bf16
scale_format: e8m0
tile_shape: [32, 32]
instance_count: 48      # 48 graph ops share this exact signature
estimated_flops: 88473600
total_flops: 4246732800
priority: 4246732800
isa_mnemonic: vmatmul.mxu0
reference_pytorch: |
  import torch
  A = torch.randn([1, 48, 960], dtype=torch.bfloat16)
  B = torch.randn([960, 960], dtype=torch.bfloat16)
  C = torch.matmul(A, B)
```

### Key fields

| Field | Meaning |
|-------|---------|
| `op_family` | Operation type: `matmul`, `conv2d`, `softmax`, `elementwise_binary`, `elementwise_unary`, `reduction` |
| `npu_unit` | NPU execution unit: `mxu` (matrix, FP8 in/BF16 accum), `vpu` (vector, BF16), `xlu` (reduction, BF16) |
| `input_dtypes` | Per-operand dtype. MXU ops get `fp8_e4m3`, VPU/XLU get `bf16` |
| `output_dtype` | Always `bf16` (NPU accumulators are BF16) |
| `scale_format` | `e8m0` means the scale is a power-of-two stored in an 8-bit exponent register |
| `tile_shape` | NPU operates on 32x32 tiles for MXU, variable for VPU |
| `instance_count` | Number of graph ops this contract covers (deduplication metric) |
| `priority` | Total FLOPs across all instances. Generate high-priority kernels first |
| `reference_pytorch` | Runnable Python code producing the correct output |

### Deduplication

SmolVLA has ~280 MXU matmul ops in its graph, but most share identical shapes. After deduplication, the typical breakdown is:

- **~10-15 unique matmul shapes** covering all 280 MXU ops (Q/K/V projections, MLP layers, action head)
- **~10-15 unique elementwise shapes** covering all 933 VPU ops
- **~3-5 unique reduction/softmax shapes**

Implementing the top 10 contracts by priority covers the vast majority of compute FLOPs.

### Summary file

`kernel_contracts/summary.yaml` provides an overview:

```yaml
total_contracts: 42
total_ops_covered: 1200
contracts_by_unit:
  mxu: 12
  vpu: 25
  xlu: 5
contracts_by_family:
  matmul: 12
  elementwise_binary: 10
  elementwise_unary: 12
  softmax: 3
  reduction: 5
```

## Step 3: Generate Kernels with Autocomp

Each contract in `autocomp_problems/` is a self-contained package that autocomp can consume directly.

### Running autocomp on a single contract

```python
import sys
sys.path.insert(0, "third_party/autocomp")

from pathlib import Path
from autocomp.search.prob import Prob
from autocomp.search.search import BeamSearchStrategy
from autocomp.agents.cuda.cuda_agent import CudaLLMAgent
from autocomp.backend.kernelbench.kb_eval import KBEvalBackend
from autocomp.hw_config.cuda_config import CudaHardwareConfig

# Load the contract's problem
contract_dir = Path("artifacts/smolvla_fp8_npu/autocomp_problems/matmul_fp8_1x48x960x960")

prob = Prob(
    prob_type="compgen_npu",
    prob_id=0,
    test_file=contract_dir / "test.py",
    sol_file=contract_dir / "reference.py",
    context=(contract_dir / "contract.yaml").read_text(),
)

# Configure for target hardware (adapt for NPU)
hw_config = CudaHardwareConfig(gpu_name="...", pytorch_version="2.11.0", cuda_version="12.0")

# Run search
agent = CudaLLMAgent(hw_config=hw_config, models=["gemini-2.0-flash"])
eval_backend = KBEvalBackend()

strategy = BeamSearchStrategy(
    output_dir=Path("search_results/matmul_fp8_1x48x960x960"),
    eval_backend=eval_backend,
    agent=agent,
    orig_code=(contract_dir / "reference.py").read_text(),
    prob=prob,
    metric="latency",
)
strategy.optimize(iterations=10)
```

### NPU-specific autocomp adaptation

For NPU kernels (as opposed to CUDA), the autocomp pipeline needs:

1. **NPU hardware config** — A `HardwareConfig` subclass describing the NPU ISA, tile geometry, and memory model. The NPU model spec is at `third_party/npu_model/`.

2. **NPU eval backend** — An `EvalBackend` subclass that can simulate/run NPU kernels. The NPU simulator is at `third_party/npu_model/npu_model/`.

3. **NPU agent** — An agent that generates NPU ISA code instead of CUDA. The agent uses contract metadata (tile_shape, ISA mnemonic, scale_format) to guide code generation.

The tiled GEMM already working in autocomp is the building block. The kernel contracts specify exactly which GEMM shapes are needed and at what priority.

### Batch processing all contracts

```bash
# Process all MXU (matmul) contracts — these are the compute bottleneck
for dir in artifacts/smolvla_fp8_npu/autocomp_problems/matmul_*; do
    contract_id=$(basename "$dir")
    echo "Processing: $contract_id"
    python run_autocomp_search.py \
        --prob-dir "$dir" \
        --output-dir "search_results/$contract_id" \
        --iterations 10
done
```

## Step 4: Validate Kernels

After generating kernels, validate each against its contract's reference:

```python
from compgen.quantization.autocomp_bridge import (
    load_autocomp_result,
    validate_kernel_against_contract,
)
from compgen.quantization.kernel_contracts import NpuKernelContract
import yaml

# Load the contract
contract_data = yaml.safe_load(open("artifacts/smolvla_fp8_npu/kernel_contracts/000_matmul_fp8_1x48x960x960.yaml"))
contract = NpuKernelContract(**contract_data)

# Load autocomp result
result = load_autocomp_result(contract.contract_id, "search_results/matmul_fp8_1x48x960x960")
if result and result.kernel_code:
    validation = validate_kernel_against_contract(result.kernel_code, contract)
    print(f"Correct: {validation['correct']}, Max error: {validation['max_error']:.6f}")
```

## Step 5: Plug Kernels Back into CompGen

Generated and validated kernels are registered via CompGen's provider registry:

```python
from compgen.kernels.provider import KernelProvider, ProviderResult
from compgen.kernels.registry import ProviderRegistry

class NpuKernelProvider(KernelProvider):
    """Serves pre-generated NPU kernels from autocomp results."""

    def __init__(self, results_dir: str):
        self._results = {}
        # Load all validated kernels
        for result_dir in Path(results_dir).iterdir():
            if result_dir.is_dir():
                result = load_autocomp_result(result_dir.name, result_dir)
                if result and result.correct:
                    self._results[result_dir.name] = result

    def accepts_contract(self, contract) -> bool:
        return contract.op_family in self._results

    def search(self, contract, budget=1) -> ProviderResult:
        result = self._results.get(contract.op_family)
        if result:
            return ProviderResult(
                found=True,
                kernel_code=result.kernel_code,
                language=result.language,
                latency_us=result.latency_us,
                correct=result.correct,
            )
        return ProviderResult(found=False)

# Register the provider
registry = ProviderRegistry()
registry.register(NpuKernelProvider("search_results/"))
```

## NPU Hardware Reference

### Execution Units

| Unit | Precision | Tile | Latency | ISA Example |
|------|-----------|------|---------|-------------|
| MXU0 | FP8 in, BF16 accum | 32x32 | 32 cycles | `vmatmul.mxu0` |
| MXU1 | FP8 in, BF16 accum | 32x32 | 32 cycles | `vmatmul.mxu1` |
| VPU | BF16 | 16 lanes | 2-8 cycles | `vadd.bf16`, `vexp.bf16` |
| XLU | BF16 | - | 4 cycles | `vredsum.bf16`, `vtrpose.xlu` |

### Quantization Model

- **Weights**: FP8 E4M3 with per-tensor power-of-two scale (E8M0 format)
- **Activations**: Dynamically quantized to FP8, dequantized to BF16 for compute
- **Accumulation**: Always BF16
- **Softmax**: Always BF16 (never quantized)
- **Scale registers**: 32 x 8-bit exponent-only values

### Memory Hierarchy

| Level | Size | Bandwidth |
|-------|------|-----------|
| DRAM | 16 GiB | 2 B/cycle |
| VMEM | 1 MiB | 64 B/cycle |
| Tensor registers | 64 x 1024 B | Internal |
| MXU weight slots | 2 x 1024 B per MXU | Internal |
| MXU accumulators | 2 x 2048 B per MXU | Internal |

## Programmatic API

All pipeline steps are available programmatically:

```python
from compgen.quantization.pipeline import QuantizedModelPipeline
from compgen.capture.torchao_pipeline import QuantizationConfig
from compgen.quantization.kernel_contracts import (
    generate_npu_kernel_contracts,
    export_contracts_yaml,
    export_contracts_autocomp,
)

# Works with any nn.Module, not just SmolVLA
pipeline = QuantizedModelPipeline(
    model=your_model,
    sample_inputs=(your_inputs,),
    model_name="your_model_fp8",
    quant_config=QuantizationConfig(scheme="fp8_e4m3_po2"),
    output_dir="artifacts/your_model",
)
report = pipeline.run()

# Generate kernel contracts from the captured graphs
contracts = generate_npu_kernel_contracts(list(report.capture_artifact.graphs))
export_contracts_yaml(contracts, "artifacts/your_model/kernel_contracts")
export_contracts_autocomp(contracts, "artifacts/your_model/autocomp_problems")
```
