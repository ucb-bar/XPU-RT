# Custom Quantization: FP8 E4M3 for NPU via torchAO

This guide documents how CompGen integrates a custom FP8 E4M3 quantization scheme into the torchAO framework, targeting an NPU with power-of-two (po2) scaling. The SmolVLA VLA model is used as the reference deployment.

## Why Custom Quantization?

The NPU's Matrix Unit (MXU) operates on **FP8 E4M3 inputs** with **BF16 accumulation**, and its scale registers use **E8M0 format** — meaning scales must be exact powers of two. Standard quantization libraries (GPTQ, AWQ, torchAO's built-in FP8) use arbitrary floating-point scales, which require an FP multiply per tile on this hardware. Power-of-two scales reduce to a bit-shift in the exponent field, which is free.

### NPU execution model

| Unit | Input dtype | Compute dtype | Scale format | Operations |
|------|------------|---------------|-------------|------------|
| MXU (32×32) | FP8 E4M3 | BF16 | E8M0 (po2) | matmul, conv2d, bmm |
| VPU | BF16 | BF16 | — | add, mul, exp, silu, gelu |
| XLU | BF16 | BF16 | — | softmax, layernorm, reduce |

## Architecture

```
torchAO quantize_() API
    ↓  FP8E4M3Po2Config (custom AOBaseConfig)
    ↓  registers handler for nn.Linear
┌──────────────────────────────────────────┐
│  FP8E4M3Po2Tensor (TorchAOBaseTensor)   │
│  ├── _quantized_data: float8_e4m3fn      │
│  ├── _scale: float (always 2^N)          │
│  └── __torch_dispatch__: intercepts mm   │
└──────────────┬───────────────────────────┘
               ↓  rewrite_for_export()
┌──────────────────────────────────────────┐
│  ExportableFP8Linear / ExportableFP8Conv2d│
│  forward(): explicit quantize→matmul→deq │
│  (visible to torch.export / TorchDynamo) │
└──────────────┬───────────────────────────┘
               ↓  TorchDynamo capture
┌──────────────────────────────────────────┐
│  FX graphs with explicit FP8 ops        │
│  → graph_analyzer → kernel_contracts    │
│  → autocomp problem generation          │
└──────────────────────────────────────────┘
```

## Step 1: Define the Quantization Primitives

`python/compgen/quantization/fp8_ops.py` — the math layer, independent of any framework:

```python
FP8_E4M3_MAX = 448.0       # Largest finite float8_e4m3fn value
FP8_E4M3_MAX_PO2 = 256.0   # Largest power-of-two in range (2^8)

def fp8_po2_scale(tensor: torch.Tensor) -> float:
    """Compute power-of-two scale for a tensor.
    
    scale = 2^floor(log2(amax / 256))
    This fits exactly in the NPU's E8M0 scale registers.
    """
    amax = tensor.abs().max().float().item()
    if amax == 0:
        return 1.0
    raw = amax / FP8_E4M3_MAX_PO2
    return float(2 ** math.floor(math.log2(raw))) if raw >= 1.0 else 1.0

def quantize_fp8_e4m3_po2(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize to FP8 E4M3 with po2 scaling."""
    scale = fp8_po2_scale(x)
    x_scaled = (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return x_scaled.to(torch.float8_e4m3fn), scale
```

Key properties:
- Scale is **always** an exact power of two → no FP multiply on NPU
- Per-tensor granularity (not per-channel) — matches MXU tile semantics
- Numerically equivalent to pi0-quant's scheme

## Step 2: Create the torchAO Tensor Subclass

`python/compgen/quantization/fp8_tensor.py` — wraps quantized data for eager inference:

```python
class FP8E4M3Po2Tensor(TorchAOBaseTensor):
    """Tensor storing FP8 quantized data with po2 scale.
    
    Intercepts aten.mm, aten.linear, aten.addmm via __torch_dispatch__
    to perform dequant→compute→requant transparently.
    """
    
    def __init__(self, quantized_data, scale, source_dtype):
        self._quantized_data = quantized_data  # float8_e4m3fn
        self._scale = scale                     # float, always 2^N
        self._source_dtype = source_dtype       # original dtype
    
    @classmethod
    def from_float(cls, tensor: torch.Tensor) -> "FP8E4M3Po2Tensor":
        data, scale = quantize_fp8_e4m3_po2(tensor)
        return cls(data, scale, tensor.dtype)
    
    def dequantize(self) -> torch.Tensor:
        return self._quantized_data.float() * self._scale
```

The tensor subclass means **existing model code works unchanged** — `nn.Linear` with an FP8 weight tensor dispatches through `__torch_dispatch__` automatically.

## Step 3: Register with torchAO's `quantize_()` API

`python/compgen/quantization/fp8_config.py` — plugs into torchAO:

```python
@dataclass
class FP8E4M3Po2Config(AOBaseConfig):
    """torchAO configuration for FP8 E4M3 po2 quantization."""
    pass

@register_quantize_module_handler(FP8E4M3Po2Config)
def _fp8_e4m3_po2_transform(module: nn.Module, config: FP8E4M3Po2Config) -> nn.Module:
    """Handler called by torchao.quantize_() for each nn.Linear."""
    if not isinstance(module, nn.Linear):
        return module
    module.weight = nn.Parameter(
        FP8E4M3Po2Tensor.from_float(module.weight.data),
        requires_grad=False,
    )
    return module
```

Usage is then standard torchAO:
```python
from torchao import quantize_
quantize_(model, FP8E4M3Po2Config())
```

## Step 4: Handle Attention Separately

`python/compgen/quantization/attention.py` — softmax must stay BF16:

```python
class ExportableFP8Attention(nn.Module):
    """Explicit unfused attention with FP8 Q/K/V.
    
    Critical constraint: softmax is ALWAYS BF16. The NPU's VPU handles
    softmax in BF16; quantizing it to FP8 destroys accuracy.
    """
    
    def forward(self, query, key, value, ...):
        # Q, K, V → quantize to FP8
        q_fp8, q_scale = quantize_fp8_e4m3_po2(query)
        k_fp8, k_scale = quantize_fp8_e4m3_po2(key)
        v_fp8, v_scale = quantize_fp8_e4m3_po2(value)
        
        # Dequantize back to BF16 for matmul
        q = q_fp8.float() * q_scale
        k = k_fp8.float() * k_scale
        
        # Attention scores → softmax in BF16 (never quantized)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        attn = torch.softmax(attn, dim=-1)
        
        # Attention weights → FP8 for value matmul
        attn_fp8, attn_scale = quantize_fp8_e4m3_po2(attn)
        attn_deq = attn_fp8.float() * attn_scale
        
        v = v_fp8.float() * v_scale
        return torch.matmul(attn_deq, v)
```

## Step 5: Per-Component Recipe (SmolVLA-specific)

`python/compgen/quantization/smolvla_recipe.py` — different components get different treatment:

```python
class SmolVLAComponent(Enum):
    VISION = "vision"           # SigLIP ViT
    LANGUAGE = "language"       # Gemma 2.5B
    ACTION_EXPERT = "expert"    # Gemma 300M action expert
    ACTION_HEAD = "head"        # Thin MLPs

@dataclass
class SmolVLAQuantRecipe:
    component_configs: dict[SmolVLAComponent, FP8E4M3Po2Config]
    skip_modules: set[str]  # e.g., {"lm_head"}

def apply_smolvla_quantization(model, recipe):
    """Three-step quantization:
    1. quantize_() on nn.Linear (component-filtered)
    2. Patch Conv2d (vision patch embedding)
    3. Replace SDPA with ExportableFP8Attention
    """
```

## Step 6: Rewrite for Export

`python/compgen/quantization/export_wrappers.py` — before `torch.export`, replace tensor subclasses with explicit modules:

```python
def rewrite_for_export(model: nn.Module) -> nn.Module:
    """Replace FP8E4M3Po2Tensor-backed modules with ExportableFP8*.
    
    torch.export needs explicit ops (not dispatch magic).
    ExportableFP8Linear.forward() has visible quantize/dequantize calls
    that TorchDynamo can trace.
    """
```

After rewrite:
```python
# Before: nn.Linear with FP8E4M3Po2Tensor weight (__torch_dispatch__)
# After:  ExportableFP8Linear with explicit forward:
#   w_bf16 = self.weight_fp8.to(f32) * self.weight_scale
#   x_bf16 = x.to(bfloat16)
#   return F.linear(x_bf16, w_bf16, self.bias)
```

## Step 7: Verify NPU Alignment

`python/compgen/quantization/verify.py` — check all constraints:

```python
result = npu_alignment_check(model, allow_unquantized={"lm_head"})
# Checks:
# - All scales are power-of-two (fit in E8M0)
# - All FP8 tensors use float8_e4m3fn dtype
# - Softmax ops are BF16 (never quantized)
# - Expected module counts match
```

## Step 8: Capture and Analyze

After quantization + rewrite, the model flows into the standard CompGen pipeline:

```python
from compgen.quantization.pipeline import QuantizedModelPipeline

pipeline = QuantizedModelPipeline(model, sample_inputs)
report = pipeline.run()
# Steps: quantize → verify → rewrite → capture → decompose → analyze →
#         patterns → golden_data → kernel_contracts → payload_ir
```

The graph analyzer (`graph_analyzer.py`) classifies every captured op into NPU execution units:

```python
analysis = analyze_for_npu(fx_graphs)
# Result: ~98.7% op coverage
# - 280 MXU ops (matmul) → 23 unique kernel shapes
# - 933 VPU ops (elementwise) → 68 unique shapes
# - 1154 total ops → 8 reusable kernel patterns
```

## Step 9: Generate Kernel Contracts

`kernel_contracts.py` deduplicates ops and produces contracts for autocomp:

```yaml
# kernel_contracts/000_matmul_fp8_1x32x960x32x960.yaml
contract_id: matmul_fp8_001
op_family: matmul
npu_unit: MXU
input_shapes: [[1, 32, 960], [32, 960]]
input_dtypes: [float8_e4m3fn, float8_e4m3fn]
output_dtype: bfloat16
accumulation_dtype: bfloat16
scale_format: E8M0_PO2
tile_shape: [32, 32, 32]
instance_count: 12
estimated_flops: 58982400
```

## Full Pipeline Entry Point

```python
from compgen.quantization.smolvla_e2e import run_smolvla_npu_pipeline

report = run_smolvla_npu_pipeline(
    output_dir="artifacts/smolvla_fp8_npu",
    device="cpu",  # or "cuda"
)
```

Or through the capture pipeline:

```python
from compgen.capture.torchao_pipeline import apply_quantization, QuantizationConfig

apply_quantization(model, QuantizationConfig(scheme="fp8_e4m3_po2_npu"))
```

## Output Artifacts

```
artifacts/smolvla_fp8_npu/
├── golden_inputs.pt              # Reference inputs
├── golden_outputs.pt             # Reference outputs (BF16)
├── graph_analysis.json           # Op coverage: 98.7%
├── alignment_report.json         # NPU constraints verified
├── payload.mlir                  # Canonical IR
├── kernel_contracts/
│   ├── summary.yaml              # 8 patterns, 23 unique matmul shapes
│   └── 000_matmul_fp8_*.yaml     # Per-kernel contracts
└── autocomp_problems/
    ├── index.yaml
    └── matmul_fp8_*/             # Autocomp-ready problem packages
        ├── reference.py
        ├── test.py
        └── contract.yaml
```

## Key Design Decisions

1. **torchAO integration, not fork** — Uses `AOBaseConfig` + `register_quantize_module_handler` to plug into the standard `quantize_()` API. No torchAO source modifications.

2. **Two-phase approach** — Eager mode uses tensor subclass (`FP8E4M3Po2Tensor`) for quick prototyping. Export mode uses explicit modules (`ExportableFP8Linear`) for graph visibility.

3. **Po2 scales only** — No arbitrary-precision scales. The NPU's E8M0 scale registers require exact powers of two. This constraint is enforced at verification time.

4. **BF16 softmax invariant** — Quantizing softmax to FP8 destroys accuracy. This is enforced in `ExportableFP8Attention` and verified by `npu_alignment_check()`.

5. **Per-component recipe** — SmolVLA has 4 distinct components (vision, language, action expert, action head) that need different quantization strategies. The recipe system handles this.

6. **Kernel contracts as bridge** — The quantization pipeline produces kernel contracts that autocomp consumes. This decouples quantization decisions from kernel generation.
