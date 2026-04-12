# SmolVLA ONNX Workload

SmolVLA is a vision-language-action model exported to ONNX format and split into 9 independent submodels for efficient compilation and deployment on embedded systems.

## Model Overview

**Model**: SmolVLA (Small Vision-Language-Action)  
**Source**: [HuggingFace - lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)  
**Export Format**: ONNX (9 independent subgraphs)  
**Target Device**: SpacemiT X60 (RISC-V with RVV), BananaPi

## Submodels (9 Total)

| # | Submodel | Size | Description |
|---|----------|------|-------------|
| 1 | `action_in_projector` | ~100MB | Projects action inputs |
| 2 | `action_out_projector` | ~100MB | Projects action outputs |
| 3 | `smolvlm_expert_decode` | ~800MB | Expert decoder (autoregressive) |
| 4 | `smolvlm_expert_prefill` | ~1.3GB | Expert prefill (parallel processing) |
| 5 | `smolvlm_text` | ~200MB | Text encoder/decoder |
| 6 | `smolvlm_vision` | ~150MB | Vision encoder (processes images) |
| 7 | `state_projector` | ~50MB | State projection |
| 8 | `time_in_projector` | ~50MB | Temporal input projection |
| 9 | `time_out_projector` | ~50MB | Temporal output projection |

## Getting the Models

The ONNX models are hosted on HuggingFace:

**Repository**: https://huggingface.co/ainekko/smolvla_base_onnx

### Download via Git

```bash
# Option 1: Clone the HuggingFace repo
cd xpu-rt/pytorch_workload/smolvla/
git clone https://huggingface.co/ainekko/smolvla_base_onnx models

# Option 2: Sparse checkout (only MLIR files, skip weights)
git clone --depth 1 --filter=blob:none --sparse \
    https://huggingface.co/ainekko/smolvla_base_onnx models
cd models
git sparse-checkout set *.mlir
```

### Download via HuggingFace CLI

```bash
# Install huggingface-cli
pip install huggingface-hub

# Download all models
huggingface-cli download ainekko/smolvla_base_onnx \
    --local-dir xpu-rt/pytorch_workload/smolvla/models
```

### Manual Download

Download individual files from:
https://huggingface.co/ainekko/smolvla_base_onnx/tree/main

## Compilation

### Prerequisites

1. **Merlin Compiler**: Ensure the `merlin` submodule is up to date with SmolVLA configurations:
   ```bash
   cd merlin
   git pull origin main
   cd ..
   ```

2. **Configuration**: Verify `merlin/models/spacemit_x60.yaml` contains SmolVLA model configs.

### Compile All 9 Submodels

```bash
cd /path/to/XPU-RT

# Set source directory to downloaded models
SOURCE_DIR=xpu-rt/pytorch_workload/smolvla/models

# Compile all submodels
SUBMODEL_GLOB="all" \
SOURCE_DIR="${SOURCE_DIR}" \
    runtime/scripts/compile_sub_models.sh
```

### Compile with Benchmarks (for profiling)

```bash
SUBMODEL_GLOB="all" \
SOURCE_DIR=xpu-rt/pytorch_workload/smolvla/models \
    runtime/scripts/compile_sub_models.sh --build-benchmarks
```

**Output**: Compiled VMFB files and benchmark bundles in `gen/vmfb/smolVLA-new/`

## Profiling on Remote Device

### Prerequisites

1. **Remote Access**: SSH access to BananaPi (default: 10.44.86.251)
2. **IREE Runtime**: Installed on remote at `/home/spacemit-merlin-perf`
3. **Compiled Benchmarks**: Models compiled with `--build-benchmarks`

### Profile All 9 Models

```bash
cd /path/to/XPU-RT

# Profile all ONNX submodels on BananaPi
runtime/scripts/profile_smolvla_onnx.sh
```

### Profile Specific Models

```bash
# Profile only expert models
runtime/scripts/profile_smolvla_onnx.sh experts

# Profile specific submodels
runtime/scripts/profile_smolvla_onnx.sh smolvlm_text smolvlm_vision
```

**Output**: Profiling results in `gen/profile/scalar/spacemit_x60/<model>/`

## Configuration

### Model-Specific Compiler Flags

In `merlin/models/spacemit_x60.yaml`:

```yaml
models:
  smolvlm_expert_decode:
    - --iree-input-type=onnx
    - --iree-opt-data-tiling=false  # Fix i1/i8 mask tensor encoding
  smolvlm_expert_prefill:
    - --iree-input-type=onnx
    - --iree-opt-data-tiling=false
  # ... (7 more models)
```

**Key Fix**: `--iree-opt-data-tiling=false` prevents i1/i8 mask tensor encoding errors in ONNX `Where` operations.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOURCE_DIR` | `/scratch2/kris/smolvla_base_onnx` | ONNX model source directory |
| `OUT_BASE` | `gen/vmfb/smolVLA-new` | Compilation output directory |
| `SUBMODEL_GLOB` | `"smolvlm_expert_decode smolvlm_expert_prefill"` | Models to compile (use `"all"` for all 9) |
| `REMOTE` | `10.44.86.251` | BananaPi IP address |
| `BENCH_REPS` | `10` | Benchmark repetitions |

## Known Issues

### Issue 1: i1/i8 Mask Tensor Encoding

**Symptom**: Compilation fails with:
```
error: failed to legalize unresolved materialization from ('tensor<NxMxi8>') to ('tensor<NxMxi1>')
```

**Cause**: ONNX `Where` operations use boolean masks. IREE's data tiling optimization converts i1→i8, causing materialization failures.

**Fix**: Set `--iree-opt-data-tiling=false` in model configuration (already applied in `spacemit_x60.yaml`).

### Issue 2: Large Model Compilation Time

- **Expert models** (decode/prefill) take 5-8 minutes each
- **Other models** typically take 1-3 minutes
- This is expected due to model complexity

## Directory Structure

```
xpu-rt/pytorch_workload/smolvla/
├── README.md                    # This file
├── models/                      # Downloaded from HuggingFace
│   ├── action_in_projector.mlir
│   ├── action_out_projector.mlir
│   ├── smolvlm_expert_decode.mlir
│   ├── smolvlm_expert_prefill.mlir
│   ├── smolvlm_text.mlir
│   ├── smolvlm_vision.mlir
│   ├── state_projector.mlir
│   ├── time_in_projector.mlir
│   └── time_out_projector.mlir
└── .gitattributes              # Git LFS configuration (if needed)
```

## References

- **Model Source**: https://huggingface.co/lerobot/smolvla_base
- **ONNX Export**: https://huggingface.co/ainekko/smolvla_base_onnx
- **Compilation Script**: `runtime/scripts/compile_sub_models.sh`
- **Profiling Script**: `runtime/scripts/profile_smolvla_onnx.sh`
- **Configuration**: `merlin/models/spacemit_x60.yaml`

## Citation

If you use SmolVLA in your research:

```bibtex
@misc{smolvla,
  title={SmolVLA: Small Vision-Language-Action Model},
  author={LeRobot Team},
  year={2024},
  url={https://huggingface.co/lerobot/smolvla_base}
}
```

## Support

For issues related to:
- **Model export/ONNX**: Check https://huggingface.co/ainekko/smolvla_base_onnx
- **Compilation**: See `merlin/models/spacemit_x60.yaml` configuration
- **Profiling**: Check `PROFILING_GUIDE.md` in XPU-RT root
