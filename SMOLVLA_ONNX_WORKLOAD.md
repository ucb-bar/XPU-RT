# SmolVLA ONNX Workload - Complete Guide

## Overview

This workload adds support for compiling and profiling 9 ONNX submodels of SmolVLA (vision-language-action model) on SpacemiT X60 (RISC-V) targets.

**Repository**: https://github.com/ucb-bar/XPU-RT/tree/dev  
**Models**: https://huggingface.co/ainekko/smolvla_base_onnx  
**Target Device**: BananaPi with SpacemiT X60 (IP: 10.44.86.251)

### ONNX Submodels (9 Total)

| # | Model | Description |
|---|-------|-------------|
| 1 | action_in_projector | Action input projection |
| 2 | action_out_projector | Action output projection |
| 3 | smolvlm_expert_decode | Expert decoder (token-by-token) |
| 4 | smolvlm_expert_prefill | Expert prefill (batch processing) |
| 5 | smolvlm_text | Text encoder/decoder |
| 6 | smolvlm_vision | Vision encoder |
| 7 | state_projector | State projection |
| 8 | time_in_projector | Time input projection |
| 9 | time_out_projector | Time output projection |

---

## Quick Start

### 1. Download Models
```bash
cd /scratch2/kris/XPU-RT/xpu-rt/pytorch_workload/smolvla/models/
git clone https://huggingface.co/ainekko/smolvla_base_onnx .
```

### 2. Compile All Models
```bash
cd /scratch2/kris/merlin
SUBMODEL_GLOB="all" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh \
  --build-benchmarks
```

### 3. Profile on BananaPi
```bash
cd /scratch2/kris/XPU-RT
./runtime/scripts/profile_smolvla_onnx.sh
```

---

## The i1/i8 Encoding Fix

### Problem

ONNX expert models (`smolvlm_expert_decode` and `smolvlm_expert_prefill`) initially failed to compile with this error:

```
error: failed to legalize unresolved materialization from ('tensor<50x227xi8>') to ('tensor<50x227xi1>')
```

### Root Cause

ONNX `Where` operations use boolean (i1) masks. When `--iree-opt-data-tiling=true` is enabled, IREE converts i1→i8 for alignment optimization, then fails to materialize the cast back to i1.

**Technical Details**:
```mlir
%4 = iree_tensor_ext.dispatch.tensor.load ... -> tensor<50x227xi8>
%5 = builtin.unrealized_conversion_cast(%4) : (tensor<50x227xi8>) -> tensor<50x227xi1>
```

The cast `i8 → i1` fails during materialization because:
1. Load produces i8 (encoded representation)
2. Cast to i1 (logical representation) remains unrealized
3. MLIR verifier rejects the unrealized cast

### Solution

Added model-specific configurations in `merlin/models/spacemit_x60.yaml` to disable data tiling for all 9 ONNX submodels:

```yaml
models:
  smolvlm_expert_decode:
    - --iree-input-type=onnx
    - --iree-opt-data-tiling=false
  smolvlm_expert_prefill:
    - --iree-input-type=onnx
    - --iree-opt-data-tiling=false
  # ... (7 more models with same pattern)
```

**Why This Works**:
- Disables data tiling optimization that causes the encoding issue
- Prevents i1→i8 conversion that fails materialization
- All ONNX models use consistent configuration

---

## Repository Structure

```
XPU-RT/
├── xpu-rt/pytorch_workload/smolvla/    # New workload directory
│   ├── README.md                       # Basic workload documentation
│   ├── .gitattributes                  # Git LFS configuration
│   └── models/                         # ONNX models from HuggingFace
│       └── README.md                   # Download instructions
│
├── runtime/scripts/
│   ├── compile_sub_models.sh           # NEW: Compile 9 ONNX submodels
│   ├── profile_smolvla_onnx.sh         # NEW: Profile convenience wrapper
│   └── profile_remote.sh               # MODIFIED: Added SUBMODEL_FILTER
│
└── merlin/                             # Submodule
    └── models/spacemit_x60.yaml        # Contains 9 ONNX model configs
```

---

## Compilation

### Compile All Models
```bash
cd /scratch2/kris/merlin
SUBMODEL_GLOB="all" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh \
  --build-benchmarks
```

### Compile Specific Models
```bash
# Compile only expert models
SUBMODEL_GLOB="smolvlm_expert_decode smolvlm_expert_prefill" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh

# Compile only projectors
SUBMODEL_GLOB="action_in_projector action_out_projector state_projector" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh
```

### Compile for Different Targets/HW
```bash
# Target: spacemit_x60, Hardware: scalar (no RVV)
COMPILE_SUB_TARGETS="spacemit_x60" \
COMPILE_SUB_HWS="scalar" \
SUBMODEL_GLOB="all" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh

# Target: spacemit_x60, Hardware: RVV (with vector extension)
COMPILE_SUB_TARGETS="spacemit_x60" \
COMPILE_SUB_HWS="RVV" \
SUBMODEL_GLOB="all" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh
```

### Output Structure
```
gen/vmfb/smolVLA-new/<model>/<target>/<hw>/<basename>/
├── <model>.vmfb                    # Compiled model
├── <model>_benchmarks.zip          # Benchmark bundle (if --build-benchmarks)
└── <model>.mlir                    # Intermediate MLIR (if --dump-artifacts)
```

---

## Profiling

### Prerequisites
1. Compiled models with `--build-benchmarks`
2. SSH access to BananaPi (10.44.86.251)
3. IREE runtime installed on remote device

### Profile All 9 Models
```bash
cd /scratch2/kris/XPU-RT
./runtime/scripts/profile_smolvla_onnx.sh
```

### Profile Only Expert Models
```bash
./runtime/scripts/profile_smolvla_onnx.sh experts
```

### Profile Specific Models
```bash
./runtime/scripts/profile_smolvla_onnx.sh smolvlm_text smolvlm_vision state_projector
```

### Advanced: Direct Use of profile_remote.sh
```bash
# Profile with custom filter
SUBMODEL_FILTER="smolvlm_expert_decode smolvlm_expert_prefill" \
  ./runtime/scripts/profile_remote.sh

# Profile with custom remote settings
REMOTE=user@192.168.1.100 \
BENCH_REPS=20 \
DEVICE=local-task \
  ./runtime/scripts/profile_remote.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REMOTE` | `10.44.86.251` | BananaPi IP address |
| `REMOTE_IREE_ROOT` | `/home/spacemit-merlin-perf` | IREE install path on remote |
| `DEVICE` | `local-task` | IREE device to benchmark on |
| `BENCH_REPS` | `10` | Number of benchmark repetitions |
| `CONTINUE_ON_ERROR` | `1` | Continue if one model fails |
| `SUBMODEL_FILTER` | _(none)_ | Space-separated list of models to profile |
| `KEEP_REMOTE_TMP` | `0` | Keep remote temp files for debugging |

### Output Structure
```
gen/profile/<hw>/<target>/<model>/<basename>/
└── <full_name>/
    ├── topo_0/
    │   ├── results.csv       # Benchmark metrics
    │   └── benchmark.log     # Detailed logs
    ├── topo_1/
    └── ...
```

---

## Troubleshooting

### Compilation Issues

**Error: No ONNX models found**
```bash
# Solution: Ensure models are in the correct directory
ls -l /scratch2/kris/smolvla_base_onnx/*.mlir
```

**Error: i1/i8 materialization failure**
```bash
# Solution: Verify spacemit_x60.yaml has --iree-opt-data-tiling=false
grep -A2 "smolvlm_expert_decode:" /scratch2/kris/merlin/models/spacemit_x60.yaml
```

### Profiling Issues

**Error: No benchmark zips found**
```bash
# Solution: Recompile with --build-benchmarks
SUBMODEL_GLOB="all" \
  /scratch2/kris/XPU-RT/runtime/scripts/compile_sub_models.sh \
  --build-benchmarks
```

**Error: SSH connection failed**
```bash
# Test SSH access
ssh 10.44.86.251 echo "Connected"

# Set up SSH key if needed
ssh-copy-id 10.44.86.251
```

**Error: Remote benchmark tool not found**
```bash
# Solution: Stage local install (default behavior)
USE_STAGED_INSTALL=1 ./runtime/scripts/profile_remote.sh
```

---

## File Organization for GitHub Submission

### Phase 1: Merlin Repository
```bash
cd /scratch2/kris/merlin
git checkout -b add-smolvla-onnx-configs
git add models/spacemit_x60.yaml
git commit -m "Add SmolVLA ONNX model configurations

- 9 ONNX submodels: experts, text, vision, projectors
- Fix: --iree-opt-data-tiling=false for i1/i8 encoding
- Target: SpacemiT X60"
git push origin add-smolvla-onnx-configs
```

### Phase 2: XPU-RT Repository
```bash
cd /scratch2/kris/XPU-RT
git checkout -b add-smolvla-onnx-workload

# Stage new files
git add xpu-rt/pytorch_workload/smolvla/
git add runtime/scripts/compile_sub_models.sh
git add runtime/scripts/profile_smolvla_onnx.sh
git add runtime/scripts/profile_remote.sh
git add SMOLVLA_ONNX_WORKLOAD.md

# Update merlin submodule reference (after merlin PR is merged)
cd merlin && git checkout main && git pull && cd ..
git add merlin

# Commit
git commit -m "Add SmolVLA ONNX workload support

New workload: xpu-rt/pytorch_workload/smolvla/
- 9 ONNX submodels for vision-language-action model
- Models: https://huggingface.co/ainekko/smolvla_base_onnx

Scripts:
- compile_sub_models.sh: Compile 9 ONNX submodels
- profile_smolvla_onnx.sh: Profile on BananaPi
- profile_remote.sh: Enhanced with SUBMODEL_FILTER

Fix: --iree-opt-data-tiling=false for i1/i8 encoding
Testing: Verified on BananaPi SpacemiT X60"

# Push
git push origin add-smolvla-onnx-workload
```

### Create Pull Request
```bash
# Using GitHub CLI
gh pr create \
  --title "Add SmolVLA ONNX workload support" \
  --body "See SMOLVLA_ONNX_WORKLOAD.md for complete documentation" \
  --base dev \
  --head add-smolvla-onnx-workload
```

---

## Testing Verification

### Compilation Success Criteria
```bash
# All 9 models should compile successfully
find gen/vmfb/smolVLA-new -name "*.vmfb" | wc -l
# Expected: 9 (or 18 if both scalar and RVV)

# Verify benchmark bundles exist
find gen/vmfb/smolVLA-new -name "*_benchmarks.zip" | wc -l
# Expected: 9 (or 18 if both scalar and RVV)
```

### Profiling Success Criteria
```bash
# All 9 models should have profiling results
find gen/profile -name "results.csv" | wc -l
# Expected: 9+ (multiple topologies per model)

# Verify specific model results
cat gen/profile/scalar/spacemit_x60/smolvlm_expert_decode/*/topo_0/results.csv
```

---

## Performance Notes

### Model Sizes
- **smolvlm_expert_decode**: 798MB MLIR → 376MB VMFB
- **smolvlm_expert_prefill**: 1.3GB MLIR → 603MB VMFB
- **Other models**: 100-500MB MLIR each

### Compilation Time (Approximate)
- Small models (projectors): 2-5 minutes
- Medium models (text/vision): 10-20 minutes
- Large models (experts): 30-60 minutes

### Target Device
- **Platform**: BananaPi with SpacemiT X60
- **Architecture**: RISC-V64 (RV64GCV)
- **Vector Extension**: RVV 1.0 (VLEN=256)
- **Memory**: 16GB RAM

---

## Key Differences: PyTorch vs ONNX Export

| Aspect | PyTorch Export | ONNX Export |
|--------|---------------|-------------|
| **File Format** | Single monolithic MLIR (1.6GB) | 9 separate submodels (100MB-1.3GB each) |
| **BF16 Issue** | ❌ Has bf16 type mismatches | ✅ No bf16 issues |
| **i1 Issue** | ❌ Has i1/i8 encoding issues | ❌ Has i1/i8 encoding issues |
| **Fix Applied** | `--iree-opt-data-tiling=false` | `--iree-opt-data-tiling=false` |
| **Status** | ❌ Cannot compile | ✅ Compiles successfully |

---

## Lessons Learned

1. **ONNX export avoids bf16 issues**: Better than PyTorch export for this model
2. **i1/i8 bug is pervasive**: Affects both PyTorch and ONNX exports
3. **Model-specific config is key**: Generic config doesn't apply to all model names
4. **Data tiling breaks mask tensors**: Must be disabled for models with boolean masks
5. **Submodel approach is better**: Easier to debug, profile, and optimize individual components

---

## References

- **HuggingFace Models**: https://huggingface.co/ainekko/smolvla_base_onnx
- **Merlin Repository**: https://github.com/ucb-bar/merlin
- **XPU-RT Repository**: https://github.com/ucb-bar/XPU-RT
- **IREE Documentation**: https://iree.dev

---

## Support

For issues or questions:
- Check compilation logs in `gen/vmfb/smolVLA-new/<model>/`
- Check profiling logs in `gen/profile/<hw>/<target>/<model>/*/benchmark.log`
- Verify SSH access to BananaPi (10.44.86.251)
- Set `KEEP_REMOTE_TMP=1` to debug remote execution
- Review `merlin/models/spacemit_x60.yaml` for model configurations
