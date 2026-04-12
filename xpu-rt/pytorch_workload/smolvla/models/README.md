# SmolVLA ONNX Models

This directory should contain the 9 ONNX submodel MLIR files exported from SmolVLA.

## Download Models

The models are hosted on HuggingFace and must be downloaded before compilation.

**Repository**: https://huggingface.co/ainekko/smolvla_base_onnx

### Quick Download

```bash
# From XPU-RT root directory
cd xpu-rt/pytorch_workload/smolvla/models/

# Option 1: Clone entire repository
git clone https://huggingface.co/ainekko/smolvla_base_onnx .

# Option 2: Download via huggingface-cli
huggingface-cli download ainekko/smolvla_base_onnx --local-dir .

# Option 3: Sparse checkout (MLIR files only, no weights)
git clone --depth 1 --filter=blob:none --sparse \
    https://huggingface.co/ainekko/smolvla_base_onnx .
git sparse-checkout set *.mlir
```

### Expected Files (9 total)

After downloading, this directory should contain:

```
models/
├── action_in_projector.mlir      (~100 MB)
├── action_out_projector.mlir     (~100 MB)
├── smolvlm_expert_decode.mlir    (~800 MB)
├── smolvlm_expert_prefill.mlir   (~1.3 GB)
├── smolvlm_text.mlir             (~200 MB)
├── smolvlm_vision.mlir           (~150 MB)
├── state_projector.mlir          (~50 MB)
├── time_in_projector.mlir        (~50 MB)
└── time_out_projector.mlir       (~50 MB)
```

**Total Size**: ~2.8 GB

## Verify Download

```bash
# Count MLIR files
ls -1 *.mlir 2>/dev/null | wc -l
# Should output: 9

# Check file sizes
ls -lh *.mlir
```

## Why Not in Repository?

The ONNX model files are large (2.8GB total) and stored separately on HuggingFace to:
1. Keep the XPU-RT repository lightweight
2. Allow version control of models independently
3. Enable easy updates without bloating git history

## Alternative: Use Custom Models

If you have your own SmolVLA ONNX export, place the 9 MLIR files here with the same naming convention.

Ensure each model has a corresponding configuration in `merlin/models/spacemit_x60.yaml`:

```yaml
models:
  <model_name>:
    - --iree-input-type=onnx
    - --iree-opt-data-tiling=false
```
