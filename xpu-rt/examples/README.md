# CompGen Examples

This directory now separates profile-style examples from hardware-spec examples used by the top-level Python API.

## Target Profiles

| Profile | File | Description |
|---------|------|-------------|
| CUDA A100 | `target_profiles/cuda_a100.yaml` | NVIDIA A100-SXM4-80GB single GPU |
| Trainium 1 | `target_profiles/trainium1.yaml` | AWS Trainium 1 (trn1.2xlarge) |
| Multi-device | `target_profiles/multi_device.yaml` | Heterogeneous CPU + NVIDIA GPU |

## Hardware Specs

| Spec | File | Description |
|------|------|-------------|
| GPU SIMT Demo | `hardware_specs/gpu_simt_demo.yaml` | Public targetgen-style hardware spec for `compgen.device(...)` |

## Models

| Model | File | Description |
|-------|------|-------------|
| Simple MLP | `models/simple_mlp.py` | Minimal 3-layer MLP for pipeline testing |

## Usage

```bash
# Inspect the CLI surface
uv run python -m compgen.cli --help

# Run the current end-to-end demo
uv run python scripts/e2e_demo.py
```

```bash
# Exercise the top-level Python API with a public hardware spec
uv run python - <<'PY'
import compgen

device = compgen.device("examples/hardware_specs/gpu_simt_demo.yaml")
print(device.profile.name)
print(device.capabilities.target_class.value)
PY
```
