# Bring Up a Target

CompGen currently has two hardware-description surfaces that matter to users:

- Target profiles in `examples/target_profiles/` for profile-centric flows and the demo
- Hardware specs for the top-level target-generation and Python API path

## The Fastest Current Path

Use the public hardware-spec example:

```bash
uv run python - <<'PY'
import compgen

device = compgen.device("examples/hardware_specs/gpu_simt_demo.yaml")
print(device.profile.name)
print(device.capabilities.target_class.value)
print(len(device.dialect_stack.stages))
PY
```

This exercises the current top-level API and target generation path.

## Target Profiles

The repo ships example target profiles under `examples/target_profiles/`:

- `cuda_a100.yaml`
- `multi_device.yaml`
- `riscv_soc.yaml`
- `trainium1.yaml`

These are the right starting point if you want to understand the simpler profile schema consumed by lower-level modules and the demo.

## Hardware Specs

`compgen.device()` currently expects a richer targetgen hardware spec. The public example at `examples/hardware_specs/gpu_simt_demo.yaml` mirrors the style exercised in `tests/targetgen/exemplars/`.

That path does the following:

- load and validate the hardware spec
- extract a `TargetProfile`
- classify the target family
- generate a support plan
- build a target-specific dialect stack
- write generation artifacts such as `classification.json`, `support_plan.json`, and `verification_manifest.json`

## Recommended Workflow Today

1. Start from `examples/hardware_specs/gpu_simt_demo.yaml`.
2. Get `compgen.device()` working with your edited spec.
3. Inspect the generated artifacts in the output directory.
4. Move to `compgen.compile_model()` once your target description is stable.

## What Is Not Ready Yet

The `compgen scaffold-target` CLI command is documented, but it is still a stub. For now, use the Python API and targetgen modules directly if you need a real bring-up workflow.
