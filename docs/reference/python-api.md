# Python API

The current public Python API is the most useful scriptable surface in the repo.

## Exports

Top-level package exports:

- `compgen.device`
- `compgen.compile_model`
- `compgen.CompGenDevice`
- `compgen.CompiledModel`

## `compgen.device(...)`

```python
import compgen

device = compgen.device("examples/hardware_specs/gpu_simt_demo.yaml")
print(device.profile.name)
print(device.capabilities.target_class.value)
print(len(device.dialect_stack.stages))
```

What it does:

- loads a hardware spec YAML
- validates and classifies it
- extracts a target profile
- builds a target-specific dialect stack
- returns a `CompGenDevice`

## `compgen.compile_model(...)`

```python
import compgen
import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


device = compgen.device("examples/hardware_specs/gpu_simt_demo.yaml")
compiled = compgen.compile_model(TinyMLP(), device)
result = compiled(torch.randn(1, 64), num_iterations=5, warmup=1)
print(result.latency_median_us)
```

What it does today:

- captures the model
- converts it to Payload IR
- runs equality saturation
- executes the generated stage pipeline
- returns a `CompiledModel` wrapper that benchmarks through the local executor

## Important Current Limitation

`compgen.device()` currently expects a targetgen hardware spec, not the simpler profile YAMLs under `examples/target_profiles/`. The repo now ships `examples/hardware_specs/gpu_simt_demo.yaml` so the documented API path points at a real public example instead of a test-only fixture.
