# Use the Demo

The demo is the current best walkthrough of the CompGen stack without asking you to wire subsystems together yourself.

## Run It

```bash
uv run python scripts/e2e_demo.py
```

## What It Exercises

`scripts/e2e_demo.py` runs a small `SimpleMLP` model through these stages:

1. Capture with `torch.export`
2. FX-to-xDSL Payload IR conversion
3. Target-profile loading from `examples/target_profiles/cuda_a100.yaml`
4. Kernel contract construction and strategy selection
5. Equality saturation
6. Execution planning
7. Transform verification
8. Bundle creation
9. Local benchmarking

## What to Look For

- The number of FX nodes captured
- The number of Payload IR ops and diagnostics
- The target profile name and device count
- Kernel strategy counts
- Equality-saturation statistics
- Execution-plan summary
- Bundle path and listed artifacts
- CPU benchmark numbers, plus GPU numbers when CUDA is available

## Why This Is the Recommended First Run

- It exercises real code instead of contract-only CLI stubs.
- It produces concrete artifacts you can inspect.
- It works as a narrow but honest vertical slice through the repo.

## After the Demo

- Read [Inspect Artifacts](inspect-artifacts.md) to understand the bundle.
- Read [Python API](../reference/python-api.md) if you want to automate the same kind of flow.
- Read [Bring Up a Target](bring-up-a-target.md) if you want to move from bundled examples to your own hardware descriptions.
