# Inspect Artifacts

CompGen produces different artifacts depending on which path you use.

## Demo Bundle Artifacts

`scripts/e2e_demo.py` creates a bundle directory with a `manifest.json` at the root. Today, the bundle builder can emit:

| Artifact | File or directory | When it appears |
|---------|--------------------|-----------------|
| Payload IR | `payload.mlir` | Always |
| Execution plan | `execution_plan.yaml` | When an execution plan is provided |
| Golden inputs | `golden_inputs.pt` | When reference inputs are provided |
| Golden outputs | `golden_outputs.pt` | When reference outputs are provided |
| Generated kernels | `generated_kernels/` | When kernel files are supplied |
| Transform scripts | `transforms/` | When transform scripts are supplied |
| Manifest | `manifest.json` | Always |

`manifest.json` is the bundle index. It records:

- bundle format version
- target profile name
- model hash
- optimization objective
- artifact paths
- creation timestamp

## Target Generation Artifacts

When you use `compgen.device()` or `generate_target(...)`, CompGen writes target-generation artifacts into the chosen output directory. The currently guaranteed files are:

- `classification.json`
- `support_plan.json`
- `verification_manifest.json`

These tell you how CompGen classified the hardware, which stages it thinks are needed, and what verification surface the target can support.

## Planned CLI Artifacts

The CLI docs still describe a fuller artifact contract for `analyze`, `generate`, `verify`, and `promote`, but those commands are not yet implemented end to end. Treat those artifact layouts as planned interfaces, not current runnable outputs.
