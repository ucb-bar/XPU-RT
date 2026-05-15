# Architecture overview

CompGen is organised around a staged pipeline plus a target-generation
subsystem. This page is a map — for the full treatment see
[Architecture → Compiler Generation](../architecture/compiler-generation.md).

## Current runnable path

The demo and the top-level Python API exercise this shape today:

1. Capture a PyTorch model (`torch.export` + `torch.compile` diagnostics).
2. Convert the graph into Payload IR.
3. Analyse kernels and choose strategies (gap analysis, kernel contracts).
4. Run equality-saturation optimisation.
5. Plan execution (placement + scheduling).
6. Bundle artifacts and benchmark locally.

## Target-generation path

Creating a `CompGenDevice`:

1. Load a hardware spec YAML.
2. Validate it against the schema.
3. Extract a target profile.
4. Classify the hardware family (Triton-friendly, accel-native, ukernel-runtime, hybrid).
5. Generate a support plan.
6. Build a target-specific dialect stack.
7. Emit target-generation artifacts.

## Public surfaces

| Surface | Purpose |
|---------|---------|
| MCP server (`compgen-mcp`) | Drive every pipeline stage from Claude Code |
| CLI (`compgen ...`) | Scriptable command surface; see [CLI Reference](../reference/cli.md) |
| Python API | Script the current working flows; see [Python API](../reference/python-api.md) |
| Extension packs | User-authored providers / dialects / targets; see [Extension Authoring](../getting-started/extension-authoring.md) |
| Demo script | Run the most complete vertical slice |
| Target profiles + hardware specs | Describe deployment targets |

## Deeper reads

- [Architecture → Runtime Model](../architecture/runtime-model.md)
- [Architecture → Target Backend Model](../architecture/target-backend-model.md)
- [Architecture → Triton Integration](../architecture/triton-integration-spec.md)
- [Architecture → Extension Points](../architecture/extension-points.md)

Deeper design records, ADRs, and roadmap material live in
`tmp/agentic_documentation/`; the public docs stay focused on user
workflows.
