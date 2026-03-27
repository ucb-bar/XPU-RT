# Troubleshooting

## `compgen` Is Not on My `PATH`

Use `uv run`:

```bash
uv run python -m compgen.cli --help
```

The docs intentionally avoid assuming an activated virtual environment.

## The CLI Command Exists but Does Not Do the Full Workflow

That is expected for most subcommands right now. Use the demo or Python API for runnable flows, and use the CLI docs as a contract/reference surface.

## `compgen.device()` Rejects My Example Target Profile

`compgen.device()` currently expects a targetgen hardware spec, not the simpler profile YAMLs under `examples/target_profiles/`.

Use:

```bash
examples/hardware_specs/gpu_simt_demo.yaml
```

if you want a public example that matches the current API.

## I Only Have CPU

That is fine for the public quickstart. The demo still runs and benchmarks on CPU. GPU-specific benchmark output only appears when CUDA is available.

## I Need the Old Design, Roadmap, or Thesis Documents

They moved to:

```text
tmp/agentic_documentation/
```

The public docs are intentionally narrower and user-facing.
