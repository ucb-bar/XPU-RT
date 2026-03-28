# CompGen

CompGen is a compiler generator for heterogeneous hardware targets. It uses LLMs as proposal engines inside a verification-first workflow that captures models, builds IR, generates target-specific artifacts, and packages the result for execution.

This docs site is user-facing first. It tells you what you can run today, what is still scaffolded, and where to start without sending you through internal roadmap or thesis material.

## Start Here

1. Read [Installation](getting-started/installation.md).
2. Run the [Quickstart](getting-started/quickstart.md).
3. Check [What Works Today](getting-started/what-works-today.md) before relying on a surface.
4. Use the guides and reference pages once you have the demo running.

## Current Reality

| Area | Current state |
|------|---------------|
| Bootstrap and package install | Available |
| `--help` / `--version` CLI surface | Available |
| End-to-end demo path | Runnable |
| Top-level Python API | Implemented and tested |
| Benchmark harness and suite adapters | Runnable with configured suite roots / commands |
| Pipeline subcommands (`analyze`, `generate`, `verify`, `run`, `promote`) | Declared, but still contract/stub surfaces |
| Agent planning, roadmap, thesis, and deep design docs | Moved to `tmp/agentic_documentation/` |

## Recommended Learning Path

- Use the [Quickstart](getting-started/quickstart.md) if you want to run something immediately.
- Use [Use the Demo](guides/use-the-demo.md) if you want to understand the current runnable pipeline.
- Use [Benchmark Suites](guides/use-benchmark-suites.md) if you want to run TorchBench, HuggingFace, TIMM, MLPerf, SOL-ExecBench, HeteroBench, or pack-backed integrations through the benchmark harness.
- Use [Bring Up a Target](guides/bring-up-a-target.md) if you want to work with hardware specs or target profiles.
- Use [CLI Reference](reference/cli.md) and [Python API](reference/python-api.md) when you need exact entrypoints.
