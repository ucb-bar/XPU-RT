# CompGen

CompGen is an LLM-driven compiler generator for heterogeneous hardware
targets. It captures PyTorch programs, runs a staged xDSL/MLIR pipeline,
proposes transforms via LLM-backed invent-slots, verifies every proposal,
and packages the result into a deterministic deployment recipe.

The primary way to drive CompGen is through Claude Code via its MCP server —
every pipeline stage is exposed as a typed tool that the LLM can call.

## Start here →

- **Install it** — [Installation](getting-started/installation.md)
- **Wire it into Claude Code** — [MCP Setup](getting-started/mcp-setup.md)
- **Run it end-to-end** — [Quickstart](getting-started/quickstart.md)
- **Extend it** — [Extension Authoring](getting-started/extension-authoring.md)
- **How does it work?** — [Architecture → Compiler Generation](architecture/compiler-generation.md)
- **Looking up an API?** — [Reference → CLI](reference/cli.md) · [Python API](reference/python-api.md) · [Extension Points](reference/extension-points.md)

## What works today

See [What Works Today](getting-started/what-works-today.md) for the honest
current state — runnable vs scaffolded vs stub per surface. Everything
listed in the nav above is code that exists; the "what works today" page
tracks which surfaces are end-to-end, which are contract-only, and where
the LLM currently delegates.
