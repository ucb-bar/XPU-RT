# CompGen

CompGen is an LLM-driven compiler generator for heterogeneous hardware
targets. It captures PyTorch programs, runs a staged xDSL/MLIR pipeline,
proposes transforms via LLM-backed invent-slots, verifies every proposal,
and packages the result into a deterministic deployment recipe.

## Start here →

- **New to CompGen?** [Getting Started → Quickstart](getting-started/quickstart.md)
- **How does it work?** [Concepts → What CompGen Is](concepts/what-is-compgen.md)
- **Running a workflow?** [User Guides → Use the Demo](guides/use-the-demo.md)
- **Understanding the design?** [Architecture → Compiler Generation](architecture/compiler-generation.md)
- **Looking up an API?** [Reference → CLI Reference](reference/cli.md)
- **Sending a PR?** [Contributing → Releasing](contributing/releasing.md)

## What works today

See [What Works Today](getting-started/what-works-today.md) for the honest
current state — runnable vs scaffolded vs stub per surface. Everything
listed in the nav above is code that exists; the "what works today" page
tracks which surfaces are end-to-end, which are contract-only, and where
the LLM currently delegates.
