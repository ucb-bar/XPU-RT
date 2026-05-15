# What CompGen is

CompGen is a compiler *generator*, not a compiler. Given a PyTorch program
and a hardware profile, it produces a verified **deployment recipe** — a
set of graph/lowering transforms, kernel decisions, placement choices, and
runtime artifacts. The LLM in the loop is a bounded proposal engine;
deterministic infrastructure executes its proposals and verification
decides what ships.

## What matters for users

- You are evaluating a workflow for new-target bring-up, not installing a
  finished production compiler.
- The MCP server (`compgen-mcp`) exposes every pipeline stage as a typed
  tool Claude Code can drive interactively.
- The repo already contains runnable pieces: capture, IR conversion,
  target generation, planning, bundling, local benchmarking — see
  [What Works Today](../getting-started/what-works-today.md).

## What CompGen is *not*

- Not a replacement for PyTorch's frontend.
- Not a wholesale rebuild of IREE.
- Not a generic VM.
- Not a system that ships unverified LLM output.

## Further reading

- [Architecture → Compiler Generation](../architecture/compiler-generation.md)
  — the staged design end-to-end.
- [Architecture → Target Backend Model](../architecture/target-backend-model.md)
  — how target packages are structured.
- [Architecture → Extension Points](../architecture/extension-points.md)
  — the full protocol contracts for user extensions.
