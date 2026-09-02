# CompGen integration

CompGen is installed as an editable pip dependency in the `merlin-dev` conda env.
Its MCP server is wired to Claude Code via the project-scoped `.mcp.json` at the
repo root, so the ~47 `mcp__compgen__*` tools are available whenever Claude Code
is launched from `/scratch2/agustin/XPU-RT`.

CompGen continues to function standalone — nothing here is load-bearing for its
own test suite. Anything in this repo that targets CompGen (the Saturn OPU
ConvNet wrapper, the `xpurt_feedback.json` bridge) is an *additional* input
channel, never a required one.

## First-time setup

```sh
conda activate merlin-dev
cd /scratch2/agustin/CompGen
uv pip install -e '.[solve,kernels,benchmarks]'
```

Extras:
- `solve` — ortools / z3 / scipy, needed by CompGen's `cost_model_gate` CP-SAT path.
- `kernels` — forward-compat placeholder for kernel-search backends.
- `benchmarks` — matplotlib (already present in merlin-dev).

Not included: `iree` (collides with Merlin's IREE build), `llm` (Claude Code
drives the LLM externally via MCP).

## Verify

Both CLI entry points collide with bash builtins only if `compgen` is used
bare in a shell — the conda env script is still first on `PATH`. Inside a
merlin-dev shell:

```sh
/scratch2/agustin/miniforge3/envs/merlin-dev/bin/compgen --version
/scratch2/agustin/miniforge3/envs/merlin-dev/bin/compgen mcp print-config
```

The second command should print exactly the contents of `.mcp.json` at this
repo root.

## Use from Claude Code

Restart Claude Code with `cwd = /scratch2/agustin/XPU-RT`. The `.mcp.json`
at the repo root is auto-loaded. Its `command` is a repo-relative wrapper,
`./scripts/compgen-mcp.sh`, which resolves `compgen-mcp` in this order:

1. `$COMPGEN_MCP` — explicit absolute path override.
2. `compgen-mcp` already on `$PATH`.
3. `conda run -n $COMPGEN_ENV compgen-mcp` (default env name: `merlin-dev`).

This keeps the checked-in `.mcp.json` portable across machines: the
wrapper finds the right `compgen-mcp` regardless of where the CompGen env
lives. Override `$COMPGEN_ENV` if your install uses a different env name.

On first load, Claude Code prompts to approve the project's MCP servers
(the `/mcp` slash command also lets you enable/disable them). Once
approved, verify by asking Claude to call `mcp__compgen__session_summary`
— it should return a fresh session.

Demos and hardware specs ship inside the installed `compgen` wheel, so
MCP tools like `compile_embedded` can reference them by name —
`demo="saturn_opu_convnet"` and `spec_demo="saturn_opu"` — without any
reference to the CompGen source tree. `compgen.examples.list_demos()`
and `compgen.examples.list_specs()` enumerate what's available.

If the tools don't appear, run `compgen mcp doctor` inside `merlin-dev`.
If Claude reports "Failed to reconnect to compgen," the most common cause
is that the `command` path in `.mcp.json` no longer resolves — re-check
`ls /scratch2/agustin/miniforge3/envs/merlin-dev/bin/compgen-mcp`.

## Scope of the integration

- CompGen → XPU-RT: bundle directories under `compgen_output/` (produced
  per-iteration by CompGen's Spike / host path).
- XPU-RT → CompGen: a single `schedules/xpurt_feedback_{run_id}.json` per
  scheduling run, produced by `xpu-rt/compgen_bridge.py` and consumed by
  CompGen's `mcp__compgen__ingest_xpurt_feedback` MCP tool.

The feedback file is optional — XPU-RT emits it whenever the scheduler runs,
and CompGen ignores it until the MCP tool is explicitly called.
