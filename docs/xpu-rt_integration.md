# XPU-RT integration

XPU-RT is installed as an editable pip dependency in the `merlin-dev` conda env.
Its MCP server is wired to Claude Code via the project-scoped `.mcp.json` at the
repo root, so the ~47 `mcp__xpu_rt__*` tools are available whenever Claude Code
is launched from `/scratch2/agustin/XPU-RT`.

XPU-RT continues to function standalone — nothing here is load-bearing for its
own test suite. Anything in this repo that targets XPU-RT (the Saturn OPU
ConvNet wrapper, the `xpurt_feedback.json` bridge) is an *additional* input
channel, never a required one.

## First-time setup

```sh
conda activate merlin-dev
cd /scratch2/agustin/XPU-RT
uv pip install -e '.[solve,kernels,benchmarks]'
```

Extras:
- `solve` — ortools / z3 / scipy, needed by XPU-RT's `cost_model_gate` CP-SAT path.
- `kernels` — forward-compat placeholder for kernel-search backends.
- `benchmarks` — matplotlib (already present in merlin-dev).

Not included: `iree` (collides with Merlin's IREE build), `llm` (Claude Code
drives the LLM externally via MCP).

## Verify

Both CLI entry points collide with bash builtins only if `xpu-rt` is used
bare in a shell — the conda env script is still first on `PATH`. Inside a
merlin-dev shell:

```sh
/scratch2/agustin/miniforge3/envs/merlin-dev/bin/xpu-rt --version
/scratch2/agustin/miniforge3/envs/merlin-dev/bin/xpu-rt mcp print-config
```

The second command should print exactly the contents of `.mcp.json` at this
repo root.

## Use from Claude Code

Restart Claude Code with `cwd = /scratch2/agustin/XPU-RT`. The `.mcp.json`
at the repo root is auto-loaded. Its `command` is a repo-relative wrapper,
`./scripts/xpu-rt-mcp.sh`, which resolves `xpu-rt-mcp` in this order:

1. `$XPU_RT_MCP` — explicit absolute path override.
2. `xpu-rt-mcp` already on `$PATH`.
3. `conda run -n $XPU_RT_ENV xpu-rt-mcp` (default env name: `merlin-dev`).

This keeps the checked-in `.mcp.json` portable across machines: the
wrapper finds the right `xpu-rt-mcp` regardless of where the XPU-RT env
lives. Override `$XPU_RT_ENV` if your install uses a different env name.

On first load, Claude Code prompts to approve the project's MCP servers
(the `/mcp` slash command also lets you enable/disable them). Once
approved, verify by asking Claude to call `mcp__xpu_rt__session_summary`
— it should return a fresh session.

Demos and hardware specs ship inside the installed `xpu-rt` wheel, so
MCP tools like `compile_embedded` can reference them by name —
`demo="saturn_opu_convnet"` and `spec_demo="saturn_opu"` — without any
reference to the XPU-RT source tree. `xpu_rt.examples.list_demos()`
and `xpu_rt.examples.list_specs()` enumerate what's available.

If the tools don't appear, run `xpu-rt mcp doctor` inside `merlin-dev`.
If Claude reports "Failed to reconnect to xpu-rt," the most common cause
is that the `command` path in `.mcp.json` no longer resolves — re-check
`ls /scratch2/agustin/miniforge3/envs/merlin-dev/bin/xpu-rt-mcp`.

## Scope of the integration

- XPU-RT → XPU-RT: bundle directories under `xpu-rt-output/` (produced
  per-iteration by XPU-RT's Spike / host path).
- XPU-RT → XPU-RT: a single `schedules/xpurt_feedback_{run_id}.json` per
  scheduling run, produced by `xpu-rt/xpu_rt_bridge.py` and consumed by
  XPU-RT's `mcp__xpu_rt__ingest_xpurt_feedback` MCP tool.

The feedback file is optional — XPU-RT emits it whenever the scheduler runs,
and XPU-RT ignores it until the MCP tool is explicitly called.
