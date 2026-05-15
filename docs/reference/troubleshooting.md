# Troubleshooting

## `xpu-rt` / `xpu-rt-mcp` not on `PATH`

Reinstall into the environment you're actually using:

```bash
pip install --upgrade xpu-rt
which xpu-rt xpu-rt-mcp
```

If you work from a source checkout, `uv run xpu-rt --help` resolves the
script from the project venv without needing it activated.

## `xpu-rt mcp doctor` fails to import tools

The error message names the missing module. It's almost always an
optional extra whose handler module imports a third-party SDK at the top
(solvers, LLM clients, etc.). Install the relevant extra:

```bash
pip install "xpu-rt[solve,llm]"
```

Then re-run `xpu-rt mcp doctor`.

## Extensions don't appear in `xpu-rt ext list`

- **Entry-point extensions**: make sure the extension package is
  `pip install -e`-ed into the *same* environment as XPU-RT. Check with
  `pip show <your-package>`.
- **User-space `.py` files**: default root is `~/.xpu_rt/extensions/`. Set
  `XPU_RT_EXTENSIONS_DIR` to point elsewhere. `_state.json` in that
  directory records which files have been loaded — delete it to force a
  reload. Set `XPU_RT_DISABLE_LOCAL_EXTENSIONS=1` to skip the loader
  entirely while debugging.

## Claude Code doesn't see the `xpu-rt` MCP server

1. `xpu-rt mcp doctor` — verifies the binary and tool tree locally.
2. `xpu-rt mcp print-config` — confirms the canonical snippet.
3. Open the target config (`~/.claude.json` for user-scoped,
   `./.mcp.json` for project-scoped) and confirm it contains the
   `mcpServers.xpu-rt` block.
4. Restart Claude Code after any edit. It only re-reads MCP config on
   process start.

## The CLI command exists but is partial

Pipeline subcommands (`init-target`, `analyze`, `generate`, `verify`,
`run`, `promote`, `scaffold-target`) implement the contract and some best-
effort stages, but are not yet a full end-to-end workflow. Use the demo
(`scripts/e2e_demo.py`) or the Python API (`xpu_rt.pipeline.compile_and_diff`)
for runnable flows. See the [CLI Reference](cli.md) status column.

## `xpu_rt.device()` rejects my target profile

`xpu_rt.device()` expects a targetgen hardware spec, not the simpler
profile YAMLs under `examples/target_profiles/`. Use:

```text
examples/hardware_specs/gpu_simt_demo.yaml
```

as a public example that matches the current API shape.

## CPU-only machine

Fine. The demo and most tests run on CPU. GPU-specific benchmark output
only appears when CUDA is available.

## Old design / roadmap / thesis documents

Moved to `tmp/agentic_documentation/`. The public docs are intentionally
narrower and user-facing.
