# Troubleshooting

## `compgen` / `compgen-mcp` not on `PATH`

Reinstall into the environment you're actually using:

```bash
pip install --upgrade compgen
which compgen compgen-mcp
```

If you work from a source checkout, `uv run compgen --help` resolves the
script from the project venv without needing it activated.

## `compgen mcp doctor` fails to import tools

The error message names the missing module. It's almost always an
optional extra whose handler module imports a third-party SDK at the top
(solvers, LLM clients, etc.). Install the relevant extra:

```bash
pip install "compgen[solve,llm]"
```

Then re-run `compgen mcp doctor`.

## Extensions don't appear in `compgen ext list`

- **Entry-point extensions**: make sure the extension package is
  `pip install -e`-ed into the *same* environment as CompGen. Check with
  `pip show <your-package>`.
- **User-space `.py` files**: default root is `~/.compgen/extensions/`. Set
  `COMPGEN_EXTENSIONS_DIR` to point elsewhere. `_state.json` in that
  directory records which files have been loaded — delete it to force a
  reload. Set `COMPGEN_DISABLE_LOCAL_EXTENSIONS=1` to skip the loader
  entirely while debugging.

## Claude Code doesn't see the `compgen` MCP server

1. `compgen mcp doctor` — verifies the binary and tool tree locally.
2. `compgen mcp print-config` — confirms the canonical snippet.
3. Open the target config (`~/.claude.json` for user-scoped,
   `./.mcp.json` for project-scoped) and confirm it contains the
   `mcpServers.compgen` block.
4. Restart Claude Code after any edit. It only re-reads MCP config on
   process start.

## The CLI command exists but is partial

Pipeline subcommands (`init-target`, `analyze`, `generate`, `verify`,
`run`, `promote`, `scaffold-target`) implement the contract and some best-
effort stages, but are not yet a full end-to-end workflow. Use the demo
(`scripts/e2e_demo.py`) or the Python API (`compgen.pipeline.compile_and_diff`)
for runnable flows. See the [CLI Reference](cli.md) status column.

## `compgen.device()` rejects my target profile

`compgen.device()` expects a targetgen hardware spec, not the simpler
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
