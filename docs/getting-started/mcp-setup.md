# Wire up the MCP server

CompGen ships an MCP (Model Context Protocol) server called `compgen-mcp`. It
exposes every pipeline stage as a typed tool so Claude Code — or any
MCP-compatible client — can drive CompGen interactively.

## 1. Install

```bash
pip install compgen
```

Verify the entry points are on your `PATH`:

```bash
which compgen compgen-mcp
```

## 2. Register the server with Claude Code

Pick one of:

### Option A — let `compgen` do the merge

```bash
compgen mcp install          # edits ~/.claude.json with a timestamped backup
```

Add `--project` to write to the current directory's `.mcp.json` instead.
Re-runs are idempotent. Pass `--force` to replace an existing entry that points
somewhere else.

### Option B — paste the snippet yourself

```bash
compgen mcp print-config
```

Copy the output into your config of choice. The canonical snippet is:

```json
{
  "mcpServers": {
    "compgen": {
      "command": "compgen-mcp"
    }
  }
}
```

## 3. Verify

```bash
compgen mcp doctor
```

`doctor` imports the tool tree, checks the MCP SDK, lists every discovered
extension, and confirms `compgen-mcp` is on `PATH`. If every block comes back
clean, restart Claude Code and the `compgen` server shows up in the tool
picker.

## 4. List the available tools

```bash
compgen mcp tools
```

The same tool set is surfaced over MCP. Each tool declares a JSON schema and a
pipeline phase (`lifecycle`, `inspect`, `transform`, `job`).

## 5. Run the server standalone

Useful for local debugging or driving it from a non-Claude-Code client:

```bash
compgen mcp serve            # stdio transport, JSON-RPC
```

## Troubleshooting

- **"compgen-mcp: command not found"** — the package installed into a venv
  that's not on your `PATH`. Activate the venv or use the full path Claude
  Code reports in its MCP logs.
- **"failed to import compgen.mcp.tools"** — run `compgen mcp doctor`; the
  exception it prints is almost always a missing optional dep (e.g. `[solve]`
  or `[llm]`) for a tool whose handler module imports it at the top.
- **Extensions not appearing** — `compgen ext list` shows what discovery
  found; see [Extension Authoring](extension-authoring.md) for the discovery
  contract.
