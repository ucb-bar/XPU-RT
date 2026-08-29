#!/usr/bin/env bash
# Locate and exec `compgen-mcp` (the CompGen MCP server entry point).
#
# Claude Code inherits a PATH that usually doesn't include any conda env's
# bin directory, so a bare `compgen-mcp` reference in .mcp.json won't find
# the command on most machines. This wrapper resolves it without hard-coding
# a host path.
#
# Resolution order (first match wins):
#   1. $COMPGEN_MCP             — explicit absolute path override.
#   2. compgen-mcp on $PATH     — an activated env already has it.
#   3. `conda run -n $COMPGEN_ENV compgen-mcp`  (default env: merlin-dev)
#
# Override the conda env name with $COMPGEN_ENV if your install lives
# elsewhere.

set -euo pipefail

if [[ -n "${COMPGEN_MCP:-}" ]]; then
  exec "$COMPGEN_MCP" "$@"
fi

if command -v compgen-mcp >/dev/null 2>&1; then
  exec compgen-mcp "$@"
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "${COMPGEN_ENV:-merlin-dev}" compgen-mcp "$@"
fi

echo "compgen-mcp not found:" >&2
echo "  - install compgen (uv pip install compgen) in an env on \$PATH," >&2
echo "  - set \$COMPGEN_MCP to the absolute path of compgen-mcp, or" >&2
echo "  - make sure 'conda' is on \$PATH and \$COMPGEN_ENV points to the env" >&2
exit 127
