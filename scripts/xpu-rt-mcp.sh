#!/usr/bin/env bash
# Locate and exec `xpu-rt-mcp` (the XPU-RT MCP server entry point).
#
# Claude Code inherits a PATH that usually doesn't include any conda env's
# bin directory, so a bare `xpu-rt-mcp` reference in .mcp.json won't find
# the command on most machines. This wrapper resolves it without hard-coding
# a host path.
#
# Resolution order (first match wins):
#   1. $XPU_RT_MCP             — explicit absolute path override.
#   2. xpu-rt-mcp on $PATH     — an activated env already has it.
#   3. `conda run -n $XPU_RT_ENV xpu-rt-mcp`  (default env: merlin-dev)
#
# Override the conda env name with $XPU_RT_ENV if your install lives
# elsewhere.

set -euo pipefail

if [[ -n "${XPU_RT_MCP:-}" ]]; then
  exec "$XPU_RT_MCP" "$@"
fi

if command -v xpu-rt-mcp >/dev/null 2>&1; then
  exec xpu-rt-mcp "$@"
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "${XPU_RT_ENV:-merlin-dev}" xpu-rt-mcp "$@"
fi

echo "xpu-rt-mcp not found:" >&2
echo "  - install xpu_rt (uv pip install xpu_rt) in an env on \$PATH," >&2
echo "  - set \$XPU_RT_MCP to the absolute path of xpu-rt-mcp, or" >&2
echo "  - make sure 'conda' is on \$PATH and \$XPU_RT_ENV points to the env" >&2
exit 127
