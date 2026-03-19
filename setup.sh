#!/bin/bash
set -euo pipefail

# XPU-RT one-step setup: builds merlin compiler + target runtime.
#
# Prerequisites:
#   - conda (Miniconda or Mamba)
#   - git submodules initialized: git submodule update --init --recursive
#   - conda env created and activated:
#       conda env create -f merlin/env_linux.yml
#       conda activate merlin-dev
#       uv sync
#
# This script must be run from the XPU-RT root with the merlin-dev conda
# environment active.
#
# Usage:
#   bash setup.sh                   # full setup (toolchain + host + target)
#   bash setup.sh --skip-toolchain  # skip toolchain download (if already installed)
#
# Env overrides:
#   MERLIN_DIR=...       Path to merlin (default: <repo>/merlin)
#   TARGET_PROFILE=...   Merlin build profile for target (default: spacemit)
#   HOST_CONFIG=...      Host build config (default: release)
#   TARGET_CONFIG=...    Target build config (default: perf)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERLIN_DIR="${MERLIN_DIR:-${SCRIPT_DIR}/merlin}"

SKIP_TOOLCHAIN=0
TARGET_PROFILE="${TARGET_PROFILE:-spacemit}"
HOST_CONFIG="${HOST_CONFIG:-release}"
TARGET_CONFIG="${TARGET_CONFIG:-perf}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-toolchain) SKIP_TOOLCHAIN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "${MERLIN_DIR}/tools/merlin.py" ]]; then
  echo "Error: merlin not found at ${MERLIN_DIR}" >&2
  echo "  If using the submodule: git submodule update --init --recursive" >&2
  echo "  If using a dev checkout: export MERLIN_DIR=/path/to/merlin" >&2
  exit 1
fi

run_merlin() {
  # Run a merlin CLI command from within the merlin directory.
  (cd "${MERLIN_DIR}" && uv run tools/merlin.py "$@")
}

# 1) Install cross-compilation toolchain
if [[ "${SKIP_TOOLCHAIN}" != "1" ]]; then
  echo "=== Step 1: Install toolchain ==="
  run_merlin setup toolchain --toolchain-target "${TARGET_PROFILE}"
else
  echo "=== Step 1: Skipping toolchain (--skip-toolchain) ==="
fi

# 2) Build host compiler tools
echo "=== Step 2: Build host compiler tools ==="
run_merlin build --profile vanilla --config "${HOST_CONFIG}"

# 3) Build target runtime (includes xpu-rt plugin and standalone archive)
echo "=== Step 3: Build target runtime (${TARGET_PROFILE}) ==="
run_merlin build --profile "${TARGET_PROFILE}" --config "${TARGET_CONFIG}"

echo ""
echo "Setup complete. Next steps:"
echo "  runtime/scripts/compile_all_models.sh   # compile model VMFBs"
echo "  runtime/scripts/profile_remote.sh       # profile on target device"
