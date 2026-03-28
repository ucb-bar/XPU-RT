#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-gen-$(date +%Y%m%d-%H%M%S)}"
BASE_DIR="../compgen-worktrees"
mkdir -p "${BASE_DIR}"

git worktree add "${BASE_DIR}/${RUN_ID}" -b "${RUN_ID}"
echo "Created worktree at ${BASE_DIR}/${RUN_ID}"
