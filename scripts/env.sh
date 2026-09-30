# shellcheck shell=bash
# Machine-specific locations for the reproduction scripts. Override with environment
# variables or with scripts/env.local.sh (ignored by git; see env.local.sh.example).
REPO="${XPURT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$REPO/scripts/env.local.sh" ] && . "$REPO/scripts/env.local.sh"
SIM_TREE="${XPURT_SIM_TREE:-$REPO}"          # the tree holding sims/IsaacLab and sims/scripts
ISAAC_PY="${ISAAC_PY:-python}"               # the Isaac Sim / IsaacLab interpreter
HOST_PY="${HOST_PY:-$REPO/.venv/bin/python}" # the host interpreter (figures, solvers, verifiers)
RES="$REPO/results/codesign_feedback"
TRAIN_OUT="${XPURT_TRAIN_OUT:-$REPO/train_out}"  # where the training drivers write checkpoints
need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "env.sh: set $v" >&2; return 2; }; done; }
