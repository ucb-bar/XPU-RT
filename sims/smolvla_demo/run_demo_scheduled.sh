#!/usr/bin/env bash
# SmolVLA in IsaacLab Arena, driven at the latency measured on the QRB5165.
#
# Same env as run_demo.sh (gr1_microwave / gr1_pink / mustard_bottle); the
# difference is that an action chunk becomes available only after the profiled
# schedule's makespan has elapsed in SIM time, so the robot has to keep moving
# on stale actions the way it would on the board.
#
#   ./run_demo_scheduled.sh                     # board timing (default)
#   LATENCY_MODE=none ./run_demo_scheduled.sh   # free-inference baseline
#   EPISODES=3 MAX_STEPS=400 ./run_demo_scheduled.sh
set -euo pipefail
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate xpurt
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# IsaacLab puts its log dir at $TMPDIR/isaaclab/logs. On a shared host that
# collides with whoever ran Isaac first -- /tmp/isaaclab ends up owned by
# another user and the run dies on PermissionError before the sim starts.
export TMPDIR="${TMPDIR:-/scratch2/dima/tmp/xpurt-$(id -un)}"
mkdir -p "$TMPDIR"

cd "$(dirname "$0")"
REPO="$(cd ../.. && pwd)"

SCHEDULE="${SCHEDULE:-$REPO/schedules/scheduled_networks_smolvla_v3_unrolled10_qrb5165_greedy_profiled.json}"
LATENCY_MODE="${LATENCY_MODE:-schedule}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
HEADLESS="${HEADLESS:-true}"

exec python scheduled_rollout.py \
    --schedule "$SCHEDULE" \
    --latency-mode "$LATENCY_MODE" \
    --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" \
    --headless "$HEADLESS" \
    "$@"
