#!/usr/bin/env bash
# The out-of-the-box deployments whose perception node uses the model's 4-hart build.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q VANILLA_DONE results/codesign_feedback/ros_traced/board_vanilla.log 2>/dev/null; do sleep 30; done
for rep in 1 2 3; do for arm in vanilla4 vanilla4t; do say "$arm r$rep"; RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "^===|error|fault"; done; done
say "VANILLA2_DONE"
