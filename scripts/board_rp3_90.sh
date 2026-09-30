#!/usr/bin/env bash
# The hand-partitioned heavier-stack arm at the 90 Hz camera, for the second load of the layers figure.
set -u; cd "$(dirname "$0")/.."
while ! grep -q VANILLA2_DONE results/codesign_feedback/ros_traced/board_vanilla2.log 2>/dev/null; do sleep 60; done
for rep in 1 2; do echo "=== $(date +%H:%M:%S) rp3@90 r$rep"; RATES="90" scripts/ros_traced_matrix.sh rp3 $rep 2>&1 | grep -E "^===|error"; done
for rep in 1 2 3; do echo "=== $(date +%H:%M:%S) rvanilla4 r$rep"; RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh rvanilla4 $rep 2>&1 | grep -E "^===|error|fault"; done
echo RVANILLA4_DONE
echo RP3_90_DONE
