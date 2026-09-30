#!/usr/bin/env bash
# Replicates of the multi-threaded-executor arm at the design rate (it had one).
set -u; cd "$(dirname "$0")/.."
while ! grep -q STRESS3_DONE results/codesign_feedback/xpurt_long/stress3.log 2>/dev/null; do sleep 30; done
for rep in 2 3; do echo "=== $(date +%H:%M:%S) smte r$rep"; RATES="45" scripts/ros_traced_matrix.sh smte $rep 2>&1 | grep -E "ROS_TRACED|error"; done
echo "SMTE_DONE"
