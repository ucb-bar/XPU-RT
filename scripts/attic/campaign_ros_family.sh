#!/usr/bin/env bash
# The out-of-the-box ROS 2 deployments in flight: once their 20 s board runs are pulled, replay
# each arm's measured control cadence (first replicate at the 45 Hz camera) across cruise speeds.
set -u; cd "$(dirname "$0")/.."
while ! grep -q VANILLA2_DONE results/codesign_feedback/ros_traced/board_vanilla2.log 2>/dev/null; do sleep 120; done
.venv/bin/python scripts/pull_ros_traced.py 45_vanilla_r1 45_vanilla4_r1 45_vanilla4t_r1 > /dev/null 2>&1
ARMS=""
for arm in vanilla vanilla4 vanilla4t; do
  .venv/bin/python scripts/ctrl_trace_from_board.py results/codesign_feedback/ros_traced/45_${arm}_r1/ctrl_gaps.csv --out results/codesign_feedback/ctrl_traces/ros_${arm}45.csv --warmup-ms 3000 || exit 1
  ARMS="$ARMS ros_${arm}:results/codesign_feedback/ctrl_traces/ros_${arm}45.csv"
done
ARMS="$ARMS" SPEEDS="${SPEEDS:-0.8 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0}" bash scripts/campaign_v2.sh
echo ROS_FAMILY_DONE
