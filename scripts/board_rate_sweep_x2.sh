#!/usr/bin/env bash
# The two-instance ROS 2 arm across the camera sweep, and XPU-RT at the same cameras.
#
# The arm's control cadence is its pipeline throughput, min(camera, ~62 Hz), so the camera rate is
# what puts the baseline above or below the control-rate floor. This measures both runtimes at every
# rate the figure and its neighbours need, three replicates each.
#
#   scripts/board_rate_sweep_x2.sh                 env RATES="25 30 36 38 40 45 50 60 75 90 120"
set -u
cd "$(dirname "$0")/.."
RATES="${RATES:-25 30 36 38 40 45 50 60 75 90 120}"
( exec 9>results/codesign_feedback/board.lock; flock 9
  for r in 1 2 3; do RATES="$RATES" scripts/ros_traced_matrix.sh vanilla4x2 $r; done )
.venv/bin/python scripts/pull_ros_traced.py | tail -n 1
for hz in $RATES; do
  [ -f "schedules/fig_w2pg${hz}_greedy_clamped.json" ] || XPURT_UNIFORM_PACKED_WIDTH=1 scripts/xpu_greedy_shard_at_rate.sh "$hz"
done
echo "RATE_SWEEP_X2_DONE $(date +%H:%M:%S)"
