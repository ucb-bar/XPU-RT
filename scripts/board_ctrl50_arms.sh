#!/usr/bin/env bash
# ROS 2 out of the box (`vanilla_c50`), measured: its control timer at 50 Hz.
#
# WHY. fig_hil_showdown draws ROS 2 at 50 Hz control on 6 statically pinned cores, crashing while
# XPU-RT completes -- and its panel I shows why: YOLO serial on one hart backs up to 112 ms, so the
# control loop is fast but acting on stale perception. That is a different mechanism from the
# two-instance arm at a slow camera (control starved, perception fresh), so reproducing the figure
# means measuring the arms that have it: a 50 Hz control timer over a backed-up perception stage.
#
# Four rungs, three replicates each, all at a 45 Hz camera (CTRL_HZ sets the timer, SUFFIX tags it):
#   vanilla_c50      serial YOLO, unpinned            -- the backed-up case
#   p3_c50           3 processes over 6 cores         -- the figure's own core count
#   vanilla8tm_c50   one instance, 8-hart pool
#   vanilla4x2tm_c50 two instances across both clusters -- the best-informed rung
#
#   scripts/board_ctrl50_arms.sh            env RATES=45 REPS=3
set -u
cd "$(dirname "$0")/.."
RATES="${RATES:-45}"; REPS="${REPS:-3}"
( exec 9>results/codesign_feedback/board.lock; flock 9
  for arm in vanilla p3 vanilla8tm vanilla4x2tm; do
    for r in $(seq 1 "$REPS"); do
      CTRL_HZ=50 SUFFIX=_c50 RATES="$RATES" scripts/ros_traced_matrix.sh "$arm" "$r"
    done
  done )
.venv/bin/python scripts/pull_ros_traced.py | tail -n 2
echo "CTRL50_ARMS_DONE $(date +%H:%M:%S)"
