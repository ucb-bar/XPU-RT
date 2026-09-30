#!/usr/bin/env bash
# The strongest measured ROS 2 arrangement, at the headline camera rate.
#
# `p3` gives each network its own core set -- percep with a 4-hart YOLO pool pinned 0x1, nav 0x10,
# control 0x20 -- AND runs control on its own 100 Hz timer instead of chaining it to the goal. On
# the board it sustains 100.0 Hz of control at every camera rate from 5 to 120 Hz, so it is not
# rate-limited by the camera at all. That makes it the measured stand-in for the modelled
# per-network-pinning baseline, and a far harder baseline than the chained deployment.
#
# Flown here at the 30 Hz camera, at the same gain (0.00500) and the same four cruise speeds as
# the existing 30 Hz censuses, so its 48 rows drop straight alongside xpu_p30free's and
# ros_cp330's without re-flying either.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$WT/results/codesign_feedback/ctrl_traces"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_p3_30}"
mkdir -p "$OUT"
echo "=== $(date +%H:%M:%S) p3 @ 30 Hz camera: 100 Hz control, 30.4 ms camera->goal"
ARMLIST="${ARMLIST:-ros_p3_30:$T/ros_p330.csv:30.4:33.3:0.00500}" \
  SPEEDS="${SPEEDS:-1.6 1.4 1.2 1.0}" SEEDS="${SEEDS:-12}" MAX_SIMS="${MAX_SIMS:-3}" OUT="$OUT" \
  bash "$WT/scripts/campaign_rate30_gain.sh"
echo "CAMPAIGN_P3_30_DONE $(date +%H:%M:%S)"
