#!/usr/bin/env bash
# The strongest ROS arrangement: two 4-wide YOLO instances across both clusters AND control on its
# 45 Hz) over the same course/density cells the XPU-RT arm already covers at 56.8 ms, so the two
# can be compared seed-for-seed. One cell per campaign_percep.sh call; that script skips cells
# already in the CSV and waits for GPU room, so this driver can be re-run.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_q1cells}"
TRACE="$WT/results/codesign_feedback/ctrl_traces/ros_vanilla4_q145.csv"
CELLS="${CELLS:-a:0.30 a:0.20 a:0.40 b:0.30 b:0.20 b:0.40 c:0.30 c:0.40}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  echo "=== $(date +%H:%M:%S) course $co density $de"
  ARMS="ros_vanilla4_q1:$TRACE:42.0" SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8}" \
    COURSE=$co DENS=$de GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" \
    bash "$WT/scripts/campaign_percep.sh"
done
echo "CAMPAIGN_X2CELLS_DONE $(date +%H:%M:%S)"
