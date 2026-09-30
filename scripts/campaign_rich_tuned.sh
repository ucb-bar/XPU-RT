#!/usr/bin/env bash
# The loaded stack (yolo + nav + mlp + ffn_block 10 Hz + dronet 30 Hz) with the ROS arms that were
# never flown: the hand-pinned one that absorbs the extra networks, the single-executor one whose
# control starves at 18.8 Hz, and the multi-threaded executor. Against the XPU-RT arm for the same
# stack. Same cells, speeds and seeds as campaign_rich so every row pairs seed for seed.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_richtuned}"
T="$WT/results/codesign_feedback/ctrl_traces"
# name:trace:latency -- each arm at its own measured camera-to-goal latency
ARMLIST="${ARMLIST:-xpu_rich:$T/xpu_rich45.csv:41.2 ros_rp3:$T/ros_rp345.csv:57.4 ros_rv4x2tm:$T/ros_rvanilla4x2tm45.csv:93.7 ros_rv8tm:$T/ros_rvanilla8tm45.csv:244.1 ros_rspin:$T/ros_rspin45.csv:198.4 ros_rmulti:$T/ros_rmulti45.csv:247.3}"
for a in $ARMLIST; do
  echo "=== $(date +%H:%M:%S) $a"
  ARMS="$a" SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8}" COURSE=a DENS=0.30 GAIN=0.0055 \
    WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"
done
echo "CAMPAIGN_RICHTUNED_DONE $(date +%H:%M:%S)"
