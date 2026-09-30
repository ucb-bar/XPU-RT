#!/usr/bin/env bash
# Where freshness should pay: a person walking across the aisle. A stale or dropped goal means
# acting on where the person WAS, and the blind distance is closing speed x latency. The static
# scene puts both runtimes on the flat part of the latency curve; this asks whether a moving
# obstacle separates them. Same cells, speeds and seeds for every arm.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_walk}"
T="$WT/results/codesign_feedback/ctrl_traces"
# arm = name:trace:latency -- the best XPU-RT arm against the two ROS arms that tie it
ARMLIST="${ARMLIST:-xpu_shard:$T/xpu_shard45.csv:36.0 ros_q1:$T/ros_vanilla4_q145.csv:42.0 ros_x2tm:$T/ros_vanilla4x2tm45.csv:37.8}"
for a in $ARMLIST; do
  echo "=== $(date +%H:%M:%S) $a"
  ARMS="$a" SPEEDS="${SPEEDS:-1.0 1.4 1.8}" COURSE=a DENS=0.30 GAIN=0.0055 \
    WALK="${WALK:-1.5}" SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"
done
echo "CAMPAIGN_WALK_DONE $(date +%H:%M:%S)"
