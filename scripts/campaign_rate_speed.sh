#!/usr/bin/env bash
# The rate x speed census for the 3-network chain. At a 45 Hz camera the two-instance ROS 2 arm
# (all eight harts) keeps up and ties us; as the camera rate rises the pipelining stops fitting
# (37.0 -> 66-89 -> 252.8 ms) while the schedule holds (56.8 / 58.5 / 55.9). This fills the cells
# needed to say whether that timing collapse reaches the flights.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_rate90}"
T="$WT/results/codesign_feedback/ctrl_traces"
# name:trace:latency:hold -- both arms at the SAME camera rate and each at its own measured latency
ARMLIST="${ARMLIST:-xpu_a90:$T/xpu_a90_cpsat.csv:55.9:11.1 ros_x2_90:$T/ros_vanilla4x290.csv:252.8:0}"
CELLS="${CELLS:-a:0.20 a:0.40 b:0.30 b:0.20 b:0.40 c:0.30 c:0.40}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  for a in $ARMLIST; do
    echo "=== $(date +%H:%M:%S) course $co density $de  $a"
    ARMS="$a" SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8}" COURSE=$co DENS=$de GAIN=0.0055 \
      WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"
  done
done
echo "CAMPAIGN_RATE90_DONE $(date +%H:%M:%S)"
