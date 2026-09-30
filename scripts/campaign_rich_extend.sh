#!/usr/bin/env bash
# Extending the loaded-stack study out from the cell where ROS 2 fails in all 59 flights and
# XPU-RT completes 13: the same two arms over more densities and an unseen course, so the result
# is not one scene. Same speeds and seeds as campaign_rich so the base cell pairs with it.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_richext}"
T="$WT/results/codesign_feedback/ctrl_traces"
ARMLIST="${ARMLIST:-xpu_b5:$T/xpu_b5_cpsat.csv:57.6:11.1 ros_rv4:$T/ros_rvanilla445.csv:243.0:26.7}"
CELLS="${CELLS:-a:0.20 a:0.40 b:0.30 c:0.30}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  for a in $ARMLIST; do
    echo "=== $(date +%H:%M:%S) course $co density $de  $a"
    ARMS="$a" SPEEDS="${SPEEDS:-1.0 1.4 1.8}" COURSE=$co DENS=$de GAIN=0.0055 \
      WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"
  done
done
echo "CAMPAIGN_RICHEXT_DONE $(date +%H:%M:%S)"
