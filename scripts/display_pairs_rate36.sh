#!/usr/bin/env bash
# Recorded display flights for the 36 Hz pairing: the (cruise, seed) cells where the census had XPU-RT
# completing and the two-instance ROS 2 graph crashing before gate 3. display_same_env.sh re-flies each
# seed with figure data dumped and keeps it only if the outcome reproduces on the scene it flies.
set -u
cd "$(dirname "$0")/.."
T=$PWD/results/codesign_feedback/ctrl_traces; OD=$PWD/results/codesign_feedback/campaign_rate3640/display36; mkdir -p "$OD"
for spec in "1.2:1005 1007" "1.4:1009" "1.8:1001 1003" "1.0:1006"; do
  cru=${spec%%:*}; seeds=${spec#*:}
  echo "=== $(date +%H:%M:%S) display cruise $cru seeds $seeds"
  CRUISE=$cru SEEDS="$seeds" DENS=0.30 GAIN=0.0055 XT=$T/xpu_w2pg36.csv RT=$T/ros_vanilla4x236.csv \
    XLAT=30.1 RLAT=32.2 XHOLD=27.8 RHOLD=27.8 ROS_GATES='[012]' RENDER=0 MAX_SIMS=3 NEED_MB=10000 \
    OUTDIR="$OD/c$cru" bash scripts/display_same_env.sh > "$OD/c$cru.log" 2>&1
  grep -E "xpu |ros |PAIR|no seed" "$OD/c$cru.log"
done
echo "DISPLAY36_DONE $(date +%H:%M:%S)"
