#!/usr/bin/env bash
# A stronger panel A: the baseline should get some way into the course and still never pass gate 3.
# Searches the display script itself for a seed where XPU-RT completes and ROS 2 crashes after exactly
# ROS_GATES gates (default 2). One cruise speed per invocation so several can run at once.
set -u
cd "$(dirname "$0")/.."
CRU="${1:?cruise}"; WANT="${ROS_GATES:-2}"
T=$PWD/results/codesign_feedback/ctrl_traces; OD=$PWD/results/codesign_feedback/campaign_rate3640/display36
CRUISE=$CRU SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011}" DENS=0.30 GAIN=0.0055 \
  XT=$T/xpu_w2pg36.csv RT=$T/ros_vanilla4x236.csv XLAT=30.1 RLAT=32.2 XHOLD=27.8 RHOLD=27.8 ROS_GATES="$WANT" \
  RENDER=0 MAX_SIMS=3 NEED_MB=10000 OUTDIR="$OD/g${WANT}_c$CRU" bash scripts/display_same_env.sh > "$OD/g${WANT}_c$CRU.log" 2>&1
grep -E "outcome=|PAIR_SEED" "$OD/g${WANT}_c$CRU.log" | tail -n 4
echo "G${WANT}_SEARCH_DONE cruise $CRU $(date +%H:%M:%S)"
