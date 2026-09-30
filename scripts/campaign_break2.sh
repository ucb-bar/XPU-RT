#!/usr/bin/env bash
# Second stage of the breaking-point test, after campaign_break: the same 1.7 m scene at density 0.40 (where does
# XPU-RT itself break?), and the hand-pinned ROS 2 layout (56 ms) at density 0.30 as the honest comparator.
#   nohup bash scripts/campaign_break2.sh > results/codesign_feedback/campaign_break2.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback; T=$R/ctrl_traces
until grep -q CAMPAIGN_BREAK_DONE $R/campaign_break.log 2>/dev/null; do sleep 300; done
export WAREHOUSE_PERSON_H=1.7
OUT=$R/campaign_break DENS=0.40 ARMS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv:56.8:0 ros_vanilla4:$T/ros_vanilla445.csv:242:0" SPEEDS="1.2 1.6 1.0 1.4 2.0 0.8 1.8" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-150
OUT=$R/campaign_break ARMS="ros_p3:$T/ros_p345.csv:55.8:0" SPEEDS="1.2 1.6 1.0 1.4 2.0 0.8 1.8" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-150
echo CAMPAIGN_BREAK2_DONE
