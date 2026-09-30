#!/usr/bin/env bash
# The unseen-gate course for the v2 arms (XPU-RT CP-SAT, ROS 2 vanilla with the 4-hart YOLO on a
# timer), three speeds, after the course-A campaigns have finished with the GPU.
set -u; cd "$(dirname "$0")/.."
WT=$PWD; . "$WT/scripts/env.sh"
while ! grep -q ROS_FAMILY2_DONE results/codesign_feedback/campaign_v2/ros_family.log 2>/dev/null; do sleep 300; done
while ! grep -q CAMPAIGN_V2_DONE results/codesign_feedback/campaign_v2/xpu_solvers.log 2>/dev/null; do sleep 300; done
WAREHOUSE_COURSE=b OUT=$RES/campaign_v2_courseB ARMS="xpu_cpsat_hard:results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv ros_vanilla4t:results/codesign_feedback/ctrl_traces/ros_vanilla4t45.csv" SPEEDS="1.0 1.2 1.4" bash scripts/campaign_v2.sh
echo COURSEB_V2_DONE
