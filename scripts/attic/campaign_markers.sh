#!/usr/bin/env bash
# Writes the completion markers the follow-on scripts wait for, once the three drivers are done.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/campaign_v2
while ! { grep -q DRIVER1_DONE $L/driver1.log && grep -q DRIVER3_DONE $L/driver3.log; } 2>/dev/null; do sleep 300; done
echo ALL_DRIVERS_DONE >> $L/all_drivers.log; echo CAMPAIGN_V2_DONE >> $L/xpu_solvers.log; echo ROS_FAMILY2_DONE >> $L/ros_family.log
echo MARKERS_WRITTEN
