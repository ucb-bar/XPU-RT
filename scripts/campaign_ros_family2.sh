#!/usr/bin/env bash
# The out-of-the-box deployments in flight, the ones that can enter the course first.
set -u; cd "$(dirname "$0")/.."
ARMS="ros_vanilla4t:results/codesign_feedback/ctrl_traces/ros_vanilla4t45.csv ros_vanilla4:results/codesign_feedback/ctrl_traces/ros_vanilla445.csv ros_vanilla:results/codesign_feedback/ctrl_traces/ros_vanilla45.csv" SPEEDS="0.8 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0" bash scripts/campaign_v2.sh
echo ROS_FAMILY2_DONE
