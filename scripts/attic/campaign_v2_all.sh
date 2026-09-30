#!/usr/bin/env bash
# The whole trace-driven campaign as three parallel single-arm drivers (the GPU holds three
# simulators), fast speeds first; the remaining arms follow in the same drivers.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/campaign_v2; SP="2.0 1.8 1.6 1.5 1.4 1.3 1.2 1.1 1.0 0.8"
( ARMS="xpu_cpsat_hard:results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv" SPEEDS="$SP" bash scripts/campaign_v2.sh
  ARMS="ros_vanilla:results/codesign_feedback/ctrl_traces/ros_vanilla45.csv" SPEEDS="1.4 1.2 1.0 0.8" bash scripts/campaign_v2.sh
  echo DRIVER1_DONE ) > $L/driver1.log 2>&1 &
( ARMS="ros_vanilla4t:results/codesign_feedback/ctrl_traces/ros_vanilla4t45.csv" SPEEDS="$SP" bash scripts/campaign_v2.sh
  echo DRIVER2_DONE ) > $L/driver2.log 2>&1 &
( ARMS="ros_vanilla4:results/codesign_feedback/ctrl_traces/ros_vanilla445.csv" SPEEDS="$SP" bash scripts/campaign_v2.sh
  ARMS="xpu_greedy:results/codesign_feedback/ctrl_traces/xpu_a_greedy.csv" SPEEDS="$SP" bash scripts/campaign_v2.sh
  echo DRIVER3_DONE ) > $L/driver3.log 2>&1 &
wait
echo CAMPAIGN_V2_DONE > $L/xpu_solvers.log; echo ROS_FAMILY2_DONE > $L/ros_family.log     # the markers the follow-on scripts wait for
echo ALL_DRIVERS_DONE
