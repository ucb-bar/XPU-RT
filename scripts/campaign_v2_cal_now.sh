#!/usr/bin/env bash
# Calibrated gain for the two arms the figure shows, in the third simulator slot now.
set -u; cd "$(dirname "$0")/.."
T=results/codesign_feedback/ctrl_traces; SP="1.8 1.6 1.4 1.2 1.0"
GAIN=0.00520 ARMS="xpu_cpsat_hard:$T/xpu_a_cpsat_hard.csv" SPEEDS="$SP" bash scripts/campaign_v2b.sh
GAIN=0.01277 ARMS="ros_vanilla4:$T/ros_vanilla445.csv" SPEEDS="$SP" bash scripts/campaign_v2b.sh
GAIN=0.01523 ARMS="ros_vanilla4t:$T/ros_vanilla4t45.csv" SPEEDS="$SP" bash scripts/campaign_v2b.sh
GAIN=0.0055 ARMS="ros_vanilla4t:$T/ros_vanilla4t45.csv" SPEEDS="1.8 1.6 1.4 1.2 1.0" bash scripts/campaign_v2b.sh
echo CAL_NOW_DONE
