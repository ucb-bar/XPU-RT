#!/usr/bin/env bash
# Third simulator slot, sequential: the energy-v2 remainder, the same-scene display pair, the
# calibrated-gain remainder, the envelope grids, then the pipelined / QoS-1 ROS sweep cells.
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; E=$R/campaign_env
bash scripts/run_energy_v2.sh >> $R/energy_runs_v2/launch.log 2>&1
bash scripts/display_same_env.sh > $R/campaign_v2/display_same.log 2>&1
bash scripts/campaign_v2_cal_now.sh >> $R/campaign_v2/cal_now.log 2>&1
until grep -q GRID_FINAL_DONE $R/gain_controlled/resume_final.log 2>/dev/null; do sleep 300; done
SPEEDS="1.8 1.4 1.0" DENS="0.30" COURSES="a" bash scripts/env_sweep.sh 3 "ros_vanilla4x2:$T/ros_vanilla4x245.csv ros_vanilla4_q1:$T/ros_vanilla4_q145.csv"
