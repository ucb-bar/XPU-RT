#!/usr/bin/env bash
# Three simulator slots on the 24 GB card, no more: two environment-sweep drivers, and a third
# slot that finishes the energy-v2 and calibrated-gain remainders, lets the envelope grids run,
# then sweeps the pipelined and QoS-1 ROS deployments (course A, density 0.30, three speeds).
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; E=$R/campaign_env
nohup bash scripts/env_sweep.sh 1 "xpu_cpsat:$T/xpu_a_cpsat_hard.csv" > $E/driver1.log 2>&1 &
sleep 240
nohup bash scripts/env_sweep.sh 2 "ros_vanilla4:$T/ros_vanilla445.csv" > $E/driver2.log 2>&1 &
sleep 240
( bash scripts/run_energy_v2.sh >> $R/energy_runs_v2/launch.log 2>&1
  bash scripts/campaign_v2_cal_now.sh >> $R/campaign_v2/cal_now.log 2>&1
  until grep -q GRID_FINAL_DONE $R/gain_controlled/resume_final.log 2>/dev/null; do sleep 300; done
  SPEEDS="1.8 1.4 1.0" DENS="0.30" COURSES="a" bash scripts/env_sweep.sh 3 "ros_vanilla4x2:$T/ros_vanilla4x245.csv ros_vanilla4_q1:$T/ros_vanilla4_q145.csv"
) > $E/driver3.log 2>&1 &
wait
