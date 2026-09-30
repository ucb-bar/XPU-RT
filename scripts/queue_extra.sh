#!/usr/bin/env bash
# The follow-up cells, one simulator at a time (GPU-room and simulator-count gated), all in the
# 2.4 m-people scene at density 0.30 unless stated: (1) the QoS-depth-1 ROS 2 deployment with its
# measured latency and cadence-only, every speed; (2) the calibrated gain (0.5 / replayed rate)
# for both main arms; (3) gate course C; (4) faster walking people (1.5 m/s).
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; X=$T/xpu_a_cpsat_hard.csv; V=$T/ros_vanilla445.csv; Q=$T/ros_vanilla4_q145.csv
ALL="1.8 1.4 1.0 1.6 1.2"
# the mechanism flights (commanded wrench -> rotor-model power) at the displayed speed in the displayed scene: 1.0 m/s, people 2.4 m
CRUISE=1.0 ER=$R/energy_runs_v3 OUTCSV=$R/flight_energy_v3.csv bash scripts/run_energy_v2.sh >> $R/energy_runs_v3.log 2>&1
D=$R/campaign_v2/display_same; ENERGY_CSV=$PWD/$R/flight_energy_v3.csv SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$D/xpu_s1005_figdata ROS_DIR=$D/ros_s1005_figdata DPI=300 \
  scripts/render_showdown_measured.sh $R/refined/warehouse_showdown_v2 > $R/refined/render_v2_energy_v3.log 2>&1
ENERGY_CSV=$PWD/$R/flight_energy_v3.csv .venv/bin/python scripts/hil_story_figure.py >> $R/refined/render_v2_energy_v3.log 2>&1
# first the tuned ROS 2 layouts on course A (phase 1 of scripts/sweep_tuned_ros.sh, shared CSV, so whichever slot gets there first flies it)
OUT=$R/campaign_percep ARMS="ros_vanilla4tm:$T/ros_vanilla4tm45.csv:242:0 ros_p3:$T/ros_p345.csv:55.8:0 ros_p3_q1:$T/ros_p3_q145.csv:31.1:0 xpu_cpsat:$X:56.8:0" SPEEDS="$ALL" bash scripts/campaign_percep.sh
OUT=$R/campaign_qos1    ARMS="ros_vanilla4_q1:$Q:42.0" SPEEDS="1.6 1.2" bash scripts/campaign_percep.sh
OUT=$R/campaign_qos1    ARMS="ros_vanilla4_q1:$Q:0"    SPEEDS="$ALL"   bash scripts/campaign_percep.sh
OUT=$R/campaign_tallcal GAIN=0.00521 ARMS="xpu_cpsat:$X:0"    SPEEDS="$ALL" bash scripts/campaign_percep.sh
OUT=$R/campaign_tallcal GAIN=0.01277 ARMS="ros_vanilla4:$V:0" SPEEDS="$ALL" bash scripts/campaign_percep.sh
OUT=$R/campaign_courseC COURSE=c ARMS="xpu_cpsat:$X:0 ros_vanilla4:$V:0" SPEEDS="$ALL" bash scripts/campaign_percep.sh
OUT=$R/campaign_walk    WALK=1.5 ARMS="xpu_cpsat:$X:0 ros_vanilla4:$V:0" SPEEDS="1.8 1.4 1.0" bash scripts/campaign_percep.sh
echo "QUEUE_EXTRA_DONE $(date +%H:%M:%S)"
