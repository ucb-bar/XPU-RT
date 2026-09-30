#!/usr/bin/env bash
# The tuned ROS 2 layouts and the camera-rate arms, flown through the whole sweep like the others
# (one simulator, GPU-room and count gated; cells skipped once in the shared CSV). Every arm
# replays its board-measured control cadence and camera-to-control latency; the rate arms also
# refresh the navigation goal at the deployment's measured goal rate (33/s for every ROS layout
# above 45 Hz, every frame for XPU-RT). Order: course A at 0.30 first, then densities, then
# courses B and C, then faster people and the calibrated gain, then the ROS environment driver resumes.
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; export OUT=$PWD/$R/campaign_percep
# slot 2: after the ROS environment driver has finished (nothing is stopped; cells already flown are skipped)
D2="${WAIT_PID:-}"; while [ -n "$D2" ] && kill -0 "$D2" 2>/dev/null; do sleep 300; done
A45="ros_vanilla4tm:$T/ros_vanilla4tm45.csv:242:0 ros_p3:$T/ros_p345.csv:55.8:0 ros_p3_q1:$T/ros_p3_q145.csv:31.1:0 xpu_cpsat:$T/xpu_a_cpsat_hard.csv:56.8:0"
ARATE="xpu_cpsat_90:$T/xpu_a90_cpsat.csv:55.9:11.1 ros_p3_90:$T/ros_p390.csv:56.2:30.3 ros_vanilla4tm_90:$T/ros_vanilla4tm90.csv:136.5:30.3 xpu_cpsat_120:$T/xpu_a_cpsat_hard.csv:59.9:8.3 xpu_greedy_90:$T/xpu_a90_greedy.csv:947.5:0"
AHOLD="xpu_cpsat_h:$T/xpu_a_cpsat_hard.csv:56.8:22.2 ros_p3_h:$T/ros_p345.csv:55.8:30.3 ros_vanilla4tm_h:$T/ros_vanilla4tm45.csv:242:30.3"
ALL="1.8 1.4 1.0 1.6 1.2"
say(){ echo "=== $(date +%H:%M:%S) $*"; }
say "phase 1: course A 0.30, tuned layouts";     ARMS="$A45"   SPEEDS="$ALL" bash scripts/campaign_percep.sh
say "phase 2: course A 0.30, camera-rate arms";  ARMS="$ARATE" SPEEDS="$ALL" bash scripts/campaign_percep.sh
say "phase 2b: course A 0.30, goal-rate holds";  ARMS="$AHOLD" SPEEDS="$ALL" bash scripts/campaign_percep.sh
for dens in 0.40 0.20; do say "phase 3: course A $dens"; DENS=$dens ARMS="$A45 $ARATE" SPEEDS="$ALL" bash scripts/campaign_percep.sh; done
for course in b c; do for dens in 0.30 0.40 0.20; do say "phase 4: course $course $dens"; COURSE=$course DENS=$dens ARMS="$A45 $ARATE" SPEEDS="$ALL" bash scripts/campaign_percep.sh; done; done
say "phase 5: faster people";                    WALK=1.5 ARMS="$A45 $ARATE" SPEEDS="1.8 1.4 1.0" bash scripts/campaign_percep.sh
say "phase 5b: calibrated gain (0.5 / 100 Hz)";  GAIN=0.005 ARMS="$A45" SPEEDS="$ALL" bash scripts/campaign_percep.sh
say "SWEEP_TUNED_ROS_DONE"
