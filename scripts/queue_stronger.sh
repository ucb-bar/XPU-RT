#!/usr/bin/env bash
# A fourth queue, sharing the three simulator slots through the GPU gate (never more than three
# simulators; nothing running is touched). In order: (1) the calibrated gain (0.5 / replayed rate) for
# both main arms in the tall-people scene; (2) the heavier stack — XPU-RT's 90 Hz rich table (CP-SAT,
# greedy) against the six-process vanilla graph at 45 and 90 Hz cameras, cadence + latency + goal rate
# replayed, and cadence-only; (3) twelve more seeds on the displayed cell; (4) people crossing the aisle;
# (5) the camera-rate arms (shared cells with the tuned sweep; whichever queue gets there first flies them).
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; X=$T/xpu_a_cpsat_hard.csv; V=$T/ros_vanilla445.csv
ALL="1.8 1.4 1.0 1.6 1.2"
say(){ echo "=== $(date +%H:%M:%S) $*"; }
say "1: calibrated gain, tall scene"
OUT=$R/campaign_tallcal GAIN=0.00521 ARMS="xpu_cpsat:$X:0"    SPEEDS="1.2 1.0 1.4 1.6 1.8" bash scripts/campaign_percep2.sh
OUT=$R/campaign_tallcal GAIN=0.01277 ARMS="ros_vanilla4:$V:0" SPEEDS="1.2 1.0 1.4 1.6 1.8" bash scripts/campaign_percep2.sh
say "2: heavier stack (ffn_block 10 Hz + dronet 30 Hz added)"
RICH="xpu_b5_cpsat:$T/xpu_b5_cpsat.csv:57.6:11.1 ros_rvanilla4:$T/ros_rvanilla445.csv:243:26.7 ros_rvanilla4_90:$T/ros_rvanilla490.csv:137:26.1 xpu_b5_greedy:$T/xpu_b5_greedy.csv:149:11.1"
OUT=$R/campaign_rich ARMS="$RICH" SPEEDS="$ALL" bash scripts/campaign_percep2.sh
OUT=$R/campaign_rich ARMS="xpu_b5_cpsat:$T/xpu_b5_cpsat.csv:0 ros_rvanilla4:$T/ros_rvanilla445.csv:0" SPEEDS="$ALL" bash scripts/campaign_percep2.sh
say "3: seeds 1012-1023 on the displayed cell"
OUT=$R/campaign_seeds24 SEED0=1012 ARMS="xpu_cpsat:$X:0 ros_vanilla4:$V:0" SPEEDS="1.0 1.2" bash scripts/campaign_percep2.sh
say "4: people crossing the aisle"
OUT=$R/campaign_cross CROSS=1 ARMS="xpu_cpsat:$X:56.8 ros_vanilla4:$V:242" SPEEDS="1.0 1.2 1.4" bash scripts/campaign_percep2.sh
OUT=$R/campaign_cross CROSS=1 ARMS="xpu_cpsat:$X:0 ros_vanilla4:$V:0" SPEEDS="1.0 1.2 1.4" bash scripts/campaign_percep2.sh
say "5: camera-rate arms (shared with the tuned sweep)"
ARATE="xpu_cpsat_90:$T/xpu_a90_cpsat.csv:55.9:11.1 ros_p3_90:$T/ros_p390.csv:56.2:30.3 ros_vanilla4tm_90:$T/ros_vanilla4tm90.csv:136.5:30.3 xpu_cpsat_120:$T/xpu_a_cpsat_hard.csv:59.9:8.3"
OUT=$R/campaign_percep ARMS="$ARATE" SPEEDS="$ALL" bash scripts/campaign_percep2.sh
say "QUEUE_STRONGER_DONE"
