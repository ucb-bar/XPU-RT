#!/usr/bin/env bash
# The rate at which the two-instance ROS 2 arm keeps every hart busy yet runs control below the cliff.
#
# WHY. In the default ROS 2 graph control is triggered by the navigation callback, which is triggered by
# perception, so the control cadence is the pipeline's throughput: min(camera rate, ~62 Hz) for two model
# instances across all eight harts (board, three replicates at 25/30/45/60/75/90/120 Hz). At 90 Hz the
# arm reaches that ~62 Hz ceiling -- on the flat part of the control-rate envelope -- so a timing collapse
# there (252.8 ms) does not have to reach the flights. At a 30 Hz camera it runs control at 30.0 Hz with
# all eight harts working, while XPU-RT's schedule runs control at 100 Hz from the same camera.
#
# Both arms: the same 30 Hz camera, each at its own measured board latency, and the navigation goal held
# for one camera period (33.3 ms) in both -- perception cannot refresh faster than the camera for either.
#
#   scripts/campaign_rate30.sh            env MAX_SIMS NEED_MB CELLS SPEEDS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_rate30}"
T="$WT/results/codesign_feedback/ctrl_traces"
ARMLIST="${ARMLIST:-xpu_a30:$T/xpu_a30_cpsat.csv:55.2:33.3 ros_x2_30:$T/ros_vanilla4x230.csv:31.4:33.3}"
CELLS="${CELLS:-a:0.30}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  for a in $ARMLIST; do
    MAX_SIMS="${MAX_SIMS:-5}" NEED_MB="${NEED_MB:-5000}" ARMS="$a" SPEEDS="${SPEEDS:-1.4 1.0 1.8 1.2 1.6}" \
      COURSE=$co DENS=$de GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh" &
  done
  wait
done
echo "CAMPAIGN_RATE30_DONE $(date +%H:%M:%S)"
