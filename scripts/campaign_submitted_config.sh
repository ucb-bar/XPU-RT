#!/usr/bin/env bash
# ROS 2 out of the box (`vanilla_c50`), flown with real ROS 2 data.
#
# fig_hil_showdown was drawn with a modelled baseline. Its panel I describes what that baseline was:
# ROS 2 with a 50 Hz control timer whose YOLO runs serially on one hart and backs up, so control
# starves. That deployment is now measured on the K1 (45_vanilla_c50_*): the timer is configured at
# 50 Hz and achieves 20.3 Hz, camera->goal 265.9 ms, one hart busy.
#
# Both arms at the same 45 Hz camera, each replaying its own board measurements:
#   xpu_cpsat     XPU-RT CP-SAT, 100 Hz control, camera->control 56.8 ms, kernels over all 8 harts
#   ros_serial50  the baseline above
#
#   scripts/campaign_submitted_config.sh        env SPEEDS CELLS MAX_SIMS NEED_MB
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_submitted}"; T="$WT/results/codesign_feedback/ctrl_traces"
ARMLIST="${ARMLIST:-xpu_cpsat:$T/xpu_a_cpsat_hard.csv:56.8:0 ros_serial50:$T/ros_vanilla_c5045.csv:265.9:0}"
CELLS="${CELLS:-a:0.30}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  for a in $ARMLIST; do
    MAX_SIMS="${MAX_SIMS:-3}" NEED_MB="${NEED_MB:-10000}" ARMS="$a" SPEEDS="${SPEEDS:-1.4 1.0 1.8 1.2 1.6}" \
      COURSE=$co DENS=$de GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh" &
  done
  wait
done
echo "CAMPAIGN_SUBMITTED_DONE $(date +%H:%M:%S)"
