#!/usr/bin/env bash
# Resume the calibrated-gain grid and the 20 Hz envelope cells once the campaigns have left the GPU.
set -u; cd "$(dirname "$0")/.."
while ! grep -q ROS_FAMILY2_DONE results/codesign_feedback/campaign_v2/ros_family.log 2>/dev/null; do sleep 600; done
while ! grep -q CAMPAIGN_V2_DONE results/codesign_feedback/campaign_v2/xpu_solvers.log 2>/dev/null; do sleep 600; done
bash scripts/gain_controlled_grid.sh > results/codesign_feedback/gain_controlled/grid_launch2.log 2>&1
bash scripts/envelope_20hz.sh > results/codesign_feedback/gain_controlled/envelope_20hz.log 2>&1
echo GRID_RESUMED_DONE
