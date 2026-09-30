#!/usr/bin/env bash
# The envelope grids (calibrated cells, then the 20 Hz cells under both policies) once the
# campaign's last drivers have left the GPU.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/campaign_v2
while ! { grep -q CAL_NOW_DONE $L/cal_now.log && grep -q COURSEB_V2_DONE $L/courseB.log && grep -q ENERGY_V2_DONE results/codesign_feedback/energy_runs_v2/ENERGY.log; } 2>/dev/null; do sleep 600; done
bash scripts/gain_controlled_grid.sh > results/codesign_feedback/gain_controlled/grid_launch3.log 2>&1
bash scripts/envelope_20hz.sh > results/codesign_feedback/gain_controlled/envelope_20hz.log 2>&1
echo GRID_FINAL_DONE
