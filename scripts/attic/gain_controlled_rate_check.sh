#!/usr/bin/env bash
# Does the rate effect survive GAIN CALIBRATION?
#
# The 240-flight envelope holds moment_scale fixed at 0.0055, a gain calibrated for a 50 Hz
# closed loop. Control authority therefore varies with rate across that grid, so a difference
# between two rates there confounds "less frequent commands" with "wrong gain". This runs the
# SAME two operating points with the gain calibrated per cell (moment_scale = 0.5/eff_hz) and
# everything else identical to the envelope, so the two arms differ in exactly one thing.
#
# The two points are not chosen: they are where the coupled-chain schedules land, through the
# simulator's own command-refresh rule ceil(latency / control_dt):
#     XPU-RT           29.93 ms E2E -> 3 steps -> 33.33 Hz   (sched_latency 28)
#     static pinning   35.58 ms E2E -> 4 steps -> 25.00 Hz   (sched_latency 38)
#
# Fixed-gain arm, already measured (results/codesign_feedback/hil_ablation.csv):
#     25.00 Hz  1/60      33.33 Hz  13/60      +20.0 pts, Fisher p = 9.8e-04
#
# WHAT EACH OUTCOME MEANS, decided before the run so it cannot be read after the fact:
#   * gap SURVIVES calibration -> the rate effect is real, the flight half of the story holds,
#     and it can be stated without the fixed-gain caveat.
#   * gap VANISHES -> what the envelope shows is substantially a gain artifact. The flight
#     claim gets dropped to context and the paper leads on the scheduling result, which does
#     not depend on it. That is a good outcome to have found ourselves.
#
# Cells alternate 25/33 so an interrupted run still leaves a balanced comparison.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/gain_controlled}"
EPS="${EPS:-8}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
CSV="$OUT/gain_controlled.csv"
LOG="$OUT/RUN.log"; : > "$LOG"
cd "$CAN" || exit 1

fly() {  # $1=cruise $2=sched_lat_ms $3=moment_scale $4=tag
  echo "=== $(date +%H:%M:%S) $4 cruise=$1 lat=$2 moment=$3 ===" | tee -a "$LOG"
  timeout 2600 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale "$3" \
    --cruise_speed "$1" --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --sweep-csv "$CSV" 2>&1 | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error" | tee -a "$LOG"
}

# moment_scale = 0.5 / eff_hz  -- the calibrated law from scripts/hil_dense_grid.sh
for CRU in 1.0 1.2 1.4 1.6 1.8; do
  fly "$CRU" 38 0.02000 "cal_25Hz"      # 0.5/25.00
  fly "$CRU" 28 0.01500 "cal_33Hz"      # 0.5/33.33
done
echo "ALL_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
