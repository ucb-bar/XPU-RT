#!/usr/bin/env bash
# The mechanism behind the crash: flights with the commanded wrench logged, so the body moment
# the controller actually asks for -- and the propulsive power a rotor model implies -- can be
# compared across runtimes.
#
# Every arm flies the same course at the SAME gain (0.0055), differing only in control cadence,
# and each cadence is a control-timer gap measured on the K1 (see scripts/campaign_showdown.sh
# for the provenance of each value). The arms are named by the cadence they produce so the
# figure can label them from the data:
#
#   xpu100    6.30 ms -> 100 Hz    XPU-RT
#   ros33    22.21 ms ->  33 Hz    ROS 2 default executor, 15 Hz camera
#   ros17    52.02 ms ->  17 Hz    ROS 2 default executor, 25 Hz camera
#
# Six seeds per arm. Both the full-flight totals (scripts/flight_energy_model.py) and the
# matched-window comparison (scripts/flight_energy_matched.py) are computed from the same dumps.
set -u
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; . "$WT/scripts/env.sh"
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W=$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt
ER=$RES/energy_runs; mkdir -p "$ER/tmp"; export TMPDIR="$ER/tmp"
LOG=$ER/ENERGY.log
SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005}"
cd "$CAN" || exit 1
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

run_cond(){ local cond=$1 lat=$2 mom=$3
  for s in $SEEDS; do
    if [ -f "$ER/${cond}_s${s}/figure_data.npz" ]; then say "have $cond seed $s"; continue; fi
    say "energy: $cond seed $s (lat=$lat mom=$mom)"
    timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
      --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
      --sched_latency_ms "$lat" --percep_hold_ms 0 --moment_scale "$mom" \
      --cruise_speed 1.4 --walk_speed 0.0 --episodes 1 --seed "$s" --max_steps 1800 \
      --dump_figure_data "$ER/${cond}_s${s}" >> "$ER/${cond}.log" 2>&1
  done
}
# Earlier three-seed arms at unmatched gains are moved aside to _unmatched/, not deleted.
for d in "$ER"/ros50_s* "$ER"/ros25_s*; do [ -d "$d" ] && mkdir -p "$ER/_unmatched" && mv "$d" "$ER/_unmatched/"; done
run_cond xpu100 6.30  0.0055
run_cond ros33  22.21 0.0055
run_cond ros17  52.02 0.0055

say "computing modeled propulsive energy from the logged wrench..."
$PY "$WT/scripts/flight_energy_model.py" --glob "$ER/*_s*/figure_data.npz" --out "$RES/flight_energy.csv" 2>&1 | tee -a "$LOG"
say "ENERGY EXPERIMENT DONE -> $RES/flight_energy.csv"
