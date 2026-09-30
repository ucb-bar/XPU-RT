#!/usr/bin/env bash
# The mechanism behind the crash, second form: flights with the commanded wrench logged, every
# arm replaying the control-output cadence measured on the K1 for it (scripts/ctrl_trace_from_board.py),
# same gain, same course, at the cruise speed the campaign's selection rule picked. Six seeds
# per arm. Totals via scripts/flight_energy_model.py into flight_energy_v2.csv.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"; W=$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt
ER="${ER:-$RES/energy_runs_v2}"; OUTCSV="${OUTCSV:-$RES/flight_energy_v2.csv}"; mkdir -p "$ER/tmp"; export TMPDIR="$ER/tmp"; LOG=$ER/ENERGY.log
SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005}"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
while ! grep -q ROS_FAMILY2_DONE $RES/campaign_v2/ros_family.log 2>/dev/null; do sleep 300; done
while ! grep -q CAMPAIGN_V2_DONE $RES/campaign_v2/xpu_solvers.log 2>/dev/null; do sleep 300; done
# the displayed cell decides the cruise speed and which ROS deployment is the figure's baseline
CHOICE=$(cd "$WT" && for ros in ros_vanilla445.csv ros_vanilla4t45.csv; do .venv/bin/python scripts/campaign_select_v2.py --xpu xpu_a_cpsat_hard.csv --ros $ros --json | python3 -c "import json,sys; c=json.load(sys.stdin)['choice']; print(f'{c[\"cruise\"]} $ros') if c else None"; done | head -n 1)
CRU="${CRUISE:-${CHOICE%% *}}"; CRU="${CRU:-1.2}"; ROSTRACE="${CHOICE##* }"; [ -n "$ROSTRACE" ] || ROSTRACE=ros_vanilla4t45.csv
say "cruise $CRU, baseline trace $ROSTRACE"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 8500 ]; }
run_cond(){ local cond=$1 trace=$2
  for s in $SEEDS; do
    if [ -f "$ER/${cond}_s${s}/figure_data.npz" ]; then say "have $cond seed $s"; continue; fi
    until gpu_room; do sleep 60; done
    say "energy: $cond seed $s (trace $(basename $trace), cruise $CRU)"
    (cd "$CAN" && timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
      --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --moment_scale 0.0055 \
      --cruise_speed "$CRU" --walk_speed 0.0 --episodes 1 --seed "$s" --max_steps 1800 --ctrl_trace "$trace" \
      --dump_figure_data "$ER/${cond}_s${s}" >> "$ER/${cond}.log" 2>&1)
  done
}
run_cond xpu_cpsat   $RES/ctrl_traces/xpu_a_cpsat_hard.csv
run_cond ros_vanilla $RES/ctrl_traces/${ROSTRACE}
run_cond xpu_greedy  $RES/ctrl_traces/xpu_a_greedy.csv
run_cond ros_shipped $RES/ctrl_traces/ros_vanilla45.csv
say "computing modeled propulsive energy from the logged wrench..."
$PY "$WT/scripts/flight_energy_model.py" --glob "$ER/*_s*/figure_data.npz" --out "$OUTCSV" 2>&1 | tee -a "$LOG"
say "ENERGY_V2_DONE -> $OUTCSV"
