#!/usr/bin/env bash
# The flight envelope under the calibrated gain law, at the envelope's own settings.
#
# The 240-flight envelope (hil_ablation.csv) holds moment_scale at 0.0055 on every cell: one
# controller, as deployed, at whatever cadence it is given. This grid is the same five cruise
# speeds, the same four control rates, the same seeds, the same simulator settings, with the
# gain calibrated per cell instead (moment_scale = 0.5 / eff_hz), so every cell has the authority
# its own cadence calls for. The two grids together say whether the floor is a property of the
# rate or of the gain.
#
# Cell-resumable: rows already present for (cruise, sched_latency, moment_scale) are not re-run.
# WAREHOUSE_COURSE=b runs the unseen-gate course into its own CSV.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/gain_controlled}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
if [ "${WAREHOUSE_COURSE:-a}" = b ]; then CSV="$OUT/gain_controlled_courseB.csv"; EPS="${EPS_B:-6}"; else CSV="$OUT/gain_controlled.csv"; fi
LOG="$OUT/grid.log"
cd "$CAN" || exit 1

have() {
  [ -f "$CSV" ] || { echo 0; return; }
  awk -F, -v c="$1" -v l="$2" -v m="$3" 'NR>1 && ($2+0==c+0) && ($6+0==l+0) && ($14+0==m+0){n++} END{print n+0}' "$CSV"
}
fly() {  # $1=cruise $2=sched_latency $3=moment $4=tag
  local n; n=$(have "$1" "$2" "$3")
  if [ "$n" -ge "$EPS" ]; then echo "skip $4 cruise=$1 ($n rows)" | tee -a "$LOG"; return; fi
  local need=$((EPS - n)); local seed=$((1000 + n))
  echo "=== $(date +%H:%M:%S) $4 cruise=$1 lat=$2 moment=$3 seeds $seed+$need course=${WAREHOUSE_COURSE:-a} ===" | tee -a "$LOG"
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale "$3" \
    --cruise_speed "$1" --walk_speed 0.0 --episodes "$need" --seed "$seed" --max_steps 1800 \
    --sweep-csv "$CSV" 2>&1 | grep -E "\[SWEEP\]|Traceback|Error" | tee -a "$LOG"
}

# rate -> (sched_latency that lands on it through ceil(lat/10), 0.5/eff_hz); same latencies as
# the fixed-gain envelope so the two grids share cells exactly.
for CRU in 1.0 1.2 1.4 1.6 1.8; do
  fly "$CRU" 48 0.02500 "cal_20Hz"
  fly "$CRU" 38 0.02000 "cal_25Hz"
  fly "$CRU" 28 0.01500 "cal_33Hz"
  fly "$CRU" 18 0.01000 "cal_50Hz"
  fly "$CRU"  8 0.00500 "cal_100Hz"
done
echo "GRID_DONE course=${WAREHOUSE_COURSE:-a} $(date +%H:%M:%S)" | tee -a "$LOG"
