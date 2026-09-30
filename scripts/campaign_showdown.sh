#!/usr/bin/env bash
# The gate-course showdown as a sweep, not a pair: every runtime arm at every cruise speed under
# both gain policies, twelve seeds each. One CSV, cell-resumable, so an interrupted run picks up
# where it stopped and never double-counts a cell.
#
# ARMS. Each arm's control cadence is a MEASURED control-timer gap on the K1, entering the
# simulator through its refresh rule ceil(gap / 10 ms):
#
#   xpu     6.30 ms   XPU-RT, mlp_control in its scheduled 10 ms window          -> 100 Hz
#   ros33  22.21 ms   ROS 2 default executor, camera 15 Hz (ros_traced/15_ship)   ->  33 Hz
#   ros20  47.01 ms   same, camera 20 Hz                     (ros_traced/20_ship) ->  20 Hz
#   ros17  52.02 ms   same, camera 25 Hz                     (ros_traced/25_ship) ->  17 Hz
#
# The ROS arms differ only in how fast the camera runs: a single-threaded executor serves the
# control timer between perception callbacks, so the faster the camera, the rarer the control
# output. The values are the replicate-1 means; the figure re-derives them from every replicate.
#
# GAIN. "fixed" holds moment_scale at 0.0055 on every arm -- one controller, as deployed, at
# whatever cadence the runtime delivers. "cal" applies the calibration law 0.5/eff_hz per cell,
# so each arm gets the authority its own cadence calls for. A conclusion that holds under both
# is not a gain artifact.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/campaign}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp" "$OUT/dumps"; export TMPDIR="$OUT/tmp"
CSV="$OUT/campaign.csv"; LOG="$OUT/campaign.log"
cd "$CAN" || exit 1

have() {  # rows already in the CSV for (cruise, sched_latency, moment_scale)
  [ -f "$CSV" ] || { echo 0; return; }
  awk -F, -v c="$1" -v l="$2" -v m="$3" 'NR>1 && ($2+0==c+0) && ($6+0==l+0) && ($14+0==m+0){n++} END{print n+0}' "$CSV"
}
fly() {  # $1=arm $2=lat $3=gain-policy $4=moment $5=cruise
  local n; n=$(have "$5" "$2" "$4")
  if [ "$n" -ge "$EPS" ]; then echo "skip $1 $3 cruise=$5 ($n rows)" | tee -a "$LOG"; return; fi
  echo "=== $(date +%H:%M:%S) $1 $3 cruise=$5 lat=$2 moment=$4 ===" | tee -a "$LOG"
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale "$4" \
    --cruise_speed "$5" --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/dumps/${1}_${3}_c${5}" --sweep-csv "$CSV" 2>&1 \
    | grep -E "\[SWEEP\]|Traceback|Error" | tee -a "$LOG"
}

ARMS="xpu:6.30:0.0050 ros33:22.21:0.0150 ros20:47.01:0.0250 ros17:52.02:0.0300"
for POLICY in fixed cal; do
  for CRU in 1.4 1.2 1.6 1.0 1.8 2.0; do
    for A in $ARMS; do
      IFS=: read -r arm lat calgain <<<"$A"
      if [ "$POLICY" = fixed ]; then g=0.0055; else g=$calgain; fi
      fly "$arm" "$lat" "$POLICY" "$g" "$CRU"
    done
  done
done
echo "CAMPAIGN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"
