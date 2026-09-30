#!/usr/bin/env bash
# Flight campaign, second form: every arm is a measured control-output trace from the K1
# (scripts/ctrl_trace_from_board.py), replayed by the simulator; cruise speed is the axis.
#
#   ARMS="xpu_cpsat:results/.../ctrl_traces/xpu_a_cpsat_hard.csv ros_vanilla:results/.../ctrl_traces/ros_vanilla445.csv" \
#   SPEEDS="1.0 1.2 1.4 1.6 1.8 2.0" OUT=results/codesign_feedback/campaign_v2 scripts/campaign_v2.sh
#
# One simulator at a time per invocation, gated on GPU room; cells already in the CSV (same trace, speed and gain) are skipped.
. "$(dirname "$0")/env.sh"
set -u
ARMS_ABS=""; for arm in ${ARMS:?}; do ARMS_ABS="$ARMS_ABS ${arm%%:*}:$(readlink -f "${arm#*:}")"; done; ARMS="$ARMS_ABS"   # traces resolved from the caller's directory
cd "$SIM_TREE"
OUT="${OUT:-$RES/campaign_v2}"; mkdir -p $OUT/tmp; export TMPDIR=$OUT/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8 2.0}"; SEEDS="${SEEDS:-12}"; SEED0="${SEED0:-1000}"; GAIN="${GAIN:-0.0055}"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-8500}" ]; }
for arm in $ARMS; do name=${arm%%:*}; trace=${arm#*:}; [ -f "$trace" ] || { echo "no trace $trace"; exit 2; }
  for cru in $SPEEDS; do
    if [ -f $OUT/campaign_v2.csv ] && python3 - "$OUT/campaign_v2.csv" "$(basename $trace)" "$cru" "$GAIN" <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r["ctrl_trace"]==sys.argv[2] and abs(float(r["cruise_speed"])-float(sys.argv[3]))<1e-9 and abs(float(r["moment_scale"])-float(sys.argv[4]))<1e-9]
sys.exit(0 if len(rows)>=12 else 1)
PY
    then echo "skip $name cruise $cru (done)"; continue; fi
    until gpu_room; do sleep 60; done
    echo "=== $(date +%H:%M:%S) $name cruise=$cru trace=$(basename $trace) ==="
    timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 \
      --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --moment_scale $GAIN --cruise_speed $cru --walk_speed 0.0 \
      --episodes $SEEDS --seed $SEED0 --max_steps 1800 --ctrl_trace $trace --sweep-csv $OUT/campaign_v2.csv > $OUT/${name}_g${GAIN}_c${cru}.log 2>&1
    grep -h "\[SWEEP\]" $OUT/${name}_g${GAIN}_c${cru}.log | cut -c1-400
  done
done
echo "CAMPAIGN_V2_DONE $(date +%H:%M:%S)"
