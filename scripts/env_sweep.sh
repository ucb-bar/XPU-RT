#!/usr/bin/env bash
# Environment x speed x cadence sweep, every flight recorded (path, commanded wrench, body rates,
# gates, outcome; scripts/env_sweep_summary.py turns the wrench into rotor power). People are
# 2.4 m tall (they must be avoided); the environment axis is obstacle density x gate course.
#
#   scripts/env_sweep.sh <driver-id> "<arm:trace ...>"     (one simulator per driver; GPU-room gated)
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; ID="${1:?driver id}"; ARMS="${2:?arms}"
OUT=$WT/results/codesign_feedback/campaign_env; mkdir -p $OUT/tmp $OUT/records; export TMPDIR=$OUT/tmp
CSV=$OUT/env_sweep.csv; PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
SPEEDS="${SPEEDS:-1.8 1.4 1.0 1.6 1.2}"; DENS="${DENS:-0.30 0.40 0.20}"; COURSES="${COURSES:-a b}"; GAIN="${GAIN:-0.0055}"; SEEDS="${SEEDS:-12}"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-8500}" ]; }
have() { python3 - "$CSV" "$1" "$2" "$3" "$4" "$GAIN" <<'PY'
import csv, sys, os
p, tr, cru, dens, course, gain = sys.argv[1:7]
if not os.path.exists(p): sys.exit(1)
n = sum(1 for r in csv.DictReader(open(p)) if r["ctrl_trace"]==tr and abs(float(r["cruise_speed"])-float(cru))<1e-9 and abs(float(r["prop_density"])-float(dens))<1e-9 and r.get("course","a")==course and abs(float(r["moment_scale"])-float(gain))<1e-9)
sys.exit(0 if n >= 12 else 1)
PY
}
cd "$SIM_TREE"
for course in $COURSES; do for dens in $DENS; do for arm in $ARMS; do name=${arm%%:*}; trace=$WT/${arm#*:}; for cru in $SPEEDS; do
  have "$(basename $trace)" "$cru" "$dens" "$course" && { echo "skip $name $course d$dens c$cru"; continue; }
  until gpu_room; do sleep 60; done
  tag="${name}_${course}_d${dens}_c${cru}"; echo "=== $(date +%H:%M:%S) [$ID] $tag ==="
  WAREHOUSE_COURSE=$course timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 \
    --obstacle_level 8 --prop_density $dens --percep_hold_ms 0 --moment_scale $GAIN --cruise_speed $cru --walk_speed 0.0 \
    --episodes $SEEDS --seed 1000 --max_steps 1800 --ctrl_trace $trace --sweep-csv $CSV --record_dir $OUT/records/$tag > $OUT/$tag.log 2>&1
  grep -h "\[SWEEP\]" $OUT/$tag.log | cut -c1-120
done; done; done; done
echo "ENV_SWEEP_DONE [$ID]"
