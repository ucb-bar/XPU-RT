#!/usr/bin/env bash
# (campaign_percep.sh with two additions: CROSS=1 flies people crossing the aisle, and the GPU gate also
# waits while a simulator launched in the last 90 s has not yet claimed its memory.)
# Flight campaign, third form: each arm replays BOTH board measurements of its deployment — the
# control-output cadence (scripts/ctrl_trace_from_board.py) and the camera-to-control latency
# (--percep_latency_ms: the navigation decision the drone acts on was computed that long ago,
# the median of the arm's board run). Every flight recorded. Cells already in the CSV are skipped.
#
#   ARMS="xpu_cpsat:<trace>:56.3 ros_vanilla4:<trace>:242.5" SPEEDS="1.8 1.4 1.0" scripts/campaign_percep.sh
. "$(dirname "$0")/env.sh"
set -u
# arm = name:trace:latency_ms[:hold_ms]  (hold = the navigation goal refreshed at the deployment's measured goal rate, 0 = every step)
ARMS_ABS=""; for arm in ${ARMS:?}; do n=${arm%%:*}; rest=${arm#*:}; tr=${rest%%:*}; rest=${rest#*:}; lat=${rest%%:*}; hold=0; [[ "$rest" == *:* ]] && hold=${rest#*:}; ARMS_ABS="$ARMS_ABS $n:$(readlink -f "$tr"):$lat:$hold"; done; ARMS="$ARMS_ABS"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_percep}"; mkdir -p "$OUT"; OUT=$(readlink -f "$OUT"); mkdir -p $OUT/tmp $OUT/records; export TMPDIR=$OUT/tmp   # absolute: the flights run from the simulator tree
PY="$ISAAC_PY"; W="${W:-sims/models/warehouse/nav_fused_v12_cnn.pt}"; [[ "$W" == /* ]] || W=$(readlink -f "$W" 2>/dev/null || echo "$W")
SPEEDS="${SPEEDS:-1.8 1.4 1.0}"; SEEDS="${SEEDS:-12}"; SEED0="${SEED0:-1000}"; GAIN="${GAIN:-0.0055}"; DENS="${DENS:-0.30}"; COURSE="${COURSE:-a}"
WALK="${WALK:-0.0}"; CROSS="${CROSS:-0}"; XARG=""; [ "$CROSS" = 1 ] && XARG="--walk_cross"
gpu_room() { local free n young; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  young=$(ps -eo etimes,args | awk '$1 < 90 && /sweep_rate_demo.py|record_sensor_demo.py/ && !/timeout/' | wc -l)
  [ "$free" -ge "${NEED_MB:-8500}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ] && [ "$young" -eq 0 ]; }
cd "$SIM_TREE"
for arm in $ARMS; do name=${arm%%:*}; rest=${arm#*:}; trace=${rest%%:*}; rest=${rest#*:}; lat=${rest%%:*}; hold=${rest#*:}; [ -f "$trace" ] || { echo "no trace $trace"; exit 2; }
  for cru in $SPEEDS; do
    if [ -f $OUT/campaign.csv ] && python3 - "$OUT/campaign.csv" "$(basename $trace)" "$cru" "$GAIN" "$lat" "$DENS" "$COURSE" "$WALK" "$hold" <<'PY'
import csv, sys
p, tr, cru, gain, lat, dens, course, walk, hold = sys.argv[1:10]
rows=[r for r in csv.DictReader(open(p)) if r["ctrl_trace"]==tr and abs(float(r["cruise_speed"])-float(cru))<1e-9 and abs(float(r["moment_scale"])-float(gain))<1e-9
      and abs(float(r["percep_latency_ms"])-float(lat))<1e-6 and abs(float(r["prop_density"])-float(dens))<1e-9 and r.get("course","a")==course
      and abs(float(r["walk_speed"])-float(walk))<1e-9 and abs(float(r["percep_hold_ms"])-float(hold))<1e-6]
sys.exit(0 if len(rows)>=12 else 1)
PY
    then echo "skip $name cruise $cru lat $lat (done)"; continue; fi
    until gpu_room; do sleep 60; done
    tag="${name}_lat${lat}_h${hold}_${COURSE}_d${DENS}_w${WALK}$([ "$CROSS" = 1 ] && echo x)_g${GAIN}_c${cru}"; echo "=== $(date +%H:%M:%S) $tag trace=$(basename $trace) weights=$(basename $W) ==="
    WAREHOUSE_COURSE=$COURSE timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 \
      --obstacle_level 8 --prop_density $DENS --percep_hold_ms $hold --percep_latency_ms $lat --moment_scale $GAIN --cruise_speed $cru --walk_speed $WALK $XARG \
      --episodes $SEEDS --seed $SEED0 --max_steps 1800 --ctrl_trace $trace --sweep-csv $OUT/campaign.csv --record_dir $OUT/records/$tag > $OUT/$tag.log 2>&1
    grep -h "\[SWEEP\]" $OUT/$tag.log | cut -c1-160
  done
done
echo "CAMPAIGN_PERCEP_DONE $(date +%H:%M:%S)"
