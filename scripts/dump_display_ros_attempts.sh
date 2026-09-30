#!/usr/bin/env bash
# The displayed baseline flight. The simulator is not run-to-run deterministic, so a seed that
# crashed mid-course in the campaign can fly clean when re-dumped; the display flight is
# therefore re-dumped seed by seed over the cell's mid-course-crash seeds until one attempt
# reproduces the rule's flight (a crash after exactly two gates). Every attempt is appended to
# display.csv, so the count of attempts is on record next to the one shown.
set -u; WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"; cd "$SIM_TREE"
D=$RES/campaign/display; mkdir -p $D/tmp; export TMPDIR=$D/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
CRU="${CRUISE:-1.4}"; ROS_LAT="${ROS_LAT:-29.93}"; SEEDS="${SEEDS:-1008 1005 1002 1007}"; MAX="${MAX_ATTEMPTS:-8}"
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --moment_scale 0.0055 --cruise_speed $CRU --walk_speed 0.0 --episodes 1 --max_steps 1900"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-7000}" ]; }
k=0
while [ $k -lt $MAX ]; do
  for s in $SEEDS; do
    k=$((k+1)); [ $k -gt $MAX ] && break
    until gpu_room; do sleep 60; done
    tag=ros_s${s}_a$k; rm -rf $D/${tag}_figdata
    echo "=== $(date +%H:%M:%S) $tag"
    timeout 1800 $PY sims/scripts/sweep_rate_demo.py $COMMON --sched_latency_ms $ROS_LAT --seed $s --dump_figure_data $D/${tag}_figdata --sweep-csv $D/display.csv > $D/$tag.log 2>&1
    line=$(grep -h SWEEP $D/$tag.log | grep -o "s${s}:[a-z]*([0-9]/4,[0-9]*,[a-z-]*)"); echo "  $line"
    if echo "$line" | grep -q "cras(2/4"; then echo "ROS_DISPLAY=$D/${tag}_figdata"; echo ROS_ATTEMPTS_DONE; exit 0; fi
  done
done
echo "no attempt reproduced a two-gate crash in $MAX tries"; echo ROS_ATTEMPTS_DONE
