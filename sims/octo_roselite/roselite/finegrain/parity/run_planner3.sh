#!/usr/bin/env bash
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"
N=${N:-12}
for T in google_robot_pick_coke_can google_robot_close_drawer; do
  short=$(echo "$T" | sed 's/google_robot_//')
  for spec in "native:3" "fine:27"; do
    act=${spec%%:*}; tick=${spec##*:}
    key="${short}_M16_${act}${tick}_n${N}"
    [ -f "parity/runs/$key/summary.json" ] && { echo "skip $key"; continue; }
    python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng 80 --n "$N" \
        --actuation "$act" --tick-hz "$tick" --planner-limit-scale 16 --allow-lag-discard \
        --out "parity/runs/$key" > "parity/logs/${key}.log" 2>&1
    echo "$key : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
  done
done
echo M16_DONE
