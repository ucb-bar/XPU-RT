#!/usr/bin/env bash
# Actuation-parity screen for google_robot: can it run at the 27 Hz (37.04 ms) fine
# grid the way widowx runs at 25 Hz (40 ms)? All runs at ZERO latency, so the only
# thing under test is whether the actuation model reproduces the registered one.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
# env.sh cd's to roselite/, so source it FIRST and cd back afterwards.
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"
source parity/configs.sh
N=${N:-3}; RNG=${RNG:-80}; TASKS=${TASKS:-"google_robot_close_drawer google_robot_pick_coke_can"}
OUT=parity/runs; mkdir -p "$OUT" parity/logs
for T in $TASKS; do
  short=$(echo "$T" | sed 's/google_robot_//')
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r name act tick cm <<< "$c"
    key="${short}_${name}_n${N}"
    [ -f "$OUT/$key/summary.json" ] && { echo "skip $key"; continue; }
    python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng "$RNG" --n "$N" \
        --actuation "$act" --tick-hz "$tick" --control-mode "$cm" \
        --out "$OUT/$key" > "parity/logs/${key}.log" 2>&1
    rc=$?
    echo "$key rc=$rc : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
  done
done
