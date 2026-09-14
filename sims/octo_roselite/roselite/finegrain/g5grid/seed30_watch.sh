#!/usr/bin/env bash
# Progress for the n=10 -> n=30 widowx seed extension. Counts REDUCED cells
# (energy2.json), never bare directories: a directory exists the moment a job
# starts, so counting those reports work that has not happened yet.
set -u; cd "$(dirname "$0")"; source seed30_hosts.sh
while true; do
  n=$(for h in "${S30[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3 2>/dev/null && for x in {egg,spoon}_*_rng1[12][0-9]/; do [ -f "$x/energy2.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  r=$(for h in "${S30[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[t]race_eval2.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) seed30=$n/240 runners=${r:-0}"
  [ "$n" -ge 240 ] && { echo SEED30_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n"; break; }
  sleep 420
done
