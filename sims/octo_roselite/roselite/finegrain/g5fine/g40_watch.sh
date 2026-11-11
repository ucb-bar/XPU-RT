#!/usr/bin/env bash
# Count DISTINCT completed keys across the fleet. A per-host sum double-counts any
# run that was reassigned between hosts -- that is how 107 distinct runs were
# reported as 138 earlier.
set -u; cd "$(dirname "$0")/.."; source g5fine/g40_hosts.sh
while true; do
  n=$(for h in "${G40[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_g40 2>/dev/null && for x in */; do [ -f "$x/summary.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  r=$(for h in "${G40[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[f]inegrain_eval.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) distinct=$n/180 runners=${r:-0}"
  [ "$n" -ge 180 ] && { echo G40_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && { echo "G40_STALLED at $n"; break; }
  sleep 600
done
