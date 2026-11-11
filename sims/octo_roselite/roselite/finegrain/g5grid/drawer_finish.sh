#!/usr/bin/env bash
# Wait for all 880 DISTINCT drawer runs, then fetch (deduplicated across hosts) and
# report. Counts distinct keys, never a per-host sum -- a per-host sum once turned 107
# runs into an apparent 138 in this project.
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
while true; do
  n=$(for h in "${DR[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs 2>/dev/null && for x in drawer_*_rng*/; do [ -f "$x/summary.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  r=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[f]inegrain_eval.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) drawer=$n/880 runners=${r:-0}"
  [ "$n" -ge 880 ] && break
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n with no runners"; break; }
  sleep 600
done
i=0
for h in "${DR[@]}"; do
  mkdir -p "runs/dr_$i"
  rsync -a --include='drawer_*_rng*/' --include='drawer_*_rng*/summary.json' --exclude='*' \
    -e "$SSH" ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/ "runs/dr_$i/" 2>/dev/null &
  i=$((i+1))
done; wait
echo "local distinct drawer runs: $(find runs -name summary.json -path '*drawer_*' | xargs -n1 dirname 2>/dev/null | xargs -n1 basename 2>/dev/null | sort -u | wc -l)/880"
echo DRAWER_FINISH_DONE
