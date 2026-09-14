#!/usr/bin/env bash
# CORRECTED finisher. The first one counted `drawer_*_rng*`, which also matches the
# ORIGINAL 9-arm sweep's run dirs (drawer_lat0_rng110, drawer_cpu685_rng110, ...).
# Those 110 stale dirs inflated the count to 889 while only 779 grid runs existed, so
# it broke out of the wait ~95 runs early and fetched an incomplete sweep.
# Grid arms are ONLY of the form g<PERIOD>_<WINDOW>; anchor the pattern to that.
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
count_grid () {
  for h in "${DR[@]}"; do
    $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs 2>/dev/null && for x in drawer_g*_rng*/; do [ -f "$x/summary.json" ] && basename "$x"; done' 2>/dev/null
  done | grep -E '^drawer_g[0-9]+_[0-9]+_rng[0-9]+$' | sort -u | wc -l
}
while true; do
  n=$(count_grid)
  r=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[f]inegrain_eval.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) grid=$n/880 runners=${r:-0}"
  [ "$n" -ge 880 ] && break
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n grid runs with no runners"; break; }
  sleep 480
done
i=0
for h in "${DR[@]}"; do
  mkdir -p "runs/dr_$i"
  rsync -a --include='drawer_g*_rng*/' --include='drawer_g*_rng*/summary.json' --exclude='*' \
    -e "$SSH" ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/ "runs/dr_$i/" 2>/dev/null &
  i=$((i+1))
done; wait
L=$(find runs -name summary.json -path '*drawer_g*' | xargs -n1 dirname | xargs -n1 basename | grep -E '^drawer_g[0-9]+_[0-9]+_rng[0-9]+$' | sort -u | wc -l)
echo "local distinct GRID drawer runs: $L/880"
echo DRAWER2_DONE
