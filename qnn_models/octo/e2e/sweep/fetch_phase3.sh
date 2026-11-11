#!/usr/bin/env bash
# Pull grid summaries from the CURRENT fleet. fetch_grid.sh targets hosts_grid.sh,
# whose IPs are stale: stopping and restarting an EC2 instance reassigns its public
# IP, so that script silently pulls nothing from the restarted workers.
set -u; cd "$(dirname "$0")"; source phase3_hosts.sh
i=0
for h in "${P3[@]}"; do
  mkdir -p "runs/p3_$i"
  rsync -a --include='*_g[0-9]*_rng*/' --include='*_g[0-9]*_rng*/summary.json' \
        --exclude='*' -e "$SSH" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/ "runs/p3_$i/" 2>/dev/null &
  i=$((i+1))
done; wait
echo "TOTAL local: $(find runs -name summary.json 2>/dev/null | wc -l)"
