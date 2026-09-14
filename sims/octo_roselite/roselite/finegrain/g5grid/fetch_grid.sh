#!/usr/bin/env bash
# Pull only the g<P>_<W> grid summaries (the AMI also carries 78 stale dirs).
set -u; cd "$(dirname "$0")"; source hosts_grid.sh
for i in $(seq 0 24); do
  w=n$i; h=${HOSTS[$w]}; mkdir -p "runs/$w"
  rsync -a --include='*_g[0-9]*_rng*/' --include='*_g[0-9]*_rng*/summary.json' \
        --exclude='*' -e "$SSH" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/ "runs/$w/" 2>/dev/null &
done; wait
echo "TOTAL: $(find runs -name summary.json 2>/dev/null | wc -l) summaries   $(date +%H:%M:%S)"
