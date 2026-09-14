#!/usr/bin/env bash
# Pull the per-cell energy2.json scalars from the 44-arm plane sweep (plane3/).
# Only energy2.json is pulled: the per-episode bodies/series files stay on the
# workers (13 MB/cell x 1760 would be ~23 GB, and every metric we plot is a
# scalar already reduced by reduce_torque3.py).
# Safe to re-run mid-sweep; rsync just adds the cells that finished since.
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
DEST=plane3_runs
i=0
for h in "${DR[@]}"; do
  mkdir -p "$DEST/h$i"
  rsync -a --include='*_g[0-9]*_rng*/' --include='*_g[0-9]*_rng*/energy2.json' \
        --exclude='*' -e "$SSH" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/plane3/ "$DEST/h$i/" 2>/dev/null &
  i=$((i+1))
done; wait
echo "cells local: $(find $DEST -name energy2.json 2>/dev/null | wc -l)"
echo "distinct   : $(find $DEST -name energy2.json -printf '%h\n' 2>/dev/null | xargs -n1 basename | sort -u | wc -l)"
