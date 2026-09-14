#!/usr/bin/env bash
# Pull the new seed cells straight into the existing traces_torque3/ tree, so
# every analysis that already reads it picks them up with no change.
set -u; cd "$(dirname "$0")"; source seed30_hosts.sh
DEST=../traces_torque3
for h in "${S30[@]}"; do
  rsync -a --include='*_rng1[12][0-9]/' --include='*_rng1[12][0-9]/energy2.json' \
        --include='*_rng1[12][0-9]/summary.json' --include='*_rng1[12][0-9]/series.npz' \
        --include='*_rng1[12][0-9]/drive_config.json' --exclude='*' -e "$SSH" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3/ "$DEST/" 2>/dev/null &
done; wait
echo "widowx cells now local: $(ls -d $DEST/egg_* $DEST/spoon_* 2>/dev/null | wc -l)"
