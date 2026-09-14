#!/usr/bin/env bash
# Recover plane3 cells that SIMULATED but were never reduced.
#
# Cause: the seed-300 pass ran before job_plane3.sh gained its reduce_torque3.py
# step. Those cells hold a complete summary.json and the full per-tick .npy set --
# they are missing only energy2.json, so the fix is a reduction, NOT a re-run.
#
# Reduction is sequential (one at a time per host) so it cannot steal CPU from the
# simulations still finishing the tail of the sweep.
#
#   ./recover_plane3.sh count   -- report how many cells are recoverable
#   ./recover_plane3.sh run     -- reduce them
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
MODE="${1:-count}"
FG=/home/ubuntu/simpler/sim_eval/roselite/finegrain

if [ "$MODE" = count ]; then
  for h in "${DR[@]}"; do
    $SSH ubuntu@$h "cd $FG/plane3 2>/dev/null || exit 0
      for x in */; do [ -f \"\$x/summary.json\" ] && [ ! -f \"\$x/energy2.json\" ] && echo \"\${x%/}\"; done" 2>/dev/null &
  done; wait
  exit 0
fi

for h in "${DR[@]}"; do
  ( $SSH ubuntu@$h "source /home/ubuntu/simpler/sim_eval/roselite/env.sh; cd $FG || exit 1
      n=0; fail=0
      for x in plane3/*/; do
        [ -f \"\$x/summary.json\" ] || continue
        [ -f \"\$x/energy2.json\" ] && continue
        if python reduce_torque3.py \"\$x\" >> /home/ubuntu/recover_plane3.log 2>&1; then n=\$((n+1)); else fail=\$((fail+1)); echo \"FAIL \$x\" >> /home/ubuntu/recover_plane3.log; fi
      done
      echo \"$h reduced=\$n failed=\$fail\"" 2>/dev/null ) &
done; wait
