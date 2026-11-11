#!/usr/bin/env bash
# Push the fine-grain harness to every worker. The wide sweep only ever shipped
# roselite/latency_eval.py (the COARSE 200 ms model); finegrain/ is new on these
# boxes, as is the task-derived timing in finegrain_eval.py.
set -u
cd "$(dirname "$0")"; source hosts.sh
LOCAL=/scratch2/dima/misc_sw/octo_work/sim_eval
for w in $(printf '%s\n' "${!HOSTS[@]}" | sort); do
  h=${HOSTS[$w]}
  $SSH ubuntu@"$h" "mkdir -p /home/ubuntu/simpler/sim_eval/roselite/finegrain/logs" || { echo "$w UNREACHABLE"; continue; }
  rsync -a -e "$SSH" \
    "$LOCAL/roselite/finegrain/finegrain_eval.py" \
    "$LOCAL/roselite/finegrain/g5fine/job.sh" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/
  rsync -a -e "$SSH" "$LOCAL/octo15_inference.py" ubuntu@"$h":/home/ubuntu/simpler/sim_eval/
  echo "$w ($h) pushed: $($SSH ubuntu@"$h" 'md5sum /home/ubuntu/simpler/sim_eval/roselite/finegrain/finegrain_eval.py | cut -c1-12')"
done
echo "local md5: $(md5sum "$LOCAL/roselite/finegrain/finegrain_eval.py" | cut -c1-12)"
