#!/usr/bin/env bash
# Pull the torque traces off the workers into the local traces_torque2/ tree.
set -u
cd "$(dirname "$0")"; source hosts.sh
D=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/traces_torque2
for w in w0 w1 w2 w3 w4 w5; do
  rsync -a -e "$SSH" ubuntu@${HOSTS[$w]}:/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque2/ "$D/" 2>/dev/null
done
echo "local torque dirs with summary.json: $(find $D -name summary.json | wc -l)/36"
