#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source g40_hosts.sh
D=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/runs_g40
mkdir -p "$D"
for h in "${G40[@]}"; do
  rsync -a -e "$SSH" ubuntu@$h:/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_g40/ "$D/" 2>/dev/null
done
echo "local g40 summaries: $(find $D -name summary.json | wc -l)/180"
