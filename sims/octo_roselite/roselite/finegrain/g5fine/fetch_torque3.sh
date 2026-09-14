#!/usr/bin/env bash
# Pull the REDUCED traces_torque3 cells into the local tree.
set -u
cd "$(dirname "$0")"; source torque3_hosts.sh
D=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/traces_torque3
mkdir -p "$D"
for i in $(seq 0 24); do
  w=t$i; h=${HOSTS[$w]}
  rsync -a -e "$SSH" ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3/ "$D/" 2>/dev/null
done
echo "local cells with energy2.json: $(ls -d $D/*/energy2.json 2>/dev/null | wc -l)/36"
ls -d "$D"/*/ 2>/dev/null | sed 's|.*/traces_torque3/||;s|/$||' | \
  grep -Ec '^(egg|spoon|coke|drawer)_(lat0|pipe110fix|p105w300|p130w275|p150w300|pipe200fix|serial283|fp32_555|cpu685)$' | \
  xargs -I{} echo "DISTINCT anchored cell dirs: {}/36"
