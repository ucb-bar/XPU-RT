#!/usr/bin/env bash
# Pull every summary.json off all six workers into g5wide/runs/<worker>/.
# Run dirs are made unique across tasks by latency_eval.py's --tag (_GW_<task>),
# and unique across workers by the per-worker subdirectory here.
set -u
G5=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/g5wide
KEY=$HOME/.ssh/firesim.pem
declare -A HOSTS=(
  [w0]=3.93.66.216      # i-043560448065532ff  (pre-existing)
  [w1]=18.234.29.157    # i-05289cfa068e00c74
  [w2]=3.232.133.234    # i-09db19b86bffe6acb
  [w3]=3.87.210.147     # i-078a22ff12e6c885d
  [w4]=32.198.80.184    # i-039d68dfbf56a4f61
  [w5]=204.236.245.40   # i-0f7e65981bc8d4d7c
)
for w in "${!HOSTS[@]}"; do
  mkdir -p "$G5/runs/$w" "$G5/validation/$w"
  rsync -a --include='*/' --include='summary.json' --exclude='*' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes" \
    "ubuntu@${HOSTS[$w]}:/home/ubuntu/simpler/sim_eval/roselite/runs/" "$G5/runs/$w/" 2>/dev/null
  rsync -a --include='*/' --include='summary.json' --exclude='*' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes" \
    "ubuntu@${HOSTS[$w]}:/home/ubuntu/simpler/sim_eval/runs/" "$G5/validation/$w/" 2>/dev/null
  echo "$w  runs=$(find "$G5/runs/$w" -name summary.json 2>/dev/null | wc -l)" \
       " gate=$(find "$G5/validation/$w" -name summary.json 2>/dev/null | wc -l)"
done
echo "TOTAL sweep summaries: $(find "$G5/runs" -name summary.json | wc -l)"
