#!/usr/bin/env bash
# Pull every fine-grain summary.json off the workers into g5fine/runs/<worker>/.
# Run dirs already carry task+arm+seed (job.sh sets --out), so names are unique
# across tasks; the per-worker subdir only guards against a re-run collision.
set -u
cd "$(dirname "$0")"; source hosts.sh
for w in $(printf '%s\n' "${!HOSTS[@]}" | sort); do
  h=${HOSTS[$w]}; mkdir -p "runs/$w"
  rsync -a --include='*/' --include='summary.json' --exclude='*' -e "$SSH" \
    ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/ "runs/$w/" 2>/dev/null
  echo "$w  $(find "runs/$w" -name summary.json 2>/dev/null | wc -l) summaries"
done
echo "TOTAL: $(find runs -name summary.json 2>/dev/null | wc -l)"
