#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source phase2_hosts.sh
tot=0
for h in "${P2[@]}"; do
  d=$($SSH ubuntu@$h 'ls -d /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_phase2/*/summary.json /home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_phase2/*/summary.json 2>/dev/null | wc -l' 2>/dev/null)
  r=$($SSH ubuntu@$h 'pgrep -cf "[f]inegrain_eval.py|[t]race_eval.py"' 2>/dev/null; echo)
  a=$($SSH ubuntu@$h 'grep -c P2_ALLDONE /home/ubuntu/p2_runner.log 2>/dev/null' 2>/dev/null; echo)
  m=$($SSH ubuntu@$h 'free -m | awk "/^Mem:/{print \$7}"' 2>/dev/null; echo)
  echo "  $h done=${d:-0} runners=${r:-0} ALLDONE=${a:-0} freeMB=${m:-?}"
  tot=$((tot+${d:-0}))
done
echo "TOTAL: $tot/198   $(date +%H:%M:%S)"
