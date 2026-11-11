#!/usr/bin/env bash
# Wait for all 180 DISTINCT g40 runs, then fetch (deduplicating across hosts) and
# analyse. One notification instead of a polling conversation.
set -u; cd "$(dirname "$0")/.."; source g5fine/g40_hosts.sh
while true; do
  n=$(for h in "${G40[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_g40 2>/dev/null && for x in */; do [ -f "$x/summary.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  r=$(for h in "${G40[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[f]inegrain_eval.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) distinct=$n/180 runners=${r:-0}"
  [ "$n" -ge 180 ] && break
  if [ "${r:-0}" -eq 0 ]; then echo "STALLED at $n with no runners"; break; fi
  sleep 540
done
bash g5fine/fetch_g40.sh
echo "===== ANALYSIS ====="
python3 g5fine/analyze_g40.py
echo G40_FINISH_DONE
