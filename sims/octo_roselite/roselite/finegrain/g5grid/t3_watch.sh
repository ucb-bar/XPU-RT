#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
while true; do
  n=$(for h in "${DR[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3 2>/dev/null && for x in */; do [ -f "$x/energy2.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  r=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[t]race_eval2.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) torque3=$n/36 runners=${r:-0}"
  [ "$n" -ge 36 ] && { echo TORQUE3_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n"; break; }
  sleep 420
done
