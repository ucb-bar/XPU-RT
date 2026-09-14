#!/usr/bin/env bash
# Tracks BOTH phases. Anchored patterns so stale dirs cannot inflate either count.
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
while true; do
  seed=$(for h in "${DR[@]}"; do
      $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3 2>/dev/null && for x in */; do [ -f "$x/energy2.json" ] && basename "$x"; done' 2>/dev/null
    done | sort -u | wc -l)
  plane=$(for h in "${DR[@]}"; do
      $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/plane3 2>/dev/null && for x in */; do [ -f "$x/energy2.json" ] && basename "$x"; done' 2>/dev/null
    done | grep -E '^(egg|spoon|coke|drawer)_g[0-9]+_[0-9]+_rng[0-9]+$' | sort -u | wc -l)
  r=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[t]race_eval2.py"' 2>/dev/null; done | paste -sd+ | bc)
  w=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[c]hain_runner.sh"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) seeded=$seed/360 plane=$plane/1760 runners=${r:-0} chain=${w:-0}"
  [ "$plane" -ge 1760 ] && { echo BOTH_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && [ "${w:-0}" -eq 0 ] && { echo "STALLED seeded=$seed plane=$plane"; break; }
  sleep 600
done
