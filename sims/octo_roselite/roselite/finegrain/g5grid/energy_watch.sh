#!/usr/bin/env bash
# Counts DISTINCT energy cells. Pattern anchored to <task>_g<P>_<W>_rng<seed> so stale
# run dirs from earlier sweeps cannot inflate it -- that inflation once made an
# incomplete drawer sweep look finished (889/880).
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
while true; do
  n=$(for h in "${DR[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_energy 2>/dev/null && for x in */; do [ -f "$x/energy.json" ] && basename "$x"; done' 2>/dev/null
      done | grep -E '^(egg|spoon|coke|drawer)_g[0-9]+_[0-9]+_rng[0-9]+$' | sort -u | wc -l)
  r=$(for h in "${DR[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[t]race_eval.py"' 2>/dev/null; done | paste -sd+ | bc)
  d=$($SSH ubuntu@"${DR[0]}" "df -h / | tail -1 | awk '{print \$5}'" 2>/dev/null)
  echo "$(date +%H:%M) energy=$n/1760 runners=${r:-0} disk0=${d:-?}"
  [ "$n" -ge 1760 ] && { echo ENERGY_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n"; break; }
  sleep 600
done
