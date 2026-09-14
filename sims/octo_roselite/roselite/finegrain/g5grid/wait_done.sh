#!/usr/bin/env bash
# Block until the launcher has finished AND all 25 workers report ALLDONE.
set -u; cd "$(dirname "$0")"; source hosts_grid.sh
while pgrep -f 'launch_gri[d]' >/dev/null 2>&1; do sleep 30; done
echo "launcher finished $(date +%H:%M:%S)"
while :; do
  done_n=0; run_n=0
  for i in $(seq 0 24); do
    h=${HOSTS[n$i]}
    r=$(timeout 30 $SSH -n ubuntu@"$h" "grep -c ALLDONE /home/ubuntu/simpler/logs/grid_sweep.log 2>/dev/null || echo 0; pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr '\n' ' ')
    set -- ${r:-0 0}; [ "${1:-0}" -ge 1 ] && done_n=$((done_n+1)); run_n=$((run_n+${2:-0}))
  done
  echo "$(date +%H:%M:%S) workers_done=$done_n/25 runners=$run_n"
  [ "$done_n" -ge 25 ] && break
  sleep 240
done
echo "SWEEP COMPLETE $(date +%H:%M:%S)"
