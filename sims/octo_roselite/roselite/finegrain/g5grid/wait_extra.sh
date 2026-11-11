#!/usr/bin/env bash
# Block until every dispatched worker has written ALLDONE.  Emits one line per poll.
# NOTE: this script never mentions the runner's script name outside a bracketed
# pattern, so its own command line cannot self-match a remote or local pgrep.
set -u
cd "$(dirname "$0")"; source phase3_hosts.sh
TAG=${1:-extra}
N=${#P3[@]}
for _ in $(seq 1 240); do
  done_n=0; run_n=0; live=0
  for i in $(seq 0 $((N-1))); do
    [ -s "jl_${TAG}_${i}.txt" ] || continue
    live=$((live+1))
    r=$(timeout 40 $SSH -n ubuntu@"${P3[$i]}" "grep -c ALLDONE /home/ubuntu/simpler/logs/${TAG}.log 2>/dev/null; pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr '\n' ' ')
    set -- $r
    [ "${1:-0}" -ge 1 ] 2>/dev/null && done_n=$((done_n+1))
    run_n=$((run_n + ${2:-0}))
  done
  echo "$(date -u +%H:%M:%S)  workers done ${done_n}/${live}   runners alive ${run_n}"
  [ "$done_n" -ge "$live" ] && { echo "ALL_WORKERS_DONE"; exit 0; }
  sleep 60
done
echo "WAIT_TIMEOUT"; exit 1
