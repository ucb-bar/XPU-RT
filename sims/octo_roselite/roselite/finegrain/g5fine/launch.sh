#!/usr/bin/env bash
# Fan the per-worker job lists out, 3 concurrent runs per box (the rate the wide
# sweep established as safe on a g5.xlarge: ~7 min per 24-episode run uncontended).
set -u
cd "$(dirname "$0")"; source hosts.sh
for w in $(printf '%s\n' "${!HOSTS[@]}" | sort); do
  h=${HOSTS[$w]}; jl=joblist_$w.txt
  [ -s "$jl" ] || { echo "$w: no jobs"; continue; }
  scp -q -i "$KEY" -o StrictHostKeyChecking=no "$jl" ubuntu@"$h":../simpler/joblist.txt || {
    echo "$w UNREACHABLE"; continue; }
  $SSH -n ubuntu@"$h" "cd ../simpler/ && chmod +x sim_eval/roselite/finegrain/job.sh && \
    setsid nohup bash -c 'xargs -a joblist.txt -I{} -P 3 bash sim_eval/roselite/finegrain/job.sh {} \
      > ../simpler/logs/fine_sweep.log 2>&1; echo ALLDONE >> ../simpler/logs/fine_sweep.log' \
    > /dev/null 2>&1 < /dev/null &" 
  echo "$w ($h): launched $(wc -l < "$jl") runs"
done
