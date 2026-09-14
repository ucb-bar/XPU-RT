#!/usr/bin/env bash
# Push trace_eval2.py + reduce_torque3.py + job_torque3.sh + this worker's joblist
# by SCP (never nested quoting through ssh), then detach one runner per worker.
#
# The runner's xargs uses -P 1: each worker has a single GPU and the joblists are
# already balanced by the measured traces_torque2 wall times.
#
# The verify pgrep is bracketed AND must not contain the literal string the runner
# matches on, or the remote shell self-matches and reports a phantom runner. Never
# trust the detaching ssh's exit code -- count the runner PIDs.
set -u
cd "$(dirname "$0")"; source torque3_hosts.sh
F=/home/ubuntu/simpler/sim_eval/roselite/finegrain
RUN="cd /home/ubuntu/simpler && setsid nohup bash -c 'xargs -a joblist_tq3.txt -I{} -P 1 bash $F/job_torque3.sh {} > /home/ubuntu/simpler/logs/tq3_sweep.log 2>&1; echo ALLDONE >> /home/ubuntu/simpler/logs/tq3_sweep.log' >/dev/null 2>&1 </dev/null & disown"
ok=0
for i in $(seq 0 24); do
  w=t$i; h=${HOSTS[$w]}
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no joblist_$w.txt ubuntu@"$h":/home/ubuntu/simpler/joblist_tq3.txt || { echo "$w SCP-JOBLIST FAILED"; continue; }
  timeout 90 scp -q -i "$KEY" -o StrictHostKeyChecking=no ../trace_eval2.py reduce_torque3.py job_torque3.sh ubuntu@"$h":$F/ || { echo "$w SCP-CODE FAILED"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" "mkdir -p /home/ubuntu/simpler/logs $F/traces_torque3; rm -f /home/ubuntu/simpler/logs/tq3_sweep.log; chmod +x $F/job_torque3.sh" >/dev/null 2>&1
  timeout 45 $SSH -n ubuntu@"$h" "$RUN" >/dev/null 2>&1
  sleep 6
  n=$(timeout 40 $SSH -n ubuntu@"$h" "pgrep -f 'trace_eva[l]2' | wc -l" 2>/dev/null | tr -cd '0-9')
  x=$(timeout 40 $SSH -n ubuntu@"$h" "pgrep -f 'joblist_tq[3].txt' | wc -l" 2>/dev/null | tr -cd '0-9')
  [ "${x:-0}" != "0" ] && ok=$((ok+1))
  echo "$w ($h): $(wc -l < joblist_$w.txt) jobs, xargs=${x:-?} trace_eval2=${n:-?}"
done
echo "workers with a live runner: $ok/25"
