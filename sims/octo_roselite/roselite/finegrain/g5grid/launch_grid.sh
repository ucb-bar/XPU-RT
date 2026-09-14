#!/usr/bin/env bash
# Push joblist+job script and detach the runner on every worker, ONE AT A TIME,
# verifying the runner count. Never trust the exit code of the detaching ssh --
# it routinely hangs past its timeout on success. The verify pgrep must not
# contain the literal path .../finegrain_eval.py anywhere in the command, or the
# remote shell self-matches and reports a phantom runner.
set -u
cd "$(dirname "$0")"; source hosts_grid.sh
RUN="cd /home/ubuntu/simpler && setsid nohup bash -c 'xargs -a joblist.txt -I{} -P 3 bash sim_eval/roselite/finegrain/job_grid.sh {} > /home/ubuntu/simpler/logs/grid_sweep.log 2>&1; echo ALLDONE >> /home/ubuntu/simpler/logs/grid_sweep.log' >/dev/null 2>&1 </dev/null & disown"
for i in $(seq 0 24); do
  w=n$i; h=${HOSTS[$w]}
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no joblist_$w.txt ubuntu@"$h":/home/ubuntu/simpler/joblist.txt || { echo "$w SCP-JOBLIST FAILED"; continue; }
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no job_grid.sh ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/job_grid.sh || { echo "$w SCP-JOB FAILED"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" "mkdir -p /home/ubuntu/simpler/logs /home/ubuntu/simpler/sim_eval/roselite/finegrain/{logs,runs}; rm -f /home/ubuntu/simpler/logs/grid_sweep.log; chmod +x /home/ubuntu/simpler/sim_eval/roselite/finegrain/job_grid.sh" >/dev/null 2>&1
  timeout 45 $SSH -n ubuntu@"$h" "$RUN" >/dev/null 2>&1
  sleep 10
  n=$(timeout 40 $SSH -n ubuntu@"$h" "pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr -cd '0-9')
  echo "$w ($h): $(wc -l < joblist_$w.txt) jobs, runners=${n:-?}"
done
