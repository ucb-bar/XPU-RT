#!/usr/bin/env bash
# Launch the corrected pipelined arms, one worker at a time, and VERIFY the runner
# count. Never trust the exit code of the ssh that detaches the remote xargs -- it
# often hangs past its timeout even on success, and trusting it previously produced
# two workers with 6 runners writing the same run directories.
set -u
cd "$(dirname "$0")"; source hosts.sh
RUN="cd /home/ubuntu/simpler && setsid nohup bash -c 'xargs -a joblist.txt -I{} -P 3 bash sim_eval/roselite/finegrain/job_fix2.sh {} > /home/ubuntu/simpler/logs/fix2_sweep.log 2>&1; echo ALLDONE >> /home/ubuntu/simpler/logs/fix2_sweep.log' >/dev/null 2>&1 </dev/null & disown"
for w in w0 w1 w2 w3 w4 w5; do
  h=${HOSTS[$w]}
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no joblist_$w.txt ubuntu@"$h":/home/ubuntu/simpler/joblist.txt || { echo "$w SCP FAILED"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" "rm -f /home/ubuntu/simpler/logs/fix2_sweep.log; chmod +x sim_eval/roselite/finegrain/job_fix2.sh" >/dev/null 2>&1
  timeout 45 $SSH -n ubuntu@"$h" "$RUN" >/dev/null 2>&1
  sleep 12
  n=$(timeout 40 $SSH -n ubuntu@"$h" "pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr -cd '0-9')
  echo "$w ($h): $(wc -l < joblist_$w.txt) jobs, runners=${n:-?}"
done
