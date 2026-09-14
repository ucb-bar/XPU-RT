#!/usr/bin/env bash
# Launch one worker at a time and VERIFY the runner count afterwards. The ssh that
# detaches the remote xargs sometimes hangs past its timeout even though the launch
# succeeded, so never trust its exit code -- check for exactly 3 python processes.
# (Trusting it last time produced two workers with 6 runners writing to the same
# run dirs.)
#
# Bracketed patterns ('finegrain_ev[a]l') everywhere so a pgrep/pkill can never
# match the command line that is running it. Note the xargs parent and the per-job
# `bash .../finegrain/job.sh` wrappers do NOT contain the string "finegrain_eval",
# so the count is exactly the number of python runners.
set -u
cd "$(dirname "$0")"; source hosts.sh
P=${P:-3}
RUN="cd /home/ubuntu/simpler && setsid nohup bash -c 'xargs -a joblist.txt -I{} -P $P bash sim_eval/roselite/finegrain/job.sh {} > /home/ubuntu/simpler/logs/fine_sweep.log 2>&1; echo ALLDONE >> /home/ubuntu/simpler/logs/fine_sweep.log' >/dev/null 2>&1 </dev/null & disown"
count() { timeout 40 $SSH -n ubuntu@"$1" "pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr -cd '0-9'; }
for w in ${WORKERS:-w0 w1 w2 w3 w4 w5}; do
  h=${HOSTS[$w]}
  pre=$(count "$h")
  if [ "${pre:-0}" != "0" ]; then
    echo "$w ($h): SKIPPED -- ${pre} runner(s) already up, refusing to double-launch"; continue
  fi
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no joblist_$w.txt ubuntu@"$h":/home/ubuntu/simpler/joblist.txt || { echo "$w SCP FAILED"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" "rm -f /home/ubuntu/simpler/logs/fine_sweep.log" >/dev/null 2>&1
  timeout 45 $SSH -n ubuntu@"$h" "$RUN" >/dev/null 2>&1
  n=0
  for _ in 1 2 3 4 5 6; do
    sleep 15
    n=$(count "$h"); [ "${n:-0}" = "$P" ] && break
  done
  flag=""; [ "${n:-0}" = "$P" ] || flag="   <-- EXPECTED $P, CHECK THIS BOX"
  echo "$w ($h): $(wc -l < joblist_$w.txt) jobs, runners=${n:-?}$flag"
done
