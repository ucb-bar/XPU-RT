#!/usr/bin/env bash
# Dispatch an arbitrary joblist (TASK:ARM:SEED per line) across the phase-3 fleet.
#
# Two traps this script exists to avoid:
#  1. EC2 public IPs change on stop/start, so hosts come from phase3_hosts.sh (current),
#     never hosts_grid.sh (stale).  Reachability is checked before anything is pushed.
#  2. The verify pgrep must NOT contain the literal finegrain_eval.py anywhere in its own
#     command line, or the remote shell self-matches and reports a phantom runner.  The
#     bracketed pattern 'finegrain_ev[a]l' is the whole point.
#
# Usage: ./launch_extra.sh <joblist.txt> [tag]
set -u
cd "$(dirname "$0")"; source phase3_hosts.sh
JOBS=${1:?usage: launch_extra.sh <joblist.txt> [tag]}
TAG=${2:-extra}
KEY=$HOME/.ssh/firesim.pem
N=${#P3[@]}
total=$(wc -l < "$JOBS")
echo "dispatching $total jobs over $N workers (tag=$TAG)"

# --- split round-robin -------------------------------------------------------
rm -f "jl_${TAG}_"*.txt
awk -v n="$N" -v t="$TAG" '{print > ("jl_" t "_" (NR-1)%n ".txt")}' "$JOBS"

RUN="cd ../simpler/ && setsid nohup bash -c 'xargs -a joblist_${TAG}.txt -I{} -P 3 bash sim_eval/roselite/finegrain/job_grid.sh {} > ../simpler/logs/${TAG}.log 2>&1; echo ALLDONE >> ../simpler/logs/${TAG}.log' >/dev/null 2>&1 </dev/null & disown"

for i in $(seq 0 $((N-1))); do
  h=${P3[$i]}; f="jl_${TAG}_${i}.txt"
  [ -s "$f" ] || { echo "w$i ($h): no jobs"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" true >/dev/null 2>&1 || { echo "w$i ($h): UNREACHABLE -- skipped"; continue; }
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no "$f" ubuntu@"$h":../simpler/joblist_${TAG}.txt || { echo "w$i ($h): SCP-JOBLIST FAILED"; continue; }
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no job_grid.sh ubuntu@"$h":../simpler/sim_eval/roselite/finegrain/job_grid.sh || { echo "w$i ($h): SCP-JOB FAILED"; continue; }
  timeout 45 $SSH -n ubuntu@"$h" "mkdir -p ../simpler/logs ../simpler/sim_eval/roselite/finegrain/{logs,runs}; rm -f ../simpler/logs/${TAG}.log; chmod +x ../simpler/sim_eval/roselite/finegrain/job_grid.sh" >/dev/null 2>&1
  timeout 45 $SSH -n ubuntu@"$h" "$RUN" >/dev/null 2>&1
  sleep 6
  n=$(timeout 40 $SSH -n ubuntu@"$h" "pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr -cd '0-9')
  echo "w$i ($h): $(wc -l < "$f") jobs, runners=${n:-?}"
done
