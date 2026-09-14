#!/usr/bin/env bash
# Worker-side status probe for the traces_torque3 sweep. Pushed by scp and run
# bare, so nothing is quoted through ssh. Emits PREFIXED lines only.
F=/home/ubuntu/simpler/sim_eval/roselite/finegrain
for f in "$F"/traces_torque3/*/energy2.json; do
  [ -e "$f" ] || continue
  d=$(dirname "$f"); echo "KEY $(basename "$d")"
done
# bracketed so this script's own command line cannot self-match
echo "RUN $(pgrep -f 'trace_eva[l]2' | wc -l)"
n=0
[ -f /home/ubuntu/simpler/logs/tq3_sweep.log ] && n=$(grep -c ALLDONE /home/ubuntu/simpler/logs/tq3_sweep.log)
echo "ALLDONE $n"
echo "FAIL $(grep -c '^FAIL' /home/ubuntu/simpler/logs/tq3_sweep.log 2>/dev/null | head -1)"
