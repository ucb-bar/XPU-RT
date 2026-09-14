#!/usr/bin/env bash
# traces_torque3 progress. Counts DISTINCT run KEYS matching the ANCHORED cell
# pattern <task>_<arm>, never a per-host sum of whatever happens to be in
# traces_torque3/ -- an unanchored count once turned an incomplete sweep into
# 889/880. A cell counts as done only when reduce_torque3.py wrote energy2.json.
# The worker side is a pushed script file (probe_tq3.sh), never quoting through
# ssh, and its pgrep pattern is bracketed so it cannot self-match.
set -u; cd "$(dirname "$0")"; source torque3_hosts.sh
TMP=$(mktemp); RUN=0; ALL=0; FAIL=0
for i in $(seq 0 24); do
  w=t$i; h=${HOSTS[$w]}
  timeout 60 scp -q -i "$KEY" -o StrictHostKeyChecking=no probe_tq3.sh ubuntu@"$h":/home/ubuntu/simpler/probe_tq3.sh 2>/dev/null
  r=$(timeout 40 $SSH -n ubuntu@"$h" "bash /home/ubuntu/simpler/probe_tq3.sh" 2>/dev/null)
  echo "$r" | sed -n 's/^KEY //p' | \
    grep -Ex '(egg|spoon|coke|drawer)_(lat0|pipe110fix|p105w300|p130w275|p150w300|pipe200fix|serial283|fp32_555|cpu685)' >> "$TMP"
  d=$(echo "$r" | grep -c '^KEY ')
  p=$(echo "$r" | sed -n 's/^RUN //p'); a=$(echo "$r" | sed -n 's/^ALLDONE //p')
  f=$(echo "$r" | sed -n 's/^FAIL //p')
  RUN=$((RUN+${p:-0})); ALL=$((ALL+${a:-0})); FAIL=$((FAIL+${f:-0}))
  printf "%-4s %-16s done=%-2s runner=%-2s ALLDONE=%-2s FAIL=%s\n" "$w" "$h" "$d" "${p:-?}" "${a:-?}" "${f:-?}"
done
echo "DISTINCT reduced cells: $(sort -u "$TMP" | wc -l)/36   runners=$RUN/25   ALLDONE=$ALL/25   FAIL=$FAIL   $(date +%H:%M:%S)"
sort -u "$TMP" | tr '\n' ' '; echo; rm -f "$TMP"
