#!/usr/bin/env bash
# Grid sweep progress. Counts ONLY g<P>_<W> run dirs -- the AMI ships 78 stale
# dirs from the earlier 9-arm study. The pgrep pattern is bracketed AND must not
# contain the literal eval script path, or the remote shell self-matches.
set -u; cd "$(dirname "$0")"; source hosts_grid.sh
R=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs
tot=0; run=0; stuck=""
for i in $(seq 0 24); do
  w=n$i; h=${HOSTS[$w]}
  r=$(timeout 30 $SSH -n ubuntu@"$h" \
      "ls -d $R/*_g[0-9]*_rng*/summary.json 2>/dev/null | wc -l; pgrep -f 'finegrain_ev[a]l' | wc -l; grep -c ALLDONE /home/ubuntu/simpler/logs/grid_sweep.log 2>/dev/null || echo 0" 2>/dev/null | tr '\n' ' ')
  set -- ${r:-? ? ?}
  d=${1:-?}; p=${2:-?}; a=${3:-0}
  [ "$d" != "?" ] && tot=$((tot+d)) && run=$((run+p))
  [ "${p:-0}" = "0" ] && [ "${a:-0}" = "0" ] && stuck="$stuck $w"
  printf "%-4s %-16s done=%-3s runners=%-2s ALLDONE=%s\n" "$w" "$h" "$d" "$p" "$a"
done
echo "TOTAL summaries=$tot/1170  runners=$run/75  idle-but-not-finished:${stuck:- none}   $(date +%H:%M:%S)"
