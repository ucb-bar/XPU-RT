#!/usr/bin/env bash
# One health line per pass. Emits only when something is worth acting on, plus a
# summary line every pass. Exits when all 25 workers report ALLDONE.
#
# Field parsing is delimited (key=value), NOT positional: `pgrep -c` and
# `grep -c` both PRINT "0" and RETURN 1 on no-match, so a positional
# "cmd || echo 0" emits TWO lines and silently shifts every later field. That
# bug made an earlier pass report 15 GB of swap use on an 8 GB swapfile.
set -u; cd "$(dirname "$0")"; source hosts_grid.sh
REMOTE='R=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs
s=$(ls -d $R/*_g[0-9]*_rng*/summary.json 2>/dev/null | wc -l)
p=$(pgrep -cf "[f]inegrain_eval.py"); p=${p:-0}
a=$(grep -c ALLDONE /home/ubuntu/simpler/logs/grid_sweep.log 2>/dev/null); a=${a:-0}
d=$(df / | tail -1 | awk "{print \$5}" | tr -d %)
m=$(free -m | awk "/Mem:/{print \$7}")
w=$(free -m | awk "/Swap:/{print \$3}")
echo "sum=$s run=$p done=$a disk=$d avail=$m swap=$w"'
while :; do
  tot=0; run=0; dn=0; bad=""; maxd=0; minav=99999999; launched=0
  for i in $(seq 0 24); do
    r=$(timeout 40 $SSH -n ubuntu@"${HOSTS[n$i]}" "$REMOTE" 2>/dev/null)
    [ -z "$r" ] && { bad="$bad n$i:UNREACH"; continue; }
    s=${r#*sum=}; s=${s%% *}
    p=${r#*run=}; p=${p%% *}
    a=${r#*done=}; a=${a%% *}
    d=${r#*disk=}; d=${d%% *}
    m=${r#*avail=}; m=${m%% *}
    w=${r#*swap=}; w=${w%% *}
    tot=$((tot+s)); run=$((run+p)); [ "$a" -ge 1 ] && dn=$((dn+1))
    [ "$p" -gt 0 ] || [ "$a" -ge 1 ] && launched=$((launched+1))
    [ "$d" -gt "$maxd" ] && maxd=$d
    [ "$m" -lt "$minav" ] && minav=$m
    [ "$d" -ge 88 ]   && bad="$bad n$i:DISK${d}%"
    [ "$p" -gt 3 ]    && bad="$bad n$i:RUNNERS=$p"
    [ "$m" -lt 250 ]  && bad="$bad n$i:LOWRAM${m}M"
    [ "$w" -gt 4000 ] && bad="$bad n$i:SWAPTHRASH${w}M"
    # only a worker that was launched and has since gone quiet is a real fault
    [ "$p" -eq 0 ] && [ "$a" -eq 0 ] && [ "$s" -gt 0 ] && bad="$bad n$i:STALLED"
  done
  echo "$(date +%H:%M:%S) summaries=$tot/1170 runners=$run launched=$launched/25 finished=$dn/25 maxdisk=${maxd}% minRAM=${minav}M${bad:+  PROBLEM:$bad}"
  [ "$dn" -ge 25 ] && break
  sleep 900
done
echo "ALL 25 WORKERS FINISHED $(date +%H:%M:%S)"
