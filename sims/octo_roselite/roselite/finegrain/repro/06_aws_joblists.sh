#!/usr/bin/env bash
# usage: 06_aws_joblists.sh <task> <seed_lo> <seed_hi> [arm ...]
#   e.g. 06_aws_joblists.sh spoon 90 99            # all six arms
#        06_aws_joblists.sh drawer 110 119 lat0    # one arm
#
# Build g5fine/joblist_w{0..5}.txt for a (task, seed range) block.  APPENDS to
# any existing joblists so several tasks can be staged into one launch; pass
# FRESH=1 to truncate them first.
#
# THE TWO DESIGN RULES THIS ENCODES -- results are wrong without them:
#
# 1. seed s -> worker (s mod 6), and EVERY ARM OF A SEED RUNS ON THE SAME BOX.
#    A box effect (different GPU) then adds variance to the paired per-seed
#    difference but cannot confound it.  It matters: google_robot is NOT
#    reproducible across GPUs -- `coke lat0 rng80` is 9/24 on the local TITAN RTX
#    and 14/24 on an A10G (measured twice, two different A10G boxes).  A marginal
#    per-arm rate is only comparable to one measured on the same GPU.
#
# 2. Job order within a worker is SHUFFLED with a fixed seed.  The tail of an
#    `xargs -P3` queue runs at lower concurrency than the body; with the arms in
#    a fixed order the same arm always lands in that tail on every box, which
#    correlates an arm with its machine load.
#
# Seeds must also be globally disjoint per (task, arm) -- record whatever you
# issue in g5wide/SEED_ALLOCATION.md before launching, or the pooled analyses
# stop being valid.  Consumed so far: 0-79 coarse, 80-84 google probe (broken,
# quarantined), 90-99 spoon, 100-109 egg, 110-119 drawer, 120-129 coke; 20 is
# reserved and unissued.
#
# Time: instant.  Produces: g5fine/joblist_w{0..5}.txt.
# It worked if the printed per-worker counts sum to (#seeds x #arms).

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ $# -lt 3 ] && { sed -n '2,30p' "$0"; exit 2; }

TASK="$1"; LO="$2"; HI="$3"; shift 3
ARMS="${*:-$ARMS_ALL}"
[ "$(task_id "$TASK")" = BADTASK ] && { echo "bad task '$TASK'"; exit 2; }
for a in $ARMS; do [ "$(arm_lat "$a")" = BADARM ] && { echo "bad arm '$a'"; exit 2; }; done

# The joblists are SHARED STATE in g5fine/ -- another sweep may be staged in them
# right now.  FRESH=1 truncates them, so back them up first and say what was there.
if [ "${FRESH:-0}" = 1 ]; then
  ts=$(date +%Y%m%d-%H%M%S)
  for i in 0 1 2 3 4 5; do
    f="$G5FINE/joblist_w$i.txt"
    [ -s "$f" ] && cp -p "$f" "$f.bak-$ts"
    : > "$f"
  done
  if ls "$G5FINE"/joblist_w*.txt.bak-"$ts" >/dev/null 2>&1; then
    echo "FRESH=1: truncated all six joblists; previous contents saved as joblist_w*.txt.bak-$ts"
    echo "         they held: $(cat "$G5FINE"/joblist_w*.txt.bak-"$ts" | cut -d: -f1,2 | sort -u | tr '\n' ' ')"
  else
    echo "FRESH=1: joblists were already empty"
  fi
fi

for s in $(seq "$LO" "$HI"); do
  w=$(( s % 6 ))
  for a in $ARMS; do echo "$TASK:$a:$s" >> "$G5FINE/joblist_w$w.txt"; done
done

# Shuffle each list with a FIXED seed so the layout is reproducible but no arm
# is systematically in the low-concurrency tail.
for i in 0 1 2 3 4 5; do
  f="$G5FINE/joblist_w$i.txt"
  [ -s "$f" ] || { : > "$f"; continue; }
  shuf --random-source=<(yes 424242) "$f" -o "$f.tmp" && mv "$f.tmp" "$f"
done

tot=0
for i in 0 1 2 3 4 5; do
  n=$(wc -l < "$G5FINE/joblist_w$i.txt"); tot=$((tot + n))
  printf '  w%s  %3d jobs\n' "$i" "$n"
done
echo "  TOTAL $tot jobs  (this call added $(( (HI-LO+1) * $(echo $ARMS | wc -w) )))"
echo "next: repro/07_aws_launch.sh"
