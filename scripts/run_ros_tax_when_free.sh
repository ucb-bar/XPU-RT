#!/usr/bin/env bash
# Run the ROS 2 middleware-tax measurement on the K1 -- BUT ONLY IF THE BOARD IS IDLE.
#
# The board is shared. This benchmark busy-waits by design, so running it next to someone
# else's measurement corrupts both. The idle check is therefore a GATE, not a courtesy, and
# it refuses rather than warns.
#
# NOTE ON THE IDLE CHECK: /proc/loadavg is useless here -- it has a permanent floor of exactly
# 2.00 from two D-state kernel threads (vq0, vq1) that never leave uninterruptible sleep, so
# any check reading it concludes "busy" forever. Read /proc/stat per-CPU instead; an idle
# board is every CPU under a few percent. See docs/K1/k1_board.md.
#
# Usage:  scripts/run_ros_tax_when_free.sh [--force]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MODELBLASTER_K1_HOST:-k1}"
OUT="$REPO/results/codesign_feedback/ros_middleware_tax"
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

echo "=== checking the board is idle (will REFUSE if not) ==="
BUSY=$(ssh -o BatchMode=yes "$HOST" 'ps aux | grep -E "harness|xpurt|python3" | grep -v grep | wc -l' 2>/dev/null)
if [ -z "$BUSY" ]; then echo "cannot reach $HOST -- aborting."; exit 1; fi
echo "  processes matching harness|xpurt|python3: $BUSY"
ssh -o BatchMode=yes "$HOST" 'awk "/^cpu[0-9]/ {u=\$2+\$3+\$4; t=u+\$5; printf \"  %s %.1f%%\n\", \$1, 100*u/t}" /proc/stat'
if [ "$BUSY" -gt 0 ] && [ "$FORCE" -eq 0 ]; then
  echo
  echo "BOARD IS IN USE -- refusing to run. Someone else's timing measurement is on it."
  echo "Re-run with --force ONLY when you know the board is yours."
  exit 2
fi

mkdir -p "$OUT"
scp -q "$REPO/scripts/ros2_middleware_tax_k1.py" "$HOST:/root/" || exit 1

# Per-stage compute = the static-pin arm's own board-recost serial durations, so the loaded
# case burns exactly what the model charges (its 35.578 ms E2E IS this sum -- the model has
# no queueing term at all, which is why any positive tax strictly worsens it).
LOAD="30.59,4.90,0.08"

for EX in single multi; do
  for LABEL in idle loaded; do
    C=0; [ "$LABEL" = loaded ] && C="$LOAD"
    echo
    echo "=== executor=$EX  load=$LABEL ==="
    ssh -o BatchMode=yes "$HOST" \
      "source /opt/ros/jazzy_prebuilt/setup.bash; python3 /root/ros2_middleware_tax_k1.py \
         --hops 3 --rate 45 --seconds 20 --executor $EX --compute-ms $C \
         --out /root/tax_${EX}_${LABEL}.csv" | tee "$OUT/tax_${EX}_${LABEL}.log"
    scp -q "$HOST:/root/tax_${EX}_${LABEL}.csv" "$OUT/" 2>/dev/null
  done
done

echo
echo "=== results in $OUT ==="
echo "Quote the SINGLE-threaded tax as the bound on our model, and report the MULTI-threaded"
echo "number alongside it: multi is the configuration that could BEAT our serial model, so the"
echo "two together bound the model from both sides."
