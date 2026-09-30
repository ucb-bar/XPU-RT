#!/usr/bin/env bash
# What rate can a REAL ROS 2 chain actually sustain on this board, doing the real work?
#
# The earlier tax run answered two narrower questions: the middleware cost with no compute
# (17.63 ms single-threaded), and what happens at a 45 Hz release rate, which oversubscribes
# 35.57 ms of per-message compute against a 22 ms period and therefore measures unbounded
# backlog rather than latency.
#
# This sweeps the release rate downward and reports end-to-end chain latency at each. The
# rate at which latency stops growing is the rate ROS 2 can actually hold on this board with
# this pipeline -- measured, not modelled. Below the knee, E2E should settle near
# (compute + middleware); above it, it diverges.
#
# Compute per stage is the static-pin arm's own board durations:
#   yolo 30.59 ms, nav 4.90 ms, control 0.08 ms  (sum 35.57 ms)
# so a chain that keeps up cannot beat ~35.6 ms + middleware however it is scheduled.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MODELBLASTER_K1_HOST:-k1}"
OUT="$REPO/results/codesign_feedback/ros_chain_sweep"
LOAD="30.59,4.90,0.08"
mkdir -p "$OUT"
scp -q "$REPO/scripts/ros2_middleware_tax_k1.py" "$HOST:/root/" || exit 1
for HZ in 5 8 10 12 15 20 25; do
  echo "=== $(date +%H:%M:%S) release rate ${HZ} Hz ==="
  ssh -o BatchMode=yes "$HOST" \
    "source /opt/ros/jazzy_prebuilt/setup.bash; python3 /root/ros2_middleware_tax_k1.py \
       --hops 3 --rate $HZ --seconds 15 --executor single --compute-ms $LOAD \
       --out /root/chain_${HZ}.csv" 2>&1 | grep -E 'end-to-end|MIDDLEWARE|no messages'
  scp -q "$HOST:/root/chain_${HZ}.csv" "$OUT/" 2>/dev/null
done
echo "CHAIN_SWEEP_DONE"
