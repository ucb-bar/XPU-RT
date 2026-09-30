#!/usr/bin/env bash
# The out-of-the-box graph with the model's 4-hart YOLO and keep-last-1 QoS on its topics.
set -u; cd "$(dirname "$0")/.."
while ssh k1 "ps aux | grep -E 'ros_mb_chain|xpurt' | grep -v grep | wc -l" | grep -qv '^0$'; do sleep 60; done
for rep in 1 2 3; do echo "=== $(date +%H:%M:%S) vanilla4 QoS 1 r$rep"; RATES="45 90" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4 $rep 2>&1 | grep -E "^===|error"; done
.venv/bin/python scripts/pull_ros_traced.py 45_vanilla4_q1_r1 45_vanilla4_q1_r2 45_vanilla4_q1_r3 90_vanilla4_q1_r1 90_vanilla4_q1_r2 90_vanilla4_q1_r3 2>&1 | cut -c1-200
echo VANILLA4_Q1_DONE
