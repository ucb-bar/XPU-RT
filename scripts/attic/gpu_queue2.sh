#!/usr/bin/env bash
# After the main GPU queue: the RoSE-style pipeline injection (control cadence + chain delay per arm).
set -u; cd "$(dirname "$0")/.."
while ! grep -q "GPU_QUEUE_DONE" results/codesign_feedback/campaign/gpu_queue.log 2>/dev/null; do sleep 120; done
echo "=== $(date +%H:%M:%S) pipeline campaign"; python3 scripts/campaign_pipeline.py
echo "=== $(date +%H:%M:%S) GPU_QUEUE2_DONE"
