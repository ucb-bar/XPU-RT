#!/usr/bin/env bash
# Build + profile the size variants that QRB5165 has no cells for.
#
# Every board step takes the SAME lock deploy_and_run.sh uses, so this
# serialises against the other agent's sweeps rather than colliding with them.
# Run it with `flock` held for the whole sequence, not per step, so a long
# build cannot be preempted halfway:
#
#     flock -w 43200 /tmp/qnn_board.lock bash results/flowc_size/board_plan.sh
#
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PYTHON:=.venv/bin/python}"
: "${QNN_BOARD_HOST:=root@10.44.120.201}"

for V in yolov8_nano_64x96 yolov8_nano_128x192; do
  echo "=== $V ==="
  # 1. slice/build the variant for QNN (needs the QNN SDK container)
  $PYTHON qnn_models/slicing_study/slice_experiment.py \
      --network "$V" --backends dsp,cpu,hta --precisions int8,fp32 \
      --journal qnn_models/slicing_study/experiments.jsonl
  # 2. emit XPURT profile cells from the measured perf json
  $PYTHON qnn_models/smolVLA/emit_vision_v3_profile.py \
      --from-perf-json "qnn_models/boards/qrb5165_v66/profiles/$V/segment_perf.json" \
      || echo "  (emit step needs the per-variant perf json; see RESULTS.md 4)"
done

echo "=== re-run the size sweep with the new cells ==="
$PYTHON scripts/flowc_size_sweep.py --out results/flowc_size
