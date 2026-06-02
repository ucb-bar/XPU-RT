#!/usr/bin/env bash
# One-command predicted-only demo of the iterative scheduling-improvement loop on
# the FireSim 1-yolo + 4-mlp + 2-dronet workload. No FireSim/spike needed: it
# schedules with existing profiles, lets the advisor diagnose, tries a bundle of
# candidates (scheduler + backend axes), picks a winner, and writes a report +
# before/after composite Gantt. See docs/iterative_firesim_loop.md.
set -euo pipefail
cd "$(dirname "$0")/.."

SPEC=data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json
OUT=artifacts/iterate

echo "### Iterative FireSim scheduling demo (predicted-only) ###"
echo "workload: 1 yolov8_nano + 4 mlp_control + 2 dronet"
echo

python3 scripts/iterate_firesim.py \
  --networks-json "$SPEC" \
  --baseline-solver decomposed \
  --deadline-us auto \
  --timeout 70 \
  --gantt \
  --out-dir "$OUT"

echo
echo "### Artifacts ###"
echo "  report:        $OUT/report.md"
echo "  result json:   $OUT/iteration_result.json"
echo "  firesim batch: $OUT/firesim_batch.json   (hand to the ModelBlaster session)"
echo "  before/after:  $OUT/before_after_gantt.png"
echo
echo "Optional axis-B deep dive:"
echo "  python3 scripts/compare_backends.py --networks-json $SPEC --solver greedy --deadline-us 70"
