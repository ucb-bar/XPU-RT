#!/usr/bin/env bash
# Re-emit the SpaceMiT K3 dispatch graphs from ModelBlaster's K3 build IRs.
#
# WHY a script: the graph must come from the SAME IR the profiled binary was
# built from (ids come from extract_graph._annotate_dispatches, the same
# source profile_writer uses), otherwise graph ids and results.csv rows drift.
# Output: gen/mb/vmfb/<model>/spacemit_k3/rvv_k3/<model>.int8/<model>.int8_dispatch_graph.json
# ModelBlaster is only read/run, never edited.
set -euo pipefail
XPU="$(cd "$(dirname "$0")/.." && pwd)"
MB="${MB:-$XPU/../ModelBlaster}"
PY="${PY:-$MB/.venv/bin/python}"
MODELS="${MODELS:-mlp_control dronet yolov8_nano}"
for m in $MODELS; do
  ir="$MB/build/k3/$m/int8/graph.json"
  [ -f "$ir" ] || { echo "ERROR: missing K3 IR $ir" >&2; exit 1; }
  PYTHONPATH="$MB/src:$MB" "$PY" -m pipeline.emit_dispatch_graph \
    --ir "$ir" --out-root "$XPU/gen/mb/vmfb" \
    --target spacemit_k3 --hw rvv_k3
done
