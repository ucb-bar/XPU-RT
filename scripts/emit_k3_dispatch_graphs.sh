#!/usr/bin/env bash
# Re-emit the SpaceMiT K3 dispatch graphs from ModelBlaster's K3 build IRs.
#
# WHY a script: the graph must come from the SAME IR the profiled binary was
# built from (ids come from extract_graph._annotate_dispatches, the same
# source profile_writer uses), otherwise graph ids and results.csv rows drift.
# Output: gen/mb/vmfb/<model>/spacemit_k3/rvv_k3/<model>.int8/<model>.int8_dispatch_graph.json
# ModelBlaster is only read/run, never edited.
#
# Split IRs (apply_split_hint output): set SUFFIX=split and IR_ROOT=<dir holding
# <model>/int8/graph.json>, e.g.
#   SUFFIX=split IR_ROOT=$MB/build/k3_split MODELS="dronet yolov8_nano" \
#     scripts/emit_k3_dispatch_graphs.sh
# -> gen/mb/vmfb/<m>/spacemit_k3/rvv_k3/<m>.split.int8/<m>.split.int8_dispatch_graph.json
# The graph never carries id_remap (emit_dispatch_graph writes only
# dispatch ids + dependencies; asserted below): the split graph is profiled
# directly, so XPU-RT's profile_loader must not re-key through id_remap.
set -euo pipefail
XPU="$(cd "$(dirname "$0")/.." && pwd)"
MB="${MB:-$XPU/../ModelBlaster}"
PY="${PY:-$MB/.venv/bin/python}"
MODELS="${MODELS:-mlp_control dronet yolov8_nano}"
IR_ROOT="${IR_ROOT:-$MB/build/k3}"
SUFFIX="${SUFFIX:-}"
for m in $MODELS; do
  ir="$IR_ROOT/$m/int8/graph.json"
  [ -f "$ir" ] || { echo "ERROR: missing K3 IR $ir" >&2; exit 1; }
  PYTHONPATH="$MB/src:$MB" "$PY" -m pipeline.emit_dispatch_graph \
    --ir "$ir" --out-root "$XPU/gen/mb/vmfb" \
    --target spacemit_k3 --hw rvv_k3 ${SUFFIX:+--suffix "$SUFFIX"}
  out="$XPU/gen/mb/vmfb/$m/spacemit_k3/rvv_k3/$m${SUFFIX:+.$SUFFIX}.int8/$m${SUFFIX:+.$SUFFIX}.int8_dispatch_graph.json"
  "$PY" - "$out" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))
bad = [k for k in ("id_remap", "split_from") if k in g]
assert not bad, f"{sys.argv[1]} carries {bad}; profile_loader would re-key through it"
PYEOF
done
