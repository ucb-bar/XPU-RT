#!/usr/bin/env bash
# Golden outputs with C-style rounding, for the output check.
# ModelBlaster's export makes each model's golden output with a simulator that rounds add_s8 half-to-even;
# the C kernels round half away from zero, so a correct board output can differ from that golden by one
# step. MB_INT8_GOLDEN_C_FLOAT=1 makes the golden follow the C. This script re-runs ONLY the export (host,
# builder image, no board) once per model, then checks that graph.json and weights.npz are byte-identical
# to a board build's, so the golden belongs to exactly that build.
# usage: golden_c.sh <golden folder> [<build folder of a board job>]      (no build folder: export only)
# Env: K3TL_MODELBLASTER (default <repo>/zephyr-chipyard-sw/modelblaster), K3TL_MODELS (space list), builder.sh's
set -u
OUT="$1"; BUILD="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
K3TL_MB="$(cd "${K3TL_MODELBLASTER:-$REPO/zephyr-chipyard-sw/modelblaster}" && pwd)" || { echo "no ModelBlaster checkout"; exit 9; }
. "$HERE/builder.sh"
md5of() { if command -v md5 >/dev/null 2>&1; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }
mkdir -p "$OUT"; O="$(cd "$OUT" && pwd)"
MODELS="${K3TL_MODELS:-mlp_control dronet yolov8_nano}"
for n in $MODELS; do
  [[ -f "$O/$n/io.npz" ]] && continue
  mkdir -p "$O/$n"
  k3tl_builder "k3tl_golden_${n}_$$" "$O" "" env MODELBLASTER_BOARD=k3 MB_XPURT_WARMUP=1 MB_XPURT_PLACEMENT=1 MB_XPURT_REPEATS=1 \
      MB_XPURT_WORKERS=hart MB_KB_RISCV=0 MODELBLASTER_K3_RT_PRIORITY=80 PYTHONDONTWRITEBYTECODE=1 MB_INT8_GOLDEN_C_FLOAT=1 \
      PYTHONPATH="$K3TL_MB/src:$K3TL_MB" bash -c \
      "cd '$K3TL_MB' && python3 -m modelblaster.pipeline.extract_graph --model $n --quant int8 --out-dir '$O/$n'" \
      > "$O/$n.log" 2>&1 </dev/null || { echo "export failed: $n"; tail -5 "$O/$n.log"; exit 1; }
done
[[ -z "$BUILD" ]] && { echo "exports done (no build given to compare with)"; exit 0; }
bad=0
for n in $MODELS; do
  B="$BUILD/$n/int8"
  g="$([[ "$(md5of "$B/graph.json")" == "$(md5of "$O/$n/graph.json")" ]] && echo same || echo DIFFERENT)"
  w="$([[ "$(md5of "$B/weights.npz")" == "$(md5of "$O/$n/weights.npz")" ]] && echo same || echo DIFFERENT)"
  i="$([[ "$(md5of "$B/io.npz")" == "$(md5of "$O/$n/io.npz")" ]] && echo "same as the build's golden" || echo "differs from the build's golden")"
  echo "$n: graph $g ($(md5of "$B/graph.json")), weights $w, C-rounding golden $i"
  [[ "$g" == same && "$w" == same ]] || bad=1
done
[[ "$bad" == 0 ]] && echo "RESULT: the C-rounding goldens were made from the build's graphs and weights (byte-identical)" || { echo "RESULT: FAIL"; exit 1; }
