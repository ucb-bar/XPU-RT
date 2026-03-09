#!/usr/bin/env bash
# run_all_topologies.sh

set -euo pipefail

############################
# User-configurable values #
############################
#
# These can be overridden via environment variables, which is useful when
# running on a remote machine with a different IREE install layout.

PY_SCRIPT="${PY_SCRIPT:-./run_vmfb_benchmarks.py}"
BENCH_TOOL="${BENCH_TOOL:-./tools/iree-benchmark-module}"

DEVICE="${DEVICE:-local-task}"
INPUT_SPEC="${INPUT_SPEC:-1xi32=1}"
BENCH_REPS="${BENCH_REPS:-10}"

abspath() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$p"
    return
  fi
  if command -v readlink >/dev/null 2>&1; then
    # GNU readlink supports -f; if unsupported, fall back to python.
    readlink -f "$p" 2>/dev/null && return
  fi
  python3 - <<'PY'
import os,sys
print(os.path.realpath(sys.argv[1]))
PY
}

# Extra flags (repeatable). Leave empty if not needed.
EXTRA_FLAGS=(
  # "--benchmark_min_time=0.5"
)

# Topologies to sweep
TOPOLOGIES=(
  "0"
  "0,1"
  "0,1,2"
  "0,1,2,3"
#  "4,5,6,7"
)

################################
# Argument handling             #
################################

if [[ $# -lt 2 ]]; then
  echo "Usage:"
  echo "  $0 <base_out_dir> <input_dir1> [input_dir2 ...]"
  exit 1
fi

BASE_OUT_DIR="$(abspath "$1")"
shift
INPUT_DIRS=("$@")

################################
# Execution                     #
################################

for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  INPUT_DIR="$(abspath "$INPUT_DIR")"
  INPUT_NAME="$(basename "$INPUT_DIR")"

  for TOPO in "${TOPOLOGIES[@]}"; do
    # sanitize: "0,1,2,3" -> "0_1_2_3"
    TOPO_SANITIZED="${TOPO//,/_}"
    TOPO_TAG="topo_${TOPO_SANITIZED}"

    OUT_DIR="${BASE_OUT_DIR}/${INPUT_NAME}/${TOPO_TAG}"

    echo "============================================================"
    echo "Input dir : ${INPUT_DIR}"
    echo "Topology  : ${TOPO}"
    echo "Output dir: ${OUT_DIR}"
    echo "============================================================"

    mkdir -p "${OUT_DIR}"

    CMD=(
      python3 "${PY_SCRIPT}"
      --input_dir "${INPUT_DIR}"
      --out_dir   "${OUT_DIR}"
      --bench_tool "${BENCH_TOOL}"
      --device "${DEVICE}"
      --input_spec "${INPUT_SPEC}"
      --task_topology_cpu_ids "${TOPO}"
      --benchmark_repetitions "${BENCH_REPS}"
      #--glob_vmfb '*.vmfb'
      #--glob_mlir '*.mlir'
    )

    for flag in "${EXTRA_FLAGS[@]}"; do
      CMD+=("--extra_flag=${flag}")
    done

    echo "Running:"
    printf '  %q' "${CMD[@]}"
    echo

    "${CMD[@]}"
    echo
  done
done

echo "All benchmarks completed."

