#!/usr/bin/env bash
# Build one XPU-RT schedule with ModelBlaster and run it on the SpaceMiT K3; copy
# the walker's stdout and trace (and the schedule) into OUT_DIR.
#
#   run_schedule_on_k3.sh <schedule.json> <out_dir> [VAR=value ...]
#
# VAR=value pairs select the walker, e.g.
#   MB_XPURT_DYNAMIC=0                                  static walker (default)
#   MB_XPURT_DYNAMIC=1 MB_XPURT_SAFETY_BOUNDARY=1.0     dynamic walker
#   MB_XPURT_POOL=yolov8_nano:15.625                    global pool for yolov8_nano
#
# Env:
#   MODELBLASTER_ROOT  ModelBlaster checkout (default: ../ModelBlaster next to XPU-RT)
#   PY                 ModelBlaster's python (default: $MODELBLASTER_ROOT/.venv/bin/python)
#   MODELBLASTER_K3_HOST  ssh alias of the K3 (default k3)
#   K3_LOCK_HELD=1     the caller already holds the board locks (k3_board_lock.sh);
#                      otherwise this script takes them for its own run
# The models' int8 IRs must already exist under $MODELBLASTER_ROOT/build/k3_xpurt
# (extracting them needs PyTorch); see README.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPURT_ROOT="$(cd "${here}/../.." && pwd)"
SCHED="$(realpath "$1")"; OUT="$(realpath -m "$2")"; shift 2
MB="${MODELBLASTER_ROOT:-${XPURT_ROOT}/../ModelBlaster}"
MB="$(realpath "${MB}")"
export MODELBLASTER_K3_HOST="${MODELBLASTER_K3_HOST:-k3}"
export PY="${PY:-${MB}/.venv/bin/python}" PATH="$(dirname "${PY}"):${PATH}"
CROSS="$("${XPURT_ROOT}/scripts/setup_spacemit_toolchain.sh" --path 2>/dev/null)"
export CROSS
mkdir -p "${OUT}"

if [[ "${K3_LOCK_HELD:-0}" != "1" ]]; then
    # shellcheck source=k3_board_lock.sh
    source "${here}/k3_board_lock.sh"
    k3_lock_acquire
fi

N="$(basename "${SCHED}" .json)"
( cd "${MB}" && env "$@" bash scripts/run_xpurt_k3.sh --schedule "${SCHED}" \
      --models mlp_control,dronet,yolov8_nano ) > "${OUT}/board_run.log" 2>&1
G="${MB}/build/k3_xpurt/_gen/${N}"
cp "${G}/${N}_stdout.txt" "${G}/${N}_trace.csv" "${OUT}/"
cp "${SCHED}" "${OUT}/"
echo "${OUT}: $(wc -l < "${OUT}/${N}_trace.csv") trace rows"
