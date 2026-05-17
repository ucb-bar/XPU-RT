#!/usr/bin/env bash
# Wrapper to run after building a new Gemmini config / bitstream.
#
# Pipeline:
#   1. Snapshot the per-config gemmini_params.h into the modelblaster tree.
#      (Implicitly runs chipyard `make verilog CONFIG=<fpga>` which
#      regenerates the header. Skip with --no-elaborate if you've
#      already elaborated.)
#   2. (Optional) Synthesize + place-and-route the bitstream.
#   3. Run the validation matrix (spike-only by default) against the
#      newly-staged config.
#
# Usage:
#   bash scripts/validate_gemmini_config.sh <fpga-config-name> [flags]
#
# Flags:
#   --no-elaborate         skip chipyard elaborate (use cached state)
#   --synth                also run `make synth-only-report` after snapshot
#   --bitstream            also run `make debug-bitstream` after synth
#   --skip-validation      stop after snapshot + (optional) synth/bitstream
#   --verilator            also run verilator validation (slow)
#   --report <path>        write JSON report
#
# Examples:
#   # Snapshot a freshly-changed config and run spike smoke tests:
#   bash scripts/validate_gemmini_config.sh Q31Ws32x32AccGemminiRocketAlinxAxku040DraftConfig
#
#   # Full build + validate (you-just-edited-Chisel flow):
#   bash scripts/validate_gemmini_config.sh \\
#       Q31Ws32x32AccGemminiRocketAlinxAxku040DraftConfig \\
#       --synth --bitstream --report gen/validate_my_change.json
#
#   # Just validate against an already-snapshotted config:
#   bash scripts/validate_gemmini_config.sh \\
#       Q31Ws32x32AccGemminiRocketAlinxAxku040DraftConfig \\
#       --no-elaborate

set -eo pipefail

FPGA_CONFIG="${1:-}"
shift || true

if [[ -z "${FPGA_CONFIG}" ]]; then
    echo "usage: $0 <fpga-config-name> [--no-elaborate] [--synth] [--bitstream]" >&2
    echo "                            [--skip-validation] [--verilator]" >&2
    echo "                            [--report <path>]" >&2
    exit 1
fi

# Locate repo root from this script's path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELBLASTER_ROOT="${REPO_ROOT}/zephyr-chipyard-sw"
CHIPYARD_FSIM="${CHIPYARD_FSIM:-/scratch2/dima/chipyard-fsim}"

# Parse flags.
NO_ELABORATE=""
RUN_SYNTH=""
RUN_BITSTREAM=""
SKIP_VALIDATION=""
RUN_VERILATOR=""
REPORT_PATH=""
while (( $# )); do
    case "$1" in
        --no-elaborate)     NO_ELABORATE=1 ;;
        --synth)            RUN_SYNTH=1 ;;
        --bitstream)        RUN_BITSTREAM=1 ;;
        --skip-validation)  SKIP_VALIDATION=1 ;;
        --verilator)        RUN_VERILATOR=1 ;;
        --report)           shift; REPORT_PATH="${1}" ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

# Derive the chipyard-target sibling config name for the matrix lookup.
# Mapping is in modelblaster/validation/config_matrix.json -- we trust that.
# Returns "" if not found (validation skipped with a hint).
MATRIX_JSON="${MODELBLASTER_ROOT}/modelblaster/validation/config_matrix.json"
CHIPYARD_CONFIG=$(python3 -c "
import json, sys
m = json.load(open('${MATRIX_JSON}'))
for c in m['configs']:
    if c['fpga'] == '${FPGA_CONFIG}':
        print(c['name'])
        break
" 2>/dev/null || echo "")

if [[ -z "${CHIPYARD_CONFIG}" ]]; then
    echo "[validate_gemmini_config] WARNING: ${FPGA_CONFIG} not in config_matrix.json" >&2
    echo "[validate_gemmini_config] add it to ${MATRIX_JSON} to enable matrix validation" >&2
fi

echo "[validate_gemmini_config] FPGA config:     ${FPGA_CONFIG}"
echo "[validate_gemmini_config] Matrix entry:    ${CHIPYARD_CONFIG:-<not in matrix>}"
echo "[validate_gemmini_config] chipyard-fsim:   ${CHIPYARD_FSIM}"
echo

# Step 1: snapshot the header. The script also runs `make verilog`
# unless we pass --no-elaborate.
SNAPSHOT_ARGS=()
[[ -n "${NO_ELABORATE}" ]] && SNAPSHOT_ARGS+=("--no-elaborate")
bash "${MODELBLASTER_ROOT}/modelblaster/scripts/snapshot_gemmini_params.sh" \
    "${FPGA_CONFIG}" "${SNAPSHOT_ARGS[@]}"

# Step 2 (optional): synth + bitstream.
if [[ -n "${RUN_SYNTH}" || -n "${RUN_BITSTREAM}" ]]; then
    if [[ -f "${CHIPYARD_FSIM}/env.sh" ]]; then
        # shellcheck disable=SC1091
        source "${CHIPYARD_FSIM}/env.sh"
    fi
fi
if [[ -n "${RUN_SYNTH}" ]]; then
    echo
    echo "[validate_gemmini_config] make synth-only-report CONFIG=${FPGA_CONFIG}..."
    ( cd "${CHIPYARD_FSIM}/fpga" && \
      make SUB_PROJECT=alinx_axku040 CONFIG="${FPGA_CONFIG}" synth-only-report )
fi
if [[ -n "${RUN_BITSTREAM}" ]]; then
    echo
    echo "[validate_gemmini_config] make debug-bitstream CONFIG=${FPGA_CONFIG}..."
    ( cd "${CHIPYARD_FSIM}/fpga" && \
      make SUB_PROJECT=alinx_axku040 CONFIG="${FPGA_CONFIG}" debug-bitstream )
fi

# Step 3 (optional): run the validation matrix on this config.
if [[ -n "${SKIP_VALIDATION}" || -z "${CHIPYARD_CONFIG}" ]]; then
    echo
    echo "[validate_gemmini_config] skipping validation"
    [[ -z "${CHIPYARD_CONFIG}" ]] && \
        echo "  (no matrix entry; add ${FPGA_CONFIG} to config_matrix.json to enable)"
    exit 0
fi

# Activate zephyr env for the west build. We rely on the user having
# done `source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr`
# and `source scripts/set_envvars_sdk.sh` already; warn if not.
if [[ -z "${ZEPHYR_BASE:-}" ]]; then
    echo
    echo "[validate_gemmini_config] WARNING: ZEPHYR_BASE not set." >&2
    echo "  Run from a shell where you've activated the zephyr env first:" >&2
    echo "    cd ${MODELBLASTER_ROOT} && \\" >&2
    echo "    source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr && \\" >&2
    echo "    source scripts/set_envvars_sdk.sh" >&2
    exit 2
fi

echo
echo "[validate_gemmini_config] running validation matrix on ${CHIPYARD_CONFIG}..."
VALIDATE_ARGS=("--config" "${CHIPYARD_CONFIG}")
if [[ -n "${RUN_VERILATOR}" ]]; then
    VALIDATE_ARGS+=("--verilator")
else
    VALIDATE_ARGS+=("--quick")
fi
[[ -n "${REPORT_PATH}" ]] && VALIDATE_ARGS+=("--report" "${REPORT_PATH}")

( cd "${MODELBLASTER_ROOT}" && \
  python3 -m modelblaster.validation.validate_configs "${VALIDATE_ARGS[@]}" )
