#!/usr/bin/env bash
set -euo pipefail

# Compile detached SmolVLA sub-graph MLIR files (from ONNX export) via Merlin's compile driver.
#
# Default compile set: smolvlm_expert_decode + smolvlm_expert_prefill only.
# Full set (all canonical *.mlir under SOURCE_DIR): SUBMODEL_GLOB=all
# Custom subset: SUBMODEL_GLOB="smolvlm_text state_projector"
#
# Canonical names (when SUBMODEL_GLOB=all) include:
#   action_in/out_projector, smolvlm_expert_{decode,prefill}, smolvlm_{text,vision},
#   state_projector, time_in/out_projector
#
# Each sub-model is compiled into its own directory under OUT_BASE (default below).
# Merlin copies the source MLIR into that folder, then iree-compile runs on the copy;
# --dump-artifacts places per-dispatch MLIR under .../<basename>/sources/.
#
# Layout:
#   One target + one hw (defaults):  <OUT_BASE>/<basename>/
#   Multiple targets and/or hws:     <OUT_BASE>/<basename>/<target>/<hw>/
#
# Usage:
#   ./runtime/scripts/compile_sub_models.sh
#
# Optional env vars:
#   SOURCE_DIR=...            Directory containing *.mlir (default: /scratch2/kris/smolvla_base_onnx)
#   OUT_BASE=...              Root for all sub-model dirs (default: <REPO>/gen/vmfb/smolVLA-new)
#   SUBMODEL_GLOB=...         Space-separated stems to compile. Unset → default: expert_decode +
#                             expert_prefill only. Use SUBMODEL_GLOB=all for every canonical + extras.
#   CONTINUE_ON_ERROR=...     Default 1 here so one failed compile still runs the rest (each gets a folder).
#                             Set to 0 to stop on first failure.
#   DRY_RUN, IREE_COMPILE_BIN, MERLIN_TOOL_BUILD_DIR, MERLIN_DIR — same as compile_all_models.sh
#   PARSE_DOT, DOT_PNG
#   COMPILE_SUB_TARGETS=...   Space-separated targets (default: spacemit_x60)
#   COMPILE_SUB_HWS=...       Space-separated hw variants (default: scalar)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_DIR="${SOURCE_DIR:-/scratch2/kris/smolvla_base_onnx}"
SOURCE_DIR="$(cd "${SOURCE_DIR}" && pwd)"

OUT_BASE="${OUT_BASE:-${REPO_ROOT}/gen/vmfb/smolVLA-new}"
mkdir -p "${OUT_BASE}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Error: SOURCE_DIR is not a directory: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -n "${MERLIN_DIR:-}" ]]; then
  MERLIN_DIR="$(cd "${MERLIN_DIR}" && pwd)"
elif [[ -f "${REPO_ROOT}/../merlin/tools/merlin.py" ]]; then
  MERLIN_DIR="$(cd "${REPO_ROOT}/../merlin" && pwd)"
else
  MERLIN_DIR="${REPO_ROOT}/merlin"
fi

if [[ ! -f "${MERLIN_DIR}/tools/merlin.py" ]]; then
  echo "Error: expected Merlin at ${MERLIN_DIR} (missing tools/merlin.py)" >&2
  exit 1
fi

# shellcheck disable=SC2206
if [[ -n "${COMPILE_SUB_TARGETS:-}" ]]; then
  TARGET_ARR=(${COMPILE_SUB_TARGETS})
else
  TARGET_ARR=(spacemit_x60)
fi
# shellcheck disable=SC2206
if [[ -n "${COMPILE_SUB_HWS:-}" ]]; then
  HW_ARR=(${COMPILE_SUB_HWS})
else
  HW_ARR=(scalar)
fi

DRY_RUN="${DRY_RUN:-0}"
# Default 1: every canonical sub-model gets an output dir and a compile attempt; do not stop at first iree failure.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MERLIN_TOOL_BUILD_DIR="${MERLIN_TOOL_BUILD_DIR:-host-vanilla-debug}"

# Fixed order when SUBMODEL_GLOB=all (or when filtering, only listed stems that appear here are candidates).
CANONICAL_SUBMODEL_STEMS=(
  action_in_projector
  action_out_projector
  smolvlm_expert_decode
  smolvlm_expert_prefill
  smolvlm_text
  smolvlm_vision
  state_projector
  time_in_projector
  time_out_projector
)

if [[ -n "${MERLIN_IREE_COMPILE:-}" ]] && [[ ! -f "${MERLIN_IREE_COMPILE}" ]]; then
  case "${MERLIN_IREE_COMPILE}" in
    */* | ./*) ;;
    *) unset MERLIN_IREE_COMPILE ;;
  esac
fi
IREE_COMPILE_BIN="${IREE_COMPILE_BIN:-${MERLIN_DIR}/build/host-vanilla-debug/install/bin/iree-compile}"
if [[ -f "${IREE_COMPILE_BIN}" ]]; then
  export MERLIN_IREE_COMPILE="${IREE_COMPILE_BIN}"
fi

PARSE_DOT="${PARSE_DOT:-1}"
DOT_PNG="${DOT_PNG:-0}"

DOT_PARSER="${REPO_ROOT}/runtime/scripts/dot_dispatch_parser.py"
if [[ "${PARSE_DOT}" == "1" && ! -f "${DOT_PARSER}" ]]; then
  echo "Error: PARSE_DOT=1 but dot parser not found at ${DOT_PARSER}" >&2
  exit 1
fi

extra_args=(
  # "--dump-artifacts"
  # "--build-benchmarks"
  # "--dump-graph"
  "--build-dir" "${MERLIN_TOOL_BUILD_DIR}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  extra_args+=("--dry-run")
fi

# Default: only the two expert graphs. SUBMODEL_GLOB=all → every canonical stem (+ extras below).
_submodel_pick_extras=0
if [[ "${SUBMODEL_GLOB:-}" == "all" ]]; then
  SUBMODEL_GLOB=""
  _submodel_pick_extras=1
elif [[ -z "${SUBMODEL_GLOB:-}" ]]; then
  SUBMODEL_GLOB="smolvlm_expert_decode smolvlm_expert_prefill"
fi

mlir_files=()
if [[ -n "${SUBMODEL_GLOB}" ]]; then
  # shellcheck disable=SC2206
  _want=(${SUBMODEL_GLOB})
  declare -A _allow=()
  for w in "${_want[@]}"; do
    _allow["${w%.mlir}"]=1
  done
fi

for stem in "${CANONICAL_SUBMODEL_STEMS[@]}"; do
  if [[ -n "${SUBMODEL_GLOB}" ]] && [[ -z "${_allow[$stem]:-}" ]]; then
    continue
  fi
  f="${SOURCE_DIR}/${stem}.mlir"
  if [[ -f "${f}" ]]; then
    mlir_files+=("${f}")
  else
    echo "Warning: canonical sub-model MLIR missing (skipped): ${f}" >&2
  fi
done

if [[ -n "${SUBMODEL_GLOB}" ]]; then
  unset _allow
fi

# Extra *.mlir not in CANONICAL list: only when SUBMODEL_GLOB=all.
if [[ "${_submodel_pick_extras}" == "1" ]]; then
  shopt -s nullglob
  _extra=("${SOURCE_DIR}"/*.mlir)
  shopt -u nullglob
  declare -A _seen=()
  for f in "${mlir_files[@]}"; do
    _seen["$(basename "${f}")"]=1
  done
  for f in "${_extra[@]}"; do
    b="$(basename "${f}")"
    if [[ -z "${_seen[$b]:-}" ]]; then
      echo "Note: compiling non-canonical MLIR (not in fixed list): ${f}" >&2
      mlir_files+=("${f}")
      _seen["${b}"]=1
    fi
  done
  unset _seen
fi

if [[ ${#mlir_files[@]} -eq 0 ]]; then
  echo "Error: no sub-model MLIR files found under ${SOURCE_DIR}" >&2
  exit 1
fi

_flat_out="$([[ ${#TARGET_ARR[@]} -eq 1 && ${#HW_ARR[@]} -eq 1 ]] && echo 1 || echo 0)"

echo "Sub-model compile: SOURCE_DIR=${SOURCE_DIR}"
echo "Output base:       ${OUT_BASE}"
if [[ "${_submodel_pick_extras}" == "1" ]]; then
  echo "Selection:         SUBMODEL_GLOB=all (canonical + any extra *.mlir)"
else
  echo "Selection:         SUBMODEL_GLOB=${SUBMODEL_GLOB}"
fi
echo "MLIR files:        ${#mlir_files[@]}"
echo "Per-model layout:  $([[ "${_flat_out}" == 1 ]] && echo "${OUT_BASE}/<basename>/" || echo "${OUT_BASE}/<basename>/<target>/<hw>/")"

pushd "${MERLIN_DIR}" >/dev/null

failures=0
for target in "${TARGET_ARR[@]}"; do
  for hw in "${HW_ARR[@]}"; do
    for _src_mlir in "${mlir_files[@]}"; do
      _base="$(basename "${_src_mlir}")"
      basename="${_base%.mlir}"
      if [[ "${_flat_out}" == 1 ]]; then
        out_dir="${OUT_BASE}/${basename}"
      else
        out_dir="${OUT_BASE}/${basename}/${target}/${hw}"
      fi
      mkdir -p "${out_dir}"

      echo "================================================================================"
      echo "Compiling: target=${target} hw=${hw}"
      echo "  source MLIR: ${_src_mlir}"
      echo "  output dir:  ${out_dir}"
      echo "================================================================================"

      set +e
      uv run tools/merlin.py compile "${_src_mlir}" \
        --target "${target}" \
        --hw "${hw}" \
        --output-dir "${out_dir}" \
        "${extra_args[@]}"
      rc=$?
      set -e

      if [[ $rc -ne 0 ]]; then
        echo "FAILED (rc=${rc}): target=${target} hw=${hw} source=${_src_mlir}" >&2
        failures=$((failures + 1))
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          popd >/dev/null
          exit "${rc}"
        fi
      else
        if [[ "${PARSE_DOT}" == "1" ]]; then
          dot_path="${out_dir}/${basename}_dispatch_graph.dot"
          json_out="${out_dir}/${basename}_dispatch_graph.json"
          png_out="${out_dir}/${basename}_dispatch_graph.png"

          if [[ -f "${dot_path}" ]]; then
            parser_args=( "${dot_path}" "--json-out" "${json_out}" )
            if [[ "${DOT_PNG}" == "1" ]]; then
              parser_args+=( "--png" "${png_out}" )
            fi

            set +e
            python3 "${DOT_PARSER}" "${parser_args[@]}"
            prc=$?
            set -e

            if [[ $prc -ne 0 ]]; then
              echo "Warning: DOT parse failed (rc=${prc}): ${dot_path}" >&2
              failures=$((failures + 1))
              if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
                popd >/dev/null
                exit "${prc}"
              fi
            fi
          else
            echo "Warning: DOT file not found (skipping parse): ${dot_path}" >&2
          fi
        fi
      fi
    done
  done
done

popd >/dev/null

if [[ "${failures}" -ne 0 ]]; then
  echo "Done with failures: ${failures}" >&2
  exit 2
fi

echo "Done."
