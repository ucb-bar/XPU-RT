#!/usr/bin/env bash
set -euo pipefail

# Compile a set of Merlin models to VMFB and dump DOT graphs + artifacts.
#
# This is a thin wrapper over `merlin/tools/compile.py`.
#
# Outputs are written by default under:
#   merlin/build/compiled_models/<model>/<target>_<basename>/
# This wrapper overrides it to:
#   gen/vmfb/<model>/<target>/<hw>/<basename>/
#
# Usage:
#   ./runtime/scripts/compile_all_models.sh
#
# Optional env vars:
#   DRY_RUN=1                 Pass --dry-run to compile.py (prints commands only)
#   CONTINUE_ON_ERROR=1       Continue compiling other configs after a failure
#   COMPILER_BUILD_DIR=...    Value for compile.py --build-dir (default: host-vanilla-release)
#   OUT_ROOT=...              Output root (default: <repo>/gen/vmfb)
#   PARSE_DOT=1               Run dot_dispatch_parser.py to produce *_dispatch_graph.json (default: 1)
#   DOT_PNG=1                 Also render *_dispatch_graph.png (default: 0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MERLIN_DIR="${REPO_ROOT}/merlin"

if [[ ! -f "${MERLIN_DIR}/tools/compile.py" ]]; then
  echo "Error: expected Merlin at ${MERLIN_DIR} (missing tools/compile.py)" >&2
  exit 1
fi

# ---- Config lists (edit as needed) ----
MODELS=(
  "models/mlp/mlp.q.int8.mlir"
  "models/dronet/dronet.q.int8.mlir"
)

TARGETS=(
  "spacemit_x60"
)

HWS=(
  "RVV"
  "scalar"
)

DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
COMPILER_BUILD_DIR="${COMPILER_BUILD_DIR:-host-vanilla-release}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/gen/vmfb}"
PARSE_DOT="${PARSE_DOT:-1}"
DOT_PNG="${DOT_PNG:-0}"

DOT_PARSER="${REPO_ROOT}/runtime/scripts/dot_dispatch_parser.py"
if [[ "${PARSE_DOT}" == "1" && ! -f "${DOT_PARSER}" ]]; then
  echo "Error: PARSE_DOT=1 but dot parser not found at ${DOT_PARSER}" >&2
  exit 1
fi

extra_args=(
  "--quantized"
  "--dump-artifacts"
  "--build-benchmarks"
  "--dump-graph"
  "--build-dir" "${COMPILER_BUILD_DIR}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  extra_args+=("--dry-run")
fi

pushd "${MERLIN_DIR}" >/dev/null

failures=0
for target in "${TARGETS[@]}"; do
  for hw in "${HWS[@]}"; do
    for model in "${MODELS[@]}"; do
      if [[ ! -f "${MERLIN_DIR}/${model}" ]]; then
        echo "Error: model file not found: ${MERLIN_DIR}/${model}" >&2
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          failures=$((failures + 1))
          continue
        else
          exit 1
        fi
      fi

      echo "================================================================================"
      echo "Compiling: model=${model} target=${target} hw=${hw}"
      echo "================================================================================"

      model_name="$(basename "$(dirname "${model}")")"
      basename="$(basename "${model}")"
      basename="${basename%.mlir}"
      basename="${basename%.onnx}"
      out_dir="${OUT_ROOT}/${model_name}/${target}/${hw}/${basename}"

      set +e
      python3 tools/compile.py "${model}" \
        --target "${target}" \
        --hw "${hw}" \
        --output-dir "${out_dir}" \
        "${extra_args[@]}"
      rc=$?
      set -e

      if [[ $rc -ne 0 ]]; then
        echo "FAILED (rc=${rc}): model=${model} target=${target} hw=${hw}" >&2
        failures=$((failures + 1))
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
          popd >/dev/null
          exit "${rc}"
        fi
      else
        # Post-process DOT -> JSON (dependency extraction).
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

