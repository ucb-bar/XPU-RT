#!/usr/bin/env bash
set -euo pipefail

# Compile a set of Merlin models to VMFB and dump DOT graphs + artifacts.
##
# Outputs are written by default under:
#   merlin/build/compiled_models/<model>/<target>_<basename>/
# This wrapper overrides it to:
#   gen/vmfb/<model>/<target>/<hw>/<basename>/
# Merlin copies the source .mlir there, then iree-compile runs on that copy; compiler errors under
# .../sources/ are artifact paths, not the path under merlin/models/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Default Merlin root: sibling of XPU-RT (e.g. .../kris/merlin next to .../kris/XPU-RT).
# Override with MERLIN_DIR if Merlin lives elsewhere (e.g. nested XPU-RT/merlin).
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



# K1 bring-up set. smolVLA is deliberately last per the plan -- it must not
# block the three-model baseline.
MODELS=(
  "models/mlp/mlp.q.int8.mlir"
  "models/dronet/dronet.q.int8.mlir"
  # "models/smolVLA/smolVLA.mlir"
  # "models/smolVLA/smolVLA.q.int8.mlir"
)
TARGETS=(
  "spacemit_x60"
)

# Both clusters run scalar and RVV; IME is added once its variant exists in
# merlin/models/spacemit_x60.yaml (cluster 0 only -- it SIGILLs on cluster 1).
HWS=(
  "RVV"
  "scalar"
)

DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
MERLIN_TOOL_BUILD_DIR="${MERLIN_TOOL_BUILD_DIR:-host-vanilla-release}"

# Prefer a real iree-compile path. Drop bogus MERLIN_IREE_COMPILE (e.g. host-merlin-release from old env).
if [[ -n "${MERLIN_IREE_COMPILE:-}" ]] && [[ ! -f "${MERLIN_IREE_COMPILE}" ]]; then
  case "${MERLIN_IREE_COMPILE}" in
    */* | ./*) ;;
    *) unset MERLIN_IREE_COMPILE ;;
  esac
fi
IREE_COMPILE_BIN="${IREE_COMPILE_BIN:-${MERLIN_DIR}/build/${MERLIN_TOOL_BUILD_DIR}/install/bin/iree-compile}"
if [[ -f "${IREE_COMPILE_BIN}" ]]; then
  export MERLIN_IREE_COMPILE="${IREE_COMPILE_BIN}"
fi

# Keep outputs under this repo (see header); override OUT_ROOT only if you want another root.
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/gen/vmfb}"
PARSE_DOT="${PARSE_DOT:-1}"
DOT_PNG="${DOT_PNG:-0}"

DOT_PARSER="${REPO_ROOT}/runtime/scripts/dot_dispatch_parser.py"
if [[ "${PARSE_DOT}" == "1" && ! -f "${DOT_PARSER}" ]]; then
  echo "Error: PARSE_DOT=1 but dot parser not found at ${DOT_PARSER}" >&2
  exit 1
fi

# --dump-graph is required: the .dot it emits is what dot_dispatch_parser.py
# turns into the *_dispatch_graph.json the scheduler consumes.
# --build-benchmarks produces the *_benchmarks.zip that profile_remote.sh
# stages to the board for per-dispatch timing.
extra_args=(
  # "--quantized"
  #"--dump-artifacts"
  "--build-benchmarks"
  "--dump-graph"
  "--build-dir" "${MERLIN_TOOL_BUILD_DIR}"
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

      _src_mlir="${MERLIN_DIR}/${model}"
      model_name="$(basename "$(dirname "${model}")")"
      basename="$(basename "${model}")"
      basename="${basename%.mlir}"
      basename="${basename%.onnx}"
      out_dir="${OUT_ROOT}/${model_name}/${target}/${hw}/${basename}"

      set +e
      uv run tools/merlin.py compile "${model}" \
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

