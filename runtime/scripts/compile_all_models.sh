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

# Both clusters run scalar and RVV. IME is cluster 0 only -- it SIGILLs on
# cluster 1 (measured, artifacts/k1_bringup/*/ime_capability_probe.txt) -- and
# its variant now exists in merlin/models/spacemit_x60.yaml, so it is built
# here. The label is not taken on trust: IME_GATE below disassembles the result
# and fails the build if no smt.vmadot reached the machine code.
HWS=(
  "RVV"
  "scalar"
  "IME"
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
# --dump-artifacts is required by the IME gate: without it no .o/.so is emitted
# and there is nothing to disassemble, which is why the gate had never actually
# run against a real build.
extra_args=(
  # "--quantized"
  "--dump-artifacts"
  "--build-benchmarks"
  "--dump-graph"
  "--build-dir" "${MERLIN_TOOL_BUILD_DIR}"
)

# IME acceptance gate. 1 = disassemble every IME build and record the vmadot
# count; 0 = skip. Per-model expectations, because a legitimate zero exists:
# MLP's matmuls are all 1xNxK (M=1, i.e. GEMV) and a matrix engine has nothing
# to bite on, so "0 vmadot for mlp" is a property of the model rather than a
# build failure. Anything not listed defaults to requiring at least one.
IME_GATE="${IME_GATE:-1}"
IME_GATE_DIR="${IME_GATE_DIR:-${REPO_ROOT}/artifacts/k1_run/ime_gate}"
ime_expect_zero_models=("mlp")
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
        # IME gate: an "IME" build that silently lowered to plain RVV must be
        # rejected, not profiled as IME performance.
        if [[ "${hw}" == IME* && "${IME_GATE}" == "1" && "${DRY_RUN}" != "1" ]]; then
          mkdir -p "${IME_GATE_DIR}"
          gate_log="${IME_GATE_DIR}/${model_name}_${target}_${hw}.txt"
          expect_zero=0
          for m in "${ime_expect_zero_models[@]}"; do
            [[ "${model_name}" == "${m}" ]] && expect_zero=1
          done

          set +e
          bash "${REPO_ROOT}/runtime/scripts/verify_ime_build.sh" "${out_dir}" \
            >"${gate_log}" 2>&1
          grc=$?
          set -e
          n_vmadot="$(grep -oE 'vmadot[ =:]+[0-9]+' "${gate_log}" | grep -oE '[0-9]+' \
                      | paste -sd+ - | bc 2>/dev/null || echo 0)"
          n_vmadot="${n_vmadot:-0}"
          {
            echo ""
            echo "gate: model=${model_name} hw=${hw} expect_zero=${expect_zero} vmadot_total=${n_vmadot}"
          } >>"${gate_log}"

          if [[ "${expect_zero}" == "1" ]]; then
            echo "IME gate: ${model_name} vmadot=${n_vmadot} (zero expected: all matmuls are GEMV) -> ${gate_log}"
          elif [[ $grc -ne 0 || "${n_vmadot}" == "0" ]]; then
            echo "IME GATE FAILED: ${model_name}/${hw} produced no smt.vmadot -- this build fell back to RVV; refusing to label it IME. See ${gate_log}" >&2
            failures=$((failures + 1))
            if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
              popd >/dev/null
              exit 1
            fi
          else
            echo "IME gate: ${model_name} vmadot=${n_vmadot} PASS -> ${gate_log}"
          fi
        fi

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

