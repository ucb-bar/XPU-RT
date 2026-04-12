#!/usr/bin/env bash
# profile_smolvla_onnx.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All 9 ONNX submodels
ALL_ONNX_MODELS=(
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

# Expert models only
EXPERT_MODELS=(
  smolvlm_expert_decode
  smolvlm_expert_prefill
)

# Parse arguments
if [[ $# -eq 0 ]]; then
  # No args: profile all ONNX models
  SUBMODEL_FILTER=""
  echo "Profiling ALL 9 ONNX submodels"
elif [[ "$1" == "experts" ]]; then
  # "experts" arg: profile only the two expert models
  SUBMODEL_FILTER="${EXPERT_MODELS[*]}"
  echo "Profiling EXPERT models only: ${SUBMODEL_FILTER}"
else
  # Explicit model names
  SUBMODEL_FILTER="$*"
  echo "Profiling specified models: ${SUBMODEL_FILTER}"
fi

# Set defaults
export REMOTE="${REMOTE:-10.44.86.251}"
export DEVICE="${DEVICE:-local-task}"
export BENCH_REPS="${BENCH_REPS:-10}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Run profile_remote.sh with filter
if [[ -n "${SUBMODEL_FILTER}" ]]; then
  export SUBMODEL_FILTER
fi

echo "========================================================================"
echo "SmolVLA ONNX Profiling on BananaPi"
echo "========================================================================"
echo "Remote: ${REMOTE}"
echo "Device: ${DEVICE}"
echo "Benchmark reps: ${BENCH_REPS}"
echo "Continue on error: ${CONTINUE_ON_ERROR}"
if [[ -n "${SUBMODEL_FILTER}" ]]; then
  echo "Models: ${SUBMODEL_FILTER}"
else
  echo "Models: ALL (9 total)"
fi
echo "========================================================================"
echo ""

# Execute
"${SCRIPT_DIR}/profile_remote.sh"

exit_code=$?

if [[ ${exit_code} -eq 0 ]]; then
  echo ""
  echo "✅ All profiling completed successfully!"
  echo "Results are in: gen/profile/"
else
  echo ""
  echo "⚠️  Profiling completed with errors (exit code: ${exit_code})"
  echo "Check logs for details"
fi

exit ${exit_code}
