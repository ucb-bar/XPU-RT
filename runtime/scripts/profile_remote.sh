#!/usr/bin/env bash
set -euo pipefail

# profile_remote.sh
#
# Stages generated *_benchmarks.zip bundles to a remote machine (BananaPi), unzips them,
# runs topology sweeps via run_all_topologies.sh, and copies CSV/log results back
# into gen/profile/<hw>/<target>/<model>/<basename>/...
#
# Default discovery scans:
#   gen/vmfb/**/**/**/**/*_benchmarks.zip
#
# SmolVLA ONNX Submodels (9 total):
#   action_in_projector, action_out_projector, smolvlm_expert_decode, smolvlm_expert_prefill,
#   smolvlm_text, smolvlm_vision, state_projector, time_in_projector, time_out_projector
#
# Remote defaults:
#   REMOTE=10.44.86.251 (BananaPi)
#   REMOTE_IREE_ROOT=/home/spacemit-merlin-perf
#
# Usage:
#   # Profile all ONNX submodels (default - profiles everything found)
#   ./runtime/scripts/profile_remote.sh
#
#   # Profile only specific ONNX submodels
#   SUBMODEL_FILTER="smolvlm_expert_decode smolvlm_expert_prefill" ./runtime/scripts/profile_remote.sh
#
#   # Profile specific benchmark zip files
#   ./runtime/scripts/profile_remote.sh path/to/specific_benchmarks.zip [...]
#
# Env overrides:
#   REMOTE=user@10.44.86.251
#   REMOTE_IREE_ROOT=/home/spacemit-merlin-perf
#   VMFB_ROOT=<repo>/gen/vmfb
#   PROFILE_ROOT=<repo>/gen/profile
#   SUBMODEL_FILTER="model1 model2"  Filter to specific submodels (space-separated)
#   REMOTE_TMP_BASE=/tmp
#   KEEP_REMOTE_TMP=1
#   CONTINUE_ON_ERROR=1
#   USE_STAGED_INSTALL=1      Stage local install to remote tmp and use it (default: 1)
#   LOCAL_INSTALL_DIR=...     Local install dir to stage (default: merlin/build/spacemit-merlin-perf/install)
#
# Forwarded to run_all_topologies.sh (optional):
#   DEVICE=local-task
#   INPUT_SPEC=1xi32=1
#   BENCH_REPS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE="${REMOTE:-10.44.86.251}"
REMOTE_IREE_ROOT="${REMOTE_IREE_ROOT:-/home/spacemit-merlin-perf}"
REMOTE_TMP_BASE="${REMOTE_TMP_BASE:-/tmp}"

VMFB_ROOT="${VMFB_ROOT:-${REPO_ROOT}/gen/vmfb}"
PROFILE_ROOT="${PROFILE_ROOT:-${REPO_ROOT}/gen/profile}"

# SmolVLA ONNX submodels (canonical list)
SMOLVLA_ONNX_SUBMODELS=(
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

# Optional filter: profile only specific submodels (space-separated)
SUBMODEL_FILTER="${SUBMODEL_FILTER:-}"

KEEP_REMOTE_TMP="${KEEP_REMOTE_TMP:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
USE_STAGED_INSTALL="${USE_STAGED_INSTALL:-1}"
LOCAL_INSTALL_DIR="${LOCAL_INSTALL_DIR:-${REPO_ROOT}/merlin/build/spacemit-merlin-perf/install}"

LOCAL_RUN_ALL="${REPO_ROOT}/runtime/scripts/run_all_topologies.sh"
LOCAL_RUN_PY="${REPO_ROOT}/runtime/scripts/run_vmfb_benchmarks.py"

if [[ ! -f "${LOCAL_RUN_ALL}" ]]; then
  echo "Error: missing ${LOCAL_RUN_ALL}" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_RUN_PY}" ]]; then
  echo "Error: missing ${LOCAL_RUN_PY}" >&2
  exit 1
fi

ssh_opts=(-o BatchMode=yes)

echo "========================================================================"
echo "SmolVLA ONNX Submodels - Remote Profiling on BananaPi"
echo "========================================================================"
echo "Remote: ${REMOTE}"
echo "Remote IREE root: ${REMOTE_IREE_ROOT}"
echo "VMFB root: ${VMFB_ROOT}"
echo "Profile root: ${PROFILE_ROOT}"
echo "Use staged install: ${USE_STAGED_INSTALL}"
echo "Local install dir: ${LOCAL_INSTALL_DIR}"
if [[ -n "${SUBMODEL_FILTER}" ]]; then
  echo "Submodel filter: ${SUBMODEL_FILTER}"
else
  echo "Submodel filter: ALL (${#SMOLVLA_ONNX_SUBMODELS[@]} models)"
fi
echo "========================================================================"

bench_tool_remote=""
remote_ld_library_path=""

remote_tmp="$(
  ssh "${ssh_opts[@]}" "${REMOTE}" "mktemp -d '${REMOTE_TMP_BASE%/}/freshsched_profile.XXXXXX'"
)"
if [[ -z "${remote_tmp}" ]]; then
  echo "Error: failed to create remote temp dir" >&2
  exit 1
fi
echo "Remote tmp: ${remote_tmp}"

cleanup_remote() {
  if [[ "${KEEP_REMOTE_TMP}" == "1" ]]; then
    echo "Keeping remote tmp (KEEP_REMOTE_TMP=1): ${remote_tmp}"
    return
  fi
  ssh "${ssh_opts[@]}" "${REMOTE}" "rm -rf '${remote_tmp}'" || true
}
trap cleanup_remote EXIT

echo "Uploading helper scripts..."
scp -q "${LOCAL_RUN_ALL}" "${LOCAL_RUN_PY}" "${REMOTE}:${remote_tmp}/"
ssh "${ssh_opts[@]}" "${REMOTE}" "chmod +x '${remote_tmp}/run_all_topologies.sh'"

if [[ "${USE_STAGED_INSTALL}" == "1" ]]; then
  if [[ ! -d "${LOCAL_INSTALL_DIR}" ]]; then
    echo "Error: USE_STAGED_INSTALL=1 but LOCAL_INSTALL_DIR is missing: ${LOCAL_INSTALL_DIR}" >&2
    exit 1
  fi
  echo "Staging local install to remote..."
  remote_install="${remote_tmp}/iree_install"
  ssh "${ssh_opts[@]}" "${REMOTE}" "mkdir -p '${remote_install}'"
  # Stream a compressed tarball to avoid many small file transfers.
  tar -C "${LOCAL_INSTALL_DIR}" -czf - . | ssh "${ssh_opts[@]}" "${REMOTE}" "tar -C '${remote_install}' -xzf -"

  # Prefer the staged install's benchmark tool.
  if ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${remote_install}/tools/iree-benchmark-module'"; then
    bench_tool_remote="${remote_install}/tools/iree-benchmark-module"
  elif ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${remote_install}/bin/iree-benchmark-module'"; then
    bench_tool_remote="${remote_install}/bin/iree-benchmark-module"
  else
    echo "Warning: staged install missing iree-benchmark-module under ${remote_install}/{tools,bin}; falling back to REMOTE_IREE_ROOT" >&2
  fi

  # If staged install has shared libs, add them to LD_LIBRARY_PATH for the remote run.
  # (Safe even if the binary is static.)
  if ssh "${ssh_opts[@]}" "${REMOTE}" "test -d '${remote_install}/lib'"; then
    remote_ld_library_path="${remote_install}/lib"
  fi
  if ssh "${ssh_opts[@]}" "${REMOTE}" "test -d '${remote_install}/lib64'"; then
    remote_ld_library_path="${remote_install}/lib64${remote_ld_library_path:+:${remote_ld_library_path}}"
  fi
fi

if [[ -z "${bench_tool_remote}" ]]; then
  if ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${REMOTE_IREE_ROOT}/tools/iree-benchmark-module'"; then
    bench_tool_remote="${REMOTE_IREE_ROOT}/tools/iree-benchmark-module"
  elif ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${REMOTE_IREE_ROOT}/bin/iree-benchmark-module'"; then
    bench_tool_remote="${REMOTE_IREE_ROOT}/bin/iree-benchmark-module"
  else
    echo "Error: could not find iree-benchmark-module under staged install or ${REMOTE_IREE_ROOT}/{tools,bin}" >&2
    exit 1
  fi
fi
echo "Remote bench tool: ${bench_tool_remote}"
if [[ -n "${remote_ld_library_path}" ]]; then
  echo "Remote LD_LIBRARY_PATH prefix: ${remote_ld_library_path}"
fi

# Build filter map if SUBMODEL_FILTER is set
declare -A submodel_allow=()
if [[ -n "${SUBMODEL_FILTER}" ]]; then
  # shellcheck disable=SC2206
  _filter_list=(${SUBMODEL_FILTER})
  for sm in "${_filter_list[@]}"; do
    submodel_allow["${sm}"]=1
  done
  echo "Filtering to ${#submodel_allow[@]} submodels: ${SUBMODEL_FILTER}"
fi

zips=()
if [[ $# -gt 0 ]]; then
  # Explicit paths provided
  zips=("$@")
  echo "Using ${#zips[@]} explicitly provided benchmark zip(s)"
else
  # Auto-discovery mode
  if [[ ! -d "${VMFB_ROOT}" ]]; then
    echo "Error: VMFB_ROOT does not exist: ${VMFB_ROOT}" >&2
    exit 1
  fi

  # Find all *_benchmarks.zip files
  all_zips=()
  while IFS= read -r -d '' z; do
    all_zips+=("$z")
  done < <(find "${VMFB_ROOT}" -type f -name '*_benchmarks.zip' -print0 | sort -z)

  echo "Found ${#all_zips[@]} benchmark zip file(s) in ${VMFB_ROOT}"

  # Apply submodel filter if set
  if [[ ${#submodel_allow[@]} -gt 0 ]]; then
    for z in "${all_zips[@]}"; do
      # Extract model name from path: <VMFB_ROOT>/<model>/...
      rel="${z#${VMFB_ROOT%/}/}"
      model="${rel%%/*}"
      if [[ -n "${submodel_allow[$model]:-}" ]]; then
        zips+=("$z")
      fi
    done
    echo "After filtering: ${#zips[@]} benchmark zip(s) match submodel filter"
  else
    # No filter - use all
    zips=("${all_zips[@]}")
  fi
fi

if [[ ${#zips[@]} -eq 0 ]]; then
  echo "Error: no *_benchmarks.zip found (VMFB_ROOT=${VMFB_ROOT})" >&2
  if [[ -n "${SUBMODEL_FILTER}" ]]; then
    echo "       (with SUBMODEL_FILTER=${SUBMODEL_FILTER})" >&2
  fi
  exit 1
fi

echo ""
echo "Will profile ${#zips[@]} benchmark bundle(s):"
for z in "${zips[@]}"; do
  rel="${z#${VMFB_ROOT%/}/}"
  echo "  - ${rel}"
done
echo ""

failures=0
success=0
skipped=0

for zip_path in "${zips[@]}"; do
  zip_abs="$(cd "$(dirname "${zip_path}")" && pwd)/$(basename "${zip_path}")"
  if [[ ! -f "${zip_abs}" ]]; then
    echo "Missing zip: ${zip_abs}" >&2
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && continue || exit 1
  fi

  # Expect layout: <VMFB_ROOT>/<model>/<target>/<hw>/<basename>/<zip>
  rel="${zip_abs#${VMFB_ROOT%/}/}"
  IFS='/' read -r model target hw basename zipfile <<<"${rel}"
  if [[ -z "${model}" || -z "${target}" || -z "${hw}" || -z "${basename}" ]]; then
    echo "Warning: zip is not under expected VMFB_ROOT layout, skipping: ${zip_abs}" >&2
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && continue || exit 1
  fi

  input_tag="${model}_${target}_${hw}_${basename}"
  remote_in="${remote_tmp}/in/${input_tag}"
  remote_out="${remote_tmp}/out/${input_tag}"
  local_out="${PROFILE_ROOT}/${hw}/${target}/${model}/${basename}"

  echo "================================================================================"
  echo "ZIP        : ${zip_abs}"
  echo "Model      : ${model}"
  echo "Target/HW  : ${target} / ${hw}"
  echo "Basename   : ${basename}"
  echo "Remote in  : ${remote_in}"
  echo "Remote out : ${remote_out}"
  echo "Local out  : ${local_out}"
  echo "================================================================================"

  mkdir -p "${local_out}"

  # Stage zip to remote and unzip.
  ssh "${ssh_opts[@]}" "${REMOTE}" "mkdir -p '${remote_in}' '${remote_out}'"
  scp -q "${zip_abs}" "${REMOTE}:${remote_tmp}/bundle.zip"
  ssh "${ssh_opts[@]}" "${REMOTE}" "cd '${remote_in}' && unzip -o -q '${remote_tmp}/bundle.zip'"

  # Run topology sweep on remote.
  # run_all_topologies.sh wants: <base_out_dir> <input_dir1> [input_dir2...]
  set +e
  ssh "${ssh_opts[@]}" "${REMOTE}" \
    "cd '${remote_tmp}' && \
     ${remote_ld_library_path:+LD_LIBRARY_PATH='${remote_ld_library_path}':\"\$LD_LIBRARY_PATH\"} \
     PY_SCRIPT='${remote_tmp}/run_vmfb_benchmarks.py' \
     BENCH_TOOL='${bench_tool_remote}' \
     DEVICE='${DEVICE:-local-task}' \
     INPUT_SPEC='${INPUT_SPEC:-1xi32=1}' \
     BENCH_REPS='${BENCH_REPS:-10}' \
     bash '${remote_tmp}/run_all_topologies.sh' '${remote_out}' '${remote_in}'"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "❌ FAILED profiling (rc=${rc}): ${model}" >&2
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  else
    echo "✅ SUCCESS: ${model}"
    success=$((success + 1))
  fi

  # Copy results back.
  # remote_out contains: <base_out_dir>/<INPUT_NAME>/topo_.../results.csv,...
  # INPUT_NAME is basename(remote_in) which is input_tag.
  scp -q -r "${REMOTE}:${remote_out}/${input_tag}/" "${local_out}/"
  echo "Results copied to: ${local_out}"
done

echo ""
echo "========================================================================"
echo "Profiling Complete"
echo "========================================================================"
echo "Total models processed: $((success + failures))"
echo "✅ Success: ${success}"
echo "❌ Failed: ${failures}"
echo "Profile results: ${PROFILE_ROOT}"
echo "========================================================================"

if [[ "${failures}" -ne 0 ]]; then
  echo "Done with failures: ${failures}" >&2
  exit 2
fi

echo "All profiles completed successfully!"

