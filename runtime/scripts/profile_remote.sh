#!/usr/bin/env bash
set -euo pipefail

# profile_remote.sh
#
# Stages generated *_benchmarks.zip bundles to a remote machine, unzips them,
# runs topology sweeps via run_all_topologies.sh, and copies CSV/log results back
# into gen/profile/<hw>/<target>/<model>/<basename>/...
#
# Default discovery scans:
#   gen/vmfb/**/**/**/**/*_benchmarks.zip
#
# Remote defaults:
#   REMOTE=10.44.86.251
#   REMOTE_IREE_ROOT=/home/spacemit-merlin-perf
#
# Usage:
#   ./runtime/scripts/profile_remote.sh
#   ./runtime/scripts/profile_remote.sh path/to/specific_benchmarks.zip [...]
#
# Env overrides:
#   REMOTE=user@10.44.86.251
#   REMOTE_IREE_ROOT=/home/spacemit-merlin-perf
#   VMFB_ROOT=<repo>/gen/vmfb
#   PROFILE_ROOT=<repo>/gen/profile
#   REMOTE_TMP_BASE=/tmp
#   KEEP_REMOTE_TMP=1
#   CONTINUE_ON_ERROR=1
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

KEEP_REMOTE_TMP="${KEEP_REMOTE_TMP:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

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

echo "Remote: ${REMOTE}"
echo "Remote IREE root: ${REMOTE_IREE_ROOT}"
echo "VMFB root: ${VMFB_ROOT}"
echo "Profile root: ${PROFILE_ROOT}"

bench_tool_remote=""
if ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${REMOTE_IREE_ROOT}/tools/iree-benchmark-module'"; then
  bench_tool_remote="${REMOTE_IREE_ROOT}/tools/iree-benchmark-module"
elif ssh "${ssh_opts[@]}" "${REMOTE}" "test -x '${REMOTE_IREE_ROOT}/bin/iree-benchmark-module'"; then
  bench_tool_remote="${REMOTE_IREE_ROOT}/bin/iree-benchmark-module"
else
  echo "Error: could not find iree-benchmark-module under ${REMOTE_IREE_ROOT}/{tools,bin}" >&2
  exit 1
fi
echo "Remote bench tool: ${bench_tool_remote}"

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

zips=()
if [[ $# -gt 0 ]]; then
  zips=("$@")
else
  if [[ ! -d "${VMFB_ROOT}" ]]; then
    echo "Error: VMFB_ROOT does not exist: ${VMFB_ROOT}" >&2
    exit 1
  fi
  while IFS= read -r -d '' z; do
    zips+=("$z")
  done < <(find "${VMFB_ROOT}" -type f -name '*_benchmarks.zip' -print0 | sort -z)
fi

if [[ ${#zips[@]} -eq 0 ]]; then
  echo "Error: no *_benchmarks.zip found (VMFB_ROOT=${VMFB_ROOT})" >&2
  exit 1
fi

failures=0

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
     PY_SCRIPT='${remote_tmp}/run_vmfb_benchmarks.py' \
     BENCH_TOOL='${bench_tool_remote}' \
     DEVICE='${DEVICE:-local-task}' \
     INPUT_SPEC='${INPUT_SPEC:-1xi32=1}' \
     BENCH_REPS='${BENCH_REPS:-10}' \
     bash '${remote_tmp}/run_all_topologies.sh' '${remote_out}' '${remote_in}'"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED profiling (rc=${rc}): ${zip_abs}" >&2
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  fi

  # Copy results back.
  # remote_out contains: <base_out_dir>/<INPUT_NAME>/topo_.../results.csv,...
  # INPUT_NAME is basename(remote_in) which is input_tag.
  scp -q -r "${REMOTE}:${remote_out}/${input_tag}/" "${local_out}/"
done

if [[ "${failures}" -ne 0 ]]; then
  echo "Done with failures: ${failures}" >&2
  exit 2
fi

echo "Done."

