#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

XPURT_LIB="${XPURT_LIB:-}"
BUILD_TYPE="Release"
TARGET="host"  # host | spacemit

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --xpurt-lib FILE    Path to either:
                        - libxpurt_iree_plugin_standalone.a (preferred), or
                        - libxpurt_iree_plugin.a (plugin-only; script will try to find/build standalone)
                      Defaults to \$XPURT_LIB if set.
  --target NAME       Build target: host | spacemit (default: host).
  --build-type TYPE   CMake build type (default: Release).
  -h, --help          Show this help.

Examples:
  # Host build:
  XPURT_LIB=/path/to/host/libxpurt_iree_plugin_standalone.a $(basename "$0")

  # SpacemiT X60 build (uses Merlin-installed toolchain if RISCV_TOOLCHAIN_ROOT unset):
  $(basename "$0") --target spacemit \\
    --xpurt-lib /path/to/spacemit-merlin-perf/.../libxpurt_iree_plugin.a
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --xpurt-lib)
      XPURT_LIB="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${XPURT_LIB}" ]]; then
  echo "Error: xpu-rt library path not set. Use --xpurt-lib or set XPURT_LIB." >&2
  exit 1
fi

# Use target-specific build dirs to avoid host/spacemit compiler cache conflicts
BUILD_DIR="${SCRIPT_DIR}/build-${TARGET}"

echo "Building xpu-rt runtime:"
echo "  TARGET      = ${TARGET}"
echo "  XPURT_LIB   = ${XPURT_LIB}"
echo "  BUILD_TYPE  = ${BUILD_TYPE}"
echo "  BUILD_DIR   = ${BUILD_DIR}"

XPURT_LIB_ABS="$(cd "$(dirname "${XPURT_LIB}")" && pwd)/$(basename "${XPURT_LIB}")"

# If the user passed the plugin-only archive inside a Merlin build tree, try to
# use the combined standalone archive (plugin + IREE runtime objects) instead.
if [[ "$(basename "${XPURT_LIB_ABS}")" == "libxpurt_iree_plugin.a" ]]; then
  # From .../build-name/runtime/plugins/xpu-rt/libxpurt_iree_plugin.a -> .../build-name
  BUILD_DIR_MERLIN="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "${XPURT_LIB_ABS}")")")")" && pwd)"
  if [[ -f "${BUILD_DIR_MERLIN}/CMakeCache.txt" ]]; then
    STANDALONE_LIB="${BUILD_DIR_MERLIN}/runtime/src/iree/runtime/libxpurt_iree_plugin_standalone.a"
    if [[ ! -f "${STANDALONE_LIB}" ]]; then
      echo "  Standalone archive not found; building it in Merlin tree: ${BUILD_DIR_MERLIN}"
      cmake --build "${BUILD_DIR_MERLIN}" --target xpurt_iree_plugin_standalone
    fi
    if [[ -f "${STANDALONE_LIB}" ]]; then
      echo "  Using standalone archive: ${STANDALONE_LIB}"
      XPURT_LIB="${STANDALONE_LIB}"
      XPURT_LIB_ABS="${STANDALONE_LIB}"
    else
      echo "Warning: standalone archive still not found at ${STANDALONE_LIB}" >&2
      echo "         Out-of-tree link may fail unless you pass the standalone archive explicitly." >&2
    fi
  fi
fi

# Auto-detect FlatCC parsing/verifier archive from the Merlin build tree when possible.
# The missing flatcc_verify_* symbols come from libflatcc_parsing.a.
XPURT_FLATCC_PARSING_LIB="${XPURT_FLATCC_PARSING_LIB:-}"
if [[ -z "${XPURT_FLATCC_PARSING_LIB}" ]]; then
  # If the xpurt lib is inside a Merlin build tree, use that as a base.
  BUILD_DIR_MERLIN=""
  if [[ "$(basename "${XPURT_LIB_ABS}")" == "libxpurt_iree_plugin_standalone.a" ]]; then
    # From .../build-name/runtime/src/iree/runtime/libxpurt_iree_plugin_standalone.a -> .../build-name
    BUILD_DIR_MERLIN="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "${XPURT_LIB_ABS}")")")")")" && pwd)"
  elif [[ "$(basename "${XPURT_LIB_ABS}")" == "libxpurt_iree_plugin.a" ]]; then
    # From .../build-name/runtime/plugins/xpu-rt/libxpurt_iree_plugin.a -> .../build-name
    BUILD_DIR_MERLIN="$(cd "$(dirname "$(dirname "$(dirname "$(dirname "${XPURT_LIB_ABS}")")")")" && pwd)"
  fi

  if [[ -n "${BUILD_DIR_MERLIN}" && -d "${BUILD_DIR_MERLIN}/build_tools/third_party/flatcc" ]]; then
    CAND="${BUILD_DIR_MERLIN}/build_tools/third_party/flatcc/libflatcc_parsing.a"
    if [[ -f "${CAND}" ]]; then
      XPURT_FLATCC_PARSING_LIB="${CAND}"
      echo "  Detected FlatCC parsing/verifier: ${XPURT_FLATCC_PARSING_LIB}"
    fi
  fi
fi

# Fallback: build in runtime dir (needs IREE libs for linking; often fails for
# spacemit/host unless you have a full IREE install).
CMAKE_EXTRA_ARGS=()
if [[ "${TARGET}" == "spacemit" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  MERLIN_ROOT="${PROJECT_ROOT}/merlin"
  IREE_SRC="${MERLIN_ROOT}/third_party/iree_bar"
  TOOLCHAIN_FILE="${IREE_SRC}/build_tools/cmake/riscv.toolchain.cmake"

  if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "Error: RISC-V toolchain file not found at ${TOOLCHAIN_FILE}" >&2
    exit 1
  fi

  RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT:-}"
  if [[ -z "${RISCV_TOOLCHAIN_ROOT}" ]]; then
    DEFAULT_TC="${MERLIN_ROOT}/build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
    if [[ -d "${DEFAULT_TC}" ]]; then
      RISCV_TOOLCHAIN_ROOT="${DEFAULT_TC}"
      echo "  (using Merlin-installed toolchain: ${RISCV_TOOLCHAIN_ROOT})"
    else
      echo "Error: RISCV_TOOLCHAIN_ROOT not set and Merlin toolchain not found at ${DEFAULT_TC}" >&2
      exit 1
    fi
  fi

  CMAKE_EXTRA_ARGS+=(
    "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
    "-DRISCV_CPU=linux-riscv_64"
    "-DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}"
    "-DCMAKE_C_FLAGS=-march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d"
  )
fi

cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DXPURT_STANDALONE_LIB_PATH="${XPURT_LIB}" \
  ${XPURT_FLATCC_PARSING_LIB:+-DXPURT_FLATCC_PARSING_LIB_PATH=${XPURT_FLATCC_PARSING_LIB}} \
  "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "${BUILD_DIR}" --target json_dispatch_runner

echo "Done. Binary is in: ${BUILD_DIR}"

