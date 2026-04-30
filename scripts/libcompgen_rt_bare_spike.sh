#!/usr/bin/env bash
#
# Cross-compile libcompgen_rt for bare-metal RISC-V and run the smoke
# test on Spike. Uses the Zephyr SDK + Spike bundled with Dima's
# chipyard tree by default; override via $ZEPHYR_SDK / $SPIKE env
# variables.
#
# Exits 0 on smoke test pass, non-zero on any toolchain / build /
# Spike failure. Designed to be called from CI and from the pytest
# wrapper in tests/runtime/test_libcompgen_rt_bare.py.

set -euo pipefail

COMPGEN_ROOT="${COMPGEN_ROOT:-/scratch2/agustin/CompGen}"
LIB_ROOT="$COMPGEN_ROOT/runtime/native/libcompgen_rt"
BUILD_DIR="${BUILD_DIR:-$LIB_ROOT/build-riscv}"

DIMA_TREE="${DIMA_TREE:-/scratch2/dima/testing/zephyr-chipyard-sw-torch-dryrun-2}"
ZEPHYR_SDK_DEFAULT="$DIMA_TREE/tools-manual/zephyr-sdk-1.0.0-beta1"
SPIKE_DEFAULT="$DIMA_TREE/tools/miniforge3/envs/zephyr/bin/spike"

ZEPHYR_SDK="${ZEPHYR_SDK:-$ZEPHYR_SDK_DEFAULT}"
SPIKE="${SPIKE:-$SPIKE_DEFAULT}"

if [[ ! -x "$ZEPHYR_SDK/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc" ]]; then
    echo "error: riscv64-zephyr-elf-gcc not found under $ZEPHYR_SDK" >&2
    exit 2
fi
if [[ ! -x "$SPIKE" ]]; then
    echo "error: spike not found at $SPIKE" >&2
    exit 2
fi

echo "==> configuring ($BUILD_DIR)"
ZEPHYR_SDK="$ZEPHYR_SDK" /usr/bin/cmake \
    -B "$BUILD_DIR" -S "$LIB_ROOT" \
    -DCMAKE_TOOLCHAIN_FILE="$LIB_ROOT/toolchains/riscv64-zephyr-elf.cmake" \
    -DCG_RT_PLATFORM=bare \
    -DCG_RT_WITH_CUDA=OFF

echo "==> building"
/usr/bin/cmake --build "$BUILD_DIR"

ELF="$BUILD_DIR/smoke_bare.elf"
if [[ ! -f "$ELF" ]]; then
    echo "error: smoke_bare.elf not produced" >&2
    exit 3
fi

echo "==> running on spike"
"$SPIKE" --isa=rv64gc "$ELF"
status=$?
echo "==> spike exit=$status"
exit $status
