# CMake toolchain file for cross-compiling libcompgen_rt to RISC-V
# bare-metal using the Zephyr SDK's riscv64-zephyr-elf GCC.
#
# Usage:
#   /usr/bin/cmake -B build-riscv -S runtime/native/libcompgen_rt \
#       -DCMAKE_TOOLCHAIN_FILE=toolchains/riscv64-zephyr-elf.cmake \
#       -DCG_RT_PLATFORM=bare \
#       -DCG_RT_WITH_CUDA=OFF
#   /usr/bin/cmake --build build-riscv
#
# The compiler path is picked up from the ``ZEPHYR_SDK`` environment
# variable with a sensible default on this host.

set(ZEPHYR_SDK "$ENV{ZEPHYR_SDK}"
    CACHE PATH "Zephyr SDK root containing gnu/riscv64-zephyr-elf/bin")
if(NOT ZEPHYR_SDK OR NOT EXISTS "${ZEPHYR_SDK}")
    set(ZEPHYR_SDK "/scratch2/dima/testing/zephyr-chipyard-sw-torch-dryrun-2/tools-manual/zephyr-sdk-1.0.0-beta1"
        CACHE PATH "Zephyr SDK root" FORCE)
endif()

set(_tc_bin "${ZEPHYR_SDK}/gnu/riscv64-zephyr-elf/bin")

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   "${_tc_bin}/riscv64-zephyr-elf-gcc"   CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${_tc_bin}/riscv64-zephyr-elf-g++"   CACHE FILEPATH "")
set(CMAKE_AR           "${_tc_bin}/riscv64-zephyr-elf-ar"    CACHE FILEPATH "")
set(CMAKE_RANLIB       "${_tc_bin}/riscv64-zephyr-elf-ranlib" CACHE FILEPATH "")

# Freestanding: no OS, no libc assumptions.  We still link libgcc for
# __atomic_* helpers used by stdatomic.h on RV64 when the hart lacks
# the ``a`` extension (Spike has it, so generally unused).
set(CMAKE_C_FLAGS_INIT
    "-ffreestanding -fno-builtin -mabi=lp64d -march=rv64imafdc_zicsr_zifencei -mcmodel=medany")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostartfiles")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Skip compiler checks — freestanding link tests fail without startup
# files, which is expected.
set(CMAKE_C_COMPILER_WORKS   TRY_COMPILE)
set(CMAKE_CXX_COMPILER_WORKS TRY_COMPILE)
