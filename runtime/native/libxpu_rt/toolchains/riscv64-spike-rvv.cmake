# CMake toolchain for cross-compiling libxpu_rt + XNNPACK to riscv64
# with RVV 1.0 (VLEN=128), targeting Chipyard's Spike via HTIF.
#
# Mirrors the toolchain Merlin uses to run IREE on Chipyard
# (`/scratch2/agustin/merlin/build_tools/firesim/riscv_firesim.toolchain.cmake`):
#
#   - Clang 18 from Merlin's IREE bundle handles compilation. Clang 18
#     ships correct RVV v1.0 intrinsic spellings the UCB-BAR XNNPACK
#     fork uses (multi-vector tuple types, segmented ld/st,
#     __RISCV_FRM_*, vsext_vf2). Chipyard's bundled GCC 13.2 has
#     intrinsic drift on ~250 microkernel files; GCC 14+ would also
#     work but Chipyard does not ship it.
#
#   - Chipyard's riscv64-unknown-elf-gcc 13.2 is the linker driver,
#     invoked through htif_nano.specs (Chipyard's newlib HTIF specs)
#     so printf / exit / _write resolve to HTIF tohost/fromhost
#     syscalls. ELF runs directly on Spike — no pk, no Linux, no
#     futex(2). Merlin's htif.ld linker script provides the memory
#     map + TLS .tdata / .tbss layout + HTIF section.
#
#   - Chipyard's riscv64-unknown-elf sysroot supplies newlib + libstdc++
#     headers. The compat/picolibc-freestanding directory injects
#     <pthread.h>, <time.h>, and a std:: math compat <cmath> for the
#     things XNNPACK references that the sysroot's libstdc++ doesn't
#     pull into namespace std::.
#
# Usage (env vars provide override hooks):
#   CHIPYARD_ROOT=/scratch2/agustin/chipyard \
#   MERLIN_IREE_TOOLCHAIN_ROOT=/scratch2/agustin/merlin/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV \
#   cmake -B build/riscv-spike -S runtime/native/libxpu_rt \
#       -DCMAKE_TOOLCHAIN_FILE=runtime/native/libxpu_rt/toolchains/riscv64-spike-rvv.cmake \
#       -DCG_RT_PLATFORM=bare \
#       -DCG_RT_WITH_CUDA=OFF \
#       -DXPURT_WITH_XNNPACK=ON
#   cmake --build build/riscv-spike --parallel
#   spike --isa=rv64gcv build/riscv-spike/test_xnnpack_bridge_riscv.elf

# --- Toolchain roots ----------------------------------------------------
set(CHIPYARD_ROOT "$ENV{CHIPYARD_ROOT}"
    CACHE PATH "Chipyard root (contains .conda-env/riscv-tools)")
if(NOT CHIPYARD_ROOT OR NOT EXISTS "${CHIPYARD_ROOT}/.conda-env/riscv-tools")
    set(CHIPYARD_ROOT "/scratch2/agustin/chipyard"
        CACHE PATH "Chipyard root" FORCE)
endif()

set(MERLIN_IREE_TOOLCHAIN_ROOT "$ENV{MERLIN_IREE_TOOLCHAIN_ROOT}"
    CACHE PATH "Merlin IREE clang toolchain root")
if(NOT MERLIN_IREE_TOOLCHAIN_ROOT OR NOT EXISTS "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/clang")
    set(MERLIN_IREE_TOOLCHAIN_ROOT
        "/scratch2/agustin/merlin/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV"
        CACHE PATH "Merlin IREE clang root" FORCE)
endif()

set(RISCV_NEWLIB_SYSROOT "${CHIPYARD_ROOT}/.conda-env/riscv-tools/riscv64-unknown-elf")
set(RISCV_GCC_BIN        "${CHIPYARD_ROOT}/.conda-env/riscv-tools/bin")

if(NOT EXISTS "${RISCV_NEWLIB_SYSROOT}/lib/htif_nano.specs")
    message(FATAL_ERROR
        "Expected Chipyard newlib htif_nano.specs at "
        "${RISCV_NEWLIB_SYSROOT}/lib/htif_nano.specs — toolchain incomplete.")
endif()

# Merlin's HTIF linker script (richer than the existing tests/bare/link.ld;
# handles TLS .tdata / .tbss and reserves 1 GiB heap for the embedded
# weight blob).
set(_merlin_firesim_dir "/scratch2/agustin/merlin/build_tools/firesim")
set(XPURT_RISCV_HTIF_LINKER_SCRIPT "${_merlin_firesim_dir}/htif.ld"
    CACHE FILEPATH "HTIF linker script (defaults to merlin's)")

# --- CMake compiler / linker assignments --------------------------------
set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/clang"   CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/clang++" CACHE FILEPATH "")
set(CMAKE_ASM_COMPILER "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/clang"   CACHE FILEPATH "")
set(CMAKE_AR           "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/llvm-ar"     CACHE FILEPATH "")
set(CMAKE_RANLIB       "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/llvm-ranlib" CACHE FILEPATH "")
set(CMAKE_STRIP        "${MERLIN_IREE_TOOLCHAIN_ROOT}/bin/llvm-strip"  CACHE FILEPATH "")

# GCC is the linker driver — needed so htif_nano.specs (which references
# nano.specs + nosys.specs) resolves the standard library cascade.
set(CMAKE_LINKER "${RISCV_GCC_BIN}/riscv64-unknown-elf-gcc" CACHE FILEPATH "")

set(CMAKE_SYSROOT "${RISCV_NEWLIB_SYSROOT}")

# --- C++ header path discovery ------------------------------------------
file(GLOB _cxx_dirs "${RISCV_NEWLIB_SYSROOT}/include/c++/*")
if(_cxx_dirs)
    list(GET _cxx_dirs 0 _cxx_dir)
endif()
if(NOT _cxx_dir)
    message(FATAL_ERROR "Could not find C++ headers under ${RISCV_NEWLIB_SYSROOT}/include/c++/")
endif()

# --- Compile flags ------------------------------------------------------
set(_arch_flags "-march=rv64gcv -mabi=lp64d -mcmodel=medany")

# Clang compile flags (path to C++ headers explicit because clang doesn't
# default-walk gcc's directory layout; -isystem so the
# picolibc-freestanding compat dir wins over the sysroot for the
# headers we override).
set(_clang_common_flags
    "--target=riscv64-unknown-elf \
--sysroot=${RISCV_NEWLIB_SYSROOT} \
-I${CMAKE_CURRENT_LIST_DIR}/../src/compat/picolibc-freestanding \
-I${_cxx_dir} \
-I${_cxx_dir}/riscv64-unknown-elf \
-I${RISCV_NEWLIB_SYSROOT}/include \
${_arch_flags} \
-D_POSIX_THREADS \
-fno-pic -fno-plt -fno-common \
-ftls-model=local-exec \
-ffunction-sections -fdata-sections")

set(CMAKE_C_FLAGS_INIT   "${_clang_common_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_clang_common_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_clang_common_flags}")

# --- GCC link rule (overrides CMake's default exe link recipe) ---------
# CMake's default would call clang to link; we need gcc + htif_nano.specs
# so newlib's HTIF stubs come in automatically. -Wl,--gc-sections drops
# the bulk of unreached XNNPACK code.
set(_gcc_link_flags
    "${_arch_flags} \
-static \
-specs=htif_nano.specs \
-T${XPURT_RISCV_HTIF_LINKER_SCRIPT} \
-Wl,--gc-sections")

set(CMAKE_EXE_LINKER_FLAGS_INIT "")  # cleared; rule below appends instead

set(CMAKE_C_LINK_EXECUTABLE
    "<CMAKE_LINKER> <OBJECTS> <LINK_LIBRARIES> ${_gcc_link_flags} -o <TARGET>"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_LINKER> <OBJECTS> <LINK_LIBRARIES> ${_gcc_link_flags} -o <TARGET>"
    CACHE STRING "" FORCE)

# --- CMake try-compile / find_* behaviour ------------------------------
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER_WORKS   TRY_COMPILE)
set(CMAKE_CXX_COMPILER_WORKS TRY_COMPILE)
set(CMAKE_FIND_ROOT_PATH "${RISCV_NEWLIB_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# Bare-metal newlib has no libdl; force CMAKE_DL_LIBS empty so any
# unconditional ${CMAKE_DL_LIBS} reference doesn't cascade into -ldl.
set(CMAKE_DL_LIBS "" CACHE STRING "" FORCE)

# Surface flag for downstream CMake.
set(XPURT_RISCV_SPIKE TRUE CACHE BOOL
    "Cross-compiling libxpu_rt for riscv64+RVV (clang 18 → Chipyard Spike HTIF)"
    FORCE)
