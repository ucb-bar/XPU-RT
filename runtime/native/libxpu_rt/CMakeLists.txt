cmake_minimum_required(VERSION 3.16)
project(xpu_rt C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# libxpu_rt — the native HAL
# ---------------------------------------------------------------------------

# --- Platform backend selection ------------------------------------------
# ``posix``: pthread mutex/condvar + clock_gettime. The only backend
#   that supports the cpu_task driver (which spawns threads).
# ``bare``:  no OS, no threading. cpu_task is omitted; cpu_sync still
#   works because the platform shim makes its mutex a no-op.
set(CG_RT_PLATFORM "posix" CACHE STRING "Platform backend: posix or bare")
set_property(CACHE CG_RT_PLATFORM PROPERTY STRINGS posix bare)

set(CG_RT_SOURCES
    src/core/instance.c
    src/core/buffer.c
    src/core/semaphore.c
    src/core/event_tensor.c
    src/core/executable.c
    src/core/command_buffer.c
    src/core/cpu_executor.c
    src/core/queue.c
    src/drivers/cpu_sync/cpu_sync.c
    # Phase G — backend scaffolds. Vtables compile unconditionally
    # and return CG_RT_ERR_UNSUPPORTED until CG_RT_WITH_<NAME> is on.
    src/drivers/hip/hip_driver.c
    src/drivers/vulkan/vulkan_driver.c
    src/drivers/firesim/firesim_driver.c
    # XNNPACK bridge — always compiled. Body is stubbed (returns
    # XPU_RT_XNN_ENOTSUP from every entry) until XPURT_WITH_XNNPACK
    # is ON. See the option block below.
    src/drivers/xnnpack/xnnpack_bridge.c
)

if(CG_RT_PLATFORM STREQUAL "posix")
    list(APPEND CG_RT_SOURCES
        src/platform/posix/posix.c
        src/drivers/cpu_task/cpu_task.c
    )
    set(CG_RT_PLATFORM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/posix)
elseif(CG_RT_PLATFORM STREQUAL "bare")
    list(APPEND CG_RT_SOURCES src/platform/bare/bare.c)
    set(CG_RT_PLATFORM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/bare)
    add_compile_definitions(CG_RT_PLATFORM_BARE)
else()
    message(FATAL_ERROR "Unknown CG_RT_PLATFORM=${CG_RT_PLATFORM}")
endif()
message(STATUS "xpu_rt: platform backend = ${CG_RT_PLATFORM}")

# --- Optional CUDA driver -------------------------------------------------
# Enabled automatically when the CUDA toolkit is findable. Users can
# disable explicitly with ``-DCG_RT_WITH_CUDA=OFF``.
option(CG_RT_WITH_CUDA "Build the CUDA driver (requires CUDA toolkit + NVRTC)" ON)

set(CG_RT_HAVE_CUDA FALSE)
option(CG_RT_WITH_NCCL
    "Build the NCCL bridge for cross-GPU peer-buffer registration + collectives (requires NCCL >= 2.23)"
    OFF)

if(CG_RT_WITH_CUDA)
    # Prefer find_package(CUDAToolkit) — CMake 3.17+. We already
    # require CMake 3.16 above, but every Linux install with CUDA
    # 11+ ships CMake 3.17 via the distro. Fall back gracefully if
    # not available so non-CUDA hosts still build the library.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        list(APPEND CG_RT_SOURCES src/drivers/cuda/cuda_driver.c)
        # Phase-4 ETC primitives + persistent-megakernel launcher +
        # live device probe. These are .cu files that need the CUDA
        # language enabled; we do that conditionally so non-CUDA
        # builds stay pure-C.
        list(APPEND CG_RT_CUDA_SOURCES
            src/drivers/cuda/event_tensor.cu
            src/drivers/cuda/dynamic_queue.cu
            src/drivers/cuda/megakernel_launcher.cu
            src/drivers/cuda/probe.cu
        )
        # Phase-4b: NCCL bridge for cross-GPU comms. Optional —
        # opt in via -DCG_RT_WITH_NCCL=ON. The bridge wraps
        # ncclCommInitAll for single-process multi-device init and
        # ``cuCtxEnablePeerAccess`` for peer atomics over PCIe-NODE.
        if(CG_RT_WITH_NCCL)
            list(APPEND CG_RT_SOURCES src/drivers/cuda/nccl_bridge.c)
            list(APPEND CG_RT_CUDA_SOURCES
                src/drivers/cuda/multi_etensor.cu
            )
        endif()
        enable_language(CUDA)
        # Default to compiling for Hopper (sm_90) + datacenter Blackwell
        # (sm_100) + workstation Blackwell (sm_120). Caller can override
        # with -DCMAKE_CUDA_ARCHITECTURES.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "90;100;120")
        endif()
        # Separable compilation for cooperative-launch patterns +
        # cluster-launch attribute support.
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(CG_RT_HAVE_CUDA TRUE)
        message(STATUS "xpu_rt: CUDA driver enabled "
                       "(toolkit ${CUDAToolkit_VERSION}, "
                       "archs ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "xpu_rt: CUDA toolkit not found — "
                       "CUDA driver disabled")
    endif()
endif()

# Shared library — Python ctypes loads this. Only built on posix.
if(CG_RT_PLATFORM STREQUAL "posix")
    # When CUDA is available we attach the .cu sources to the same
    # library target so the Python ctypes loader sees one shared
    # object with both C and CUDA symbols. Separable-compilation +
    # device-code linking is set up earlier (CMAKE_CUDA_SEPARABLE_COMPILATION).
    set(CG_RT_LIB_SOURCES ${CG_RT_SOURCES})
    if(CG_RT_HAVE_CUDA)
        list(APPEND CG_RT_LIB_SOURCES ${CG_RT_CUDA_SOURCES})
    endif()
    add_library(xpu_rt SHARED ${CG_RT_LIB_SOURCES})
    target_include_directories(xpu_rt PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(xpu_rt PRIVATE ${CG_RT_PLATFORM_INCLUDE_DIR})
    target_compile_options(xpu_rt PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    )
    if(CG_RT_HAVE_CUDA)
        # CUDA-language flags only apply to .cu translation units.
        # --use_fast_math safe for our atomic-ops + spin code.
        target_compile_options(xpu_rt PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>
        )
        # Device-link separable-compilation objects so cooperative
        # launches + cluster-launch attributes resolve.
        set_target_properties(xpu_rt PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
endif()

# Static library — usable on bare-metal / Zephyr cross-compiles.
add_library(xpu_rt_static STATIC ${CG_RT_SOURCES})
target_include_directories(xpu_rt_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(xpu_rt_static PRIVATE ${CG_RT_PLATFORM_INCLUDE_DIR})
target_compile_options(xpu_rt_static PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
)

if(CG_RT_PLATFORM STREQUAL "posix")
    find_package(Threads REQUIRED)
    target_link_libraries(xpu_rt        PUBLIC Threads::Threads)
    target_link_libraries(xpu_rt_static PUBLIC Threads::Threads)
endif()

if(CG_RT_HAVE_CUDA AND CG_RT_PLATFORM STREQUAL "posix")
    target_compile_definitions(xpu_rt        PRIVATE CG_RT_WITH_CUDA)
    target_compile_definitions(xpu_rt_static PRIVATE CG_RT_WITH_CUDA)
    target_include_directories(xpu_rt        PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(xpu_rt_static PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(xpu_rt        PRIVATE CUDA::cuda_driver)
    target_link_libraries(xpu_rt_static PRIVATE CUDA::cuda_driver)

    # NVRTC: statically link to insulate our PTX generation from
    # process-wide symbol conflicts. A typical Python process that
    # also imports PyTorch will have a different NVRTC (e.g. 12.8)
    # already loaded via pip's ``nvidia-cuda-nvrtc`` wheel; dynamically
    # binding would let torch's version satisfy our nvrtcCompile
    # calls, and its PTX output may target a newer driver ISA than
    # the system ``libcuda.so`` accepts — surfacing as
    # CUDA_ERROR_UNSUPPORTED_PTX_VERSION only in mixed test runs.
    # Static NVRTC ships with the CUDA toolkit and pulls in the
    # ptxcompiler static library.
    find_library(CG_RT_NVRTC_STATIC
        NAMES nvrtc_static
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64
              ${CUDAToolkit_TARGET_DIR}/lib64)
    find_library(CG_RT_NVRTC_BUILTINS_STATIC
        NAMES nvrtc-builtins_static
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64
              ${CUDAToolkit_TARGET_DIR}/lib64)
    find_library(CG_RT_NVPTXCOMPILER_STATIC
        NAMES nvptxcompiler_static
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64
              ${CUDAToolkit_TARGET_DIR}/lib64)
    if(CG_RT_NVRTC_STATIC AND CG_RT_NVPTXCOMPILER_STATIC AND CG_RT_NVRTC_BUILTINS_STATIC)
        message(STATUS "xpu_rt: using static NVRTC at ${CG_RT_NVRTC_STATIC}")
        # --exclude-libs hides the static NVRTC's internal symbols
        # from the .so's export table so they can't be stolen by a
        # later dlopen of a different libnvrtc.
        target_link_libraries(xpu_rt PRIVATE
            -Wl,--exclude-libs=libnvrtc_static.a
            -Wl,--exclude-libs=libnvptxcompiler_static.a
            -Wl,--exclude-libs=libnvrtc-builtins_static.a
            ${CG_RT_NVRTC_STATIC}
            ${CG_RT_NVRTC_BUILTINS_STATIC}
            ${CG_RT_NVPTXCOMPILER_STATIC}
            dl stdc++
        )
        target_link_libraries(xpu_rt_static PRIVATE
            ${CG_RT_NVRTC_STATIC}
            ${CG_RT_NVRTC_BUILTINS_STATIC}
            ${CG_RT_NVPTXCOMPILER_STATIC}
            dl stdc++
        )
    else()
        message(STATUS "xpu_rt: static NVRTC not found, falling back to shared")
        target_link_libraries(xpu_rt        PRIVATE CUDA::nvrtc)
        target_link_libraries(xpu_rt_static PRIVATE CUDA::nvrtc)
    endif()

    # Phase-4b NCCL bridge — optional. Header found via the system
    # CUDA toolkit (ABI-compatible with the pip nvidia-nccl-cu13 the
    # process actually loads at runtime). Library linked via SONAME
    # so whichever ``libnccl.so.2`` is in LD_LIBRARY_PATH at process
    # start (typically torch's pip-shipped cu13) gets used.
    if(CG_RT_WITH_NCCL)
        find_path(CG_RT_NCCL_INCLUDE_DIR nccl.h
            HINTS
              ${CUDAToolkit_INCLUDE_DIRS}
              /usr/local/cuda/include
              /usr/include)
        find_library(CG_RT_NCCL_LIB
            NAMES nccl
            HINTS
              ${CUDAToolkit_LIBRARY_DIR}
              /usr/local/cuda/lib64
              /usr/lib/x86_64-linux-gnu)
        if(CG_RT_NCCL_INCLUDE_DIR AND CG_RT_NCCL_LIB)
            message(STATUS "xpu_rt: NCCL bridge enabled (header ${CG_RT_NCCL_INCLUDE_DIR}/nccl.h, lib ${CG_RT_NCCL_LIB})")
            target_include_directories(xpu_rt        PRIVATE ${CG_RT_NCCL_INCLUDE_DIR})
            target_include_directories(xpu_rt_static PRIVATE ${CG_RT_NCCL_INCLUDE_DIR})
            target_link_libraries(xpu_rt        PRIVATE ${CG_RT_NCCL_LIB})
            target_link_libraries(xpu_rt_static PRIVATE ${CG_RT_NCCL_LIB})
            target_compile_definitions(xpu_rt        PRIVATE CG_RT_WITH_NCCL)
            target_compile_definitions(xpu_rt_static PRIVATE CG_RT_WITH_NCCL)
        else()
            message(FATAL_ERROR
                "CG_RT_WITH_NCCL=ON but NCCL header or library not found. "
                "Set NCCL_HOME or install the nvidia-nccl-cu13 wheel + system "
                "CUDA toolkit headers.")
        endif()
    endif()
endif()

# --- Optional XNNPACK driver ----------------------------------------------
# When ON, the libxpu_rt build pulls in third_party/XNNPACK as a sub-
# project, links the static XNNPACK library, and defines XPURT_HAS_XNNPACK
# so the bridge translation unit (src/drivers/xnnpack/xnnpack_bridge.c)
# forwards to xnn_*. When OFF (default), the bridge keeps compiling but
# every entry returns -XPU_RT_XNN_ENOTSUP and create() returns NULL —
# the python provider's probe() detects this honestly.
option(XPURT_WITH_XNNPACK "Build the XNNPACK CPU-operator driver" OFF)

# Detect bare-metal cross-compile (set by the riscv64-spike-rvv.cmake
# toolchain file). When true: no glibc, no pthreads, no cpuinfo — XNNPACK's
# RVV path uses inline `vsetvli` for vector-length detection and our
# single-threaded pthreadpool shim takes the parallelize calls serial.
if(NOT DEFINED XPURT_RISCV_SPIKE)
    set(XPURT_RISCV_SPIKE FALSE)
endif()
# Auto-detect: any riscv64 cross-compile is a Spike+pk target unless
# the caller opts out explicitly.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" AND CMAKE_CROSSCOMPILING)
    set(XPURT_RISCV_SPIKE TRUE)
endif()
if(XPURT_RISCV_SPIKE)
    message(STATUS "xpu_rt: cross-compiling for riscv64+RVV (Spike+pk)")
endif()

set(XPURT_HAVE_XNNPACK FALSE)
if(XPURT_WITH_XNNPACK)
    set(XNNPACK_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/XNNPACK"
        CACHE PATH "Path to the vendored XNNPACK source tree")
    if(NOT EXISTS "${XNNPACK_ROOT_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "XPURT_WITH_XNNPACK=ON but no XNNPACK CMakeLists at "
            "${XNNPACK_ROOT_DIR}. Run "
            "`git submodule update --init --recursive third_party/XNNPACK`.")
    endif()

    # XNNPACK's own option set — turn off everything we don't need.
    set(XNNPACK_LIBRARY_TYPE "static" CACHE STRING "" FORCE)
    set(XNNPACK_BUILD_LIBRARY ON  CACHE BOOL "" FORCE)
    set(XNNPACK_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    # Pull microkernel set; XNNPACK chooses per-host automatically.
    set(XNNPACK_BUILD_ALL_MICROKERNELS ON CACHE BOOL "" FORCE)
    if(XPURT_RISCV_SPIKE)
        # RVV ON: the Zephyr SDK 1.0.0-beta1 ships GCC 14.3 which
        # implements the correct RVV v1.0 intrinsic spellings the
        # UCB-BAR XNNPACK fork uses. (Chipyard's bundled GCC 13.2 had
        # intrinsic drift across ~250 microkernel files — see the
        # toolchain file at toolchains/riscv64-spike-rvv.cmake for the
        # full toolchain-choice rationale.)
        set(XNNPACK_ENABLE_RISCV_VECTOR ON  CACHE BOOL "" FORCE)
        set(XNNPACK_ENABLE_RISCV_FP16_VECTOR OFF CACHE BOOL "" FORCE)
        # cpuinfo doesn't usefully cross-compile to riscv64 (no /proc,
        # no auxv on bare-metal pk); inline `vsetvli` detection inside
        # XNNPACK's hardware-config still works for runtime VLEN
        # discovery if RVV were enabled.
        set(XNN_ENABLE_CPUINFO "0" CACHE STRING "" FORCE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Force cpuinfo include path on host Linux. The fork's hardcoded =0
        # leaves hardware-config.c with implicit-decl external refs to
        # cpuinfo_has_x86_* (since cpuinfo.h is suppressed) — set to 1 so
        # the header is included and the predicates inline cleanly.
        set(XNN_ENABLE_CPUINFO "1" CACHE STRING "" FORCE)
    endif()
    # The UCB-BAR torch-1.0.0-bump branch defaults USE_GNU_SOURCE=OFF
    # (it targets bare-metal / Zephyr where _GNU_SOURCE is wrong).
    # When the host is Linux (the common XPU-RT build), forcing ON is
    # required so memory.c sees MREMAP_MAYMOVE / MAP_FAILED via <sys/mman.h>.
    # On bare-metal riscv we leave it OFF — newlib has neither.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(USE_GNU_SOURCE ON CACHE BOOL "" FORCE)
    endif()

    # On bare-metal riscv we ship a single-threaded pthreadpool target
    # BEFORE XNNPACK pulls its own. XNNPACK guards its FetchContent
    # block with IF(NOT TARGET pthreadpool), so an in-tree target
    # short-circuits the download. We use pthreadpool's own shim.c —
    # the serial single-threaded fallback path it ships for wasm /
    # platforms without pthread. pthreadpool's source still needs to be
    # available for the header; we FetchContent it explicitly.
    if(XPURT_RISCV_SPIKE)
        if(NOT TARGET pthreadpool)
            include(FetchContent)
            FetchContent_Declare(
                pthreadpool_src
                GIT_REPOSITORY https://github.com/google/pthreadpool.git
                GIT_TAG        main
                GIT_SHALLOW    TRUE
            )
            FetchContent_GetProperties(pthreadpool_src)
            if(NOT pthreadpool_src_POPULATED)
                FetchContent_Populate(pthreadpool_src)
            endif()
            add_library(pthreadpool STATIC
                ${pthreadpool_src_SOURCE_DIR}/src/shim.c)
            target_include_directories(pthreadpool PUBLIC
                ${pthreadpool_src_SOURCE_DIR}/include)
            target_compile_definitions(pthreadpool PRIVATE
                PTHREADPOOL_NO_DEPRECATED_API=1)
            # The shim's PTHREADPOOL_PRIVATE_IMPL machinery expects
            # the dispatch-table header. shim.c is self-contained;
            # this include path keeps threadpool-common.h / -utils.h
            # discoverable.
            target_include_directories(pthreadpool PRIVATE
                ${pthreadpool_src_SOURCE_DIR}/src)
        endif()
    endif()

    # cpuinfo dependency. The fork's CMakeLists commented out its own
    # FetchContent block for cpuinfo and hardcoded XNN_ENABLE_CPUINFO=0,
    # but src/configs/hardware-config.c still calls cpuinfo_has_x86_*
    # unconditionally on x86. We pull cpuinfo here so XNNPACK's
    # hardware-config has the symbols to link against. cpuinfo is small
    # (<1MB), BSD-2 licensed, and a standard XNNPACK dependency upstream.
    # On bare-metal riscv we skip cpuinfo entirely — XNN_ENABLE_CPUINFO=0
    # is the canonical setup for that target and the fork's RVV detection
    # uses inline `vsetvli` instead of cpuinfo predicates.
    if(NOT XPURT_RISCV_SPIKE AND NOT TARGET cpuinfo)
        include(FetchContent)
        FetchContent_Declare(
            cpuinfo
            GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
            GIT_TAG        main
            GIT_SHALLOW    TRUE
        )
        set(CPUINFO_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_UNIT_TESTS  OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_MOCK_TESTS  OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_BENCHMARKS  OFF CACHE BOOL "" FORCE)
        set(CPUINFO_LIBRARY_TYPE      "static" CACHE STRING "" FORCE)
        FetchContent_MakeAvailable(cpuinfo)
    endif()
    # Some downstream consumers of XNNPACK want their own pthreadpool;
    # we accept whatever XNNPACK fetches. Mark EXCLUDE_FROM_ALL so its
    # tests / benchmarks don't show up in the default build.
    add_subdirectory(${XNNPACK_ROOT_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}/_deps/xnnpack
                     EXCLUDE_FROM_ALL)
    if(TARGET XNNPACK)
        # XNNPACK's hardware-config / microkernels reference cpuinfo;
        # the fork's CMakeLists has the cpuinfo dependency block commented
        # out, so we wire it up here AFTER add_subdirectory(). cpuinfo
        # exposes its include dir via its INTERFACE; PUBLIC-linking it on
        # the relevant xnnpack-* targets propagates -I and the symbols.
        if(TARGET cpuinfo)
            foreach(_xnn_tgt XNNPACK xnnpack-hardware-config
                              xnnpack-microkernels-prod xnnpack-microkernels-all
                              xnnpack-operator-run xnnpack-operators
                              xnnpack-subgraph)
                if(TARGET ${_xnn_tgt})
                    target_link_libraries(${_xnn_tgt} PRIVATE cpuinfo)
                endif()
            endforeach()
        endif()
        if(XPURT_RISCV_SPIKE)
            # pthread_compat.c provides single-threaded no-op pthread_*
            # implementations the chipyard newlib riscv64-unknown-elf
            # toolchain does not ship. Linked into xpu_rt_static so
            # XNNPACK's pthread_once / pthread_mutex_* references
            # resolve at exe link time.
            target_sources(xpu_rt_static PRIVATE
                src/compat/picolibc-freestanding/pthread_compat.c)
            # Inject our picolibc compat directory on every XNNPACK
            # target (mostly for the std::math compat <cmath> shim;
            # pthread.h is supplied by newlib once _POSIX_THREADS is
            # defined via the toolchain file).
            set(_xnn_pthread_inc
                "${CMAKE_CURRENT_SOURCE_DIR}/src/compat/picolibc-freestanding")
            foreach(_xnn_tgt XNNPACK xnnpack-base xnnpack-hardware-config
                              xnnpack-microkernels-prod xnnpack-microkernels-all
                              xnnpack-operator-run xnnpack-operators
                              xnnpack-subgraph xnnpack-memory xnnpack-logging
                              xnnpack-cache xnnpack-mutex xnnpack-pack
                              xnnpack-config xnnpack-allocator xnnpack-datatype
                              xnnpack-indirection xnnpack-microparams-init
                              xnnpack-normalization xnnpack-operator-utils
                              xnnpack-post-operation xnnpack-quantization
                              xnnpack-reference-ukernels xnnpack-runtime
                              xnnpack-table xnnpack-sanitizers)
                if(TARGET ${_xnn_tgt})
                    get_target_property(_xnn_kind ${_xnn_tgt} TYPE)
                    if(_xnn_kind STREQUAL "INTERFACE_LIBRARY")
                        target_include_directories(${_xnn_tgt} BEFORE INTERFACE
                            ${_xnn_pthread_inc})
                    else()
                        target_include_directories(${_xnn_tgt} BEFORE PRIVATE
                            ${_xnn_pthread_inc})
                    endif()
                endif()
            endforeach()
            # pthreadpool target also needs the stub (its shim.c
            # transitively includes <pthread.h> in some configurations).
            if(TARGET pthreadpool)
                target_include_directories(pthreadpool BEFORE PUBLIC
                    ${_xnn_pthread_inc})
            endif()

            # GCC 14.3 (Zephyr SDK) compiles XNNPACK's RVV microkernels
            # cleanly — no source filtering needed. fp16-arith RVV
            # kernels (`rvvfp16arith`) require zvfh, which Spike at
            # VLEN=128 doesn't expose; drop just those.
            foreach(_xnn_tgt xnnpack-microkernels-prod xnnpack-microkernels-all)
                if(NOT TARGET ${_xnn_tgt})
                    continue()
                endif()
                get_target_property(_srcs ${_xnn_tgt} SOURCES)
                set(_kept "")
                set(_dropped_count 0)
                foreach(_s ${_srcs})
                    if(_s MATCHES "rvvfp16arith")
                        math(EXPR _dropped_count "${_dropped_count} + 1")
                    else()
                        list(APPEND _kept "${_s}")
                    endif()
                endforeach()
                set_target_properties(${_xnn_tgt} PROPERTIES SOURCES "${_kept}")
                message(STATUS
                    "xpu_rt: dropped ${_dropped_count} rvvfp16arith "
                    "microkernels from ${_xnn_tgt} (Spike has no zvfh)")
            endforeach()
        endif()
        if(TARGET xpu_rt)
            target_compile_definitions(xpu_rt        PRIVATE XPURT_HAS_XNNPACK=1)
            target_include_directories(xpu_rt        PRIVATE ${XNNPACK_ROOT_DIR}/include)
        endif()
        target_compile_definitions(xpu_rt_static PRIVATE XPURT_HAS_XNNPACK=1)
        target_include_directories(xpu_rt_static PRIVATE ${XNNPACK_ROOT_DIR}/include)
        if(XPURT_RISCV_SPIKE)
            # riscv64 bare-metal (Zephyr SDK picolibc): no cpuinfo
            # (XNNPACK fork has it disabled on riscv anyway).
            # pthreadpool is our single-threaded shim (target
            # pre-defined before XNNPACK). picolibc supplies libm via
            # specs files; libstdc++ via the toolchain.
            target_compile_definitions(xpu_rt_static PRIVATE XPURT_RISCV_SPIKE=1)
            target_link_libraries(xpu_rt_static PUBLIC
                XNNPACK pthreadpool stdc++ m)
        else()
            # Host (Linux) path. XNNPACK builds a stack of separate
            # static libs (XNNPACK, xnnpack-microkernels-prod,
            # xnnpack-microkernels-all, ...) and the microkernels libs
            # reference cpuinfo unconditionally on x86. When libxpu_rt
            # is a shared lib, the linker is single-pass: we must put
            # cpuinfo AFTER every XNNPACK-* lib so its symbols resolve.
            # The cleanest way is `-Wl,--start-group ... --end-group`
            # so the linker iterates until fixed-point. libstdc++
            # needed because XNNPACK includes C++ sources.
            target_link_libraries(xpu_rt        PRIVATE
                -Wl,--start-group
                XNNPACK cpuinfo
                -Wl,--end-group
                stdc++)
            target_link_libraries(xpu_rt_static PRIVATE
                XNNPACK cpuinfo stdc++)
            # Downstream consumers (test binaries, user apps) of
            # libxpu_rt.so need cpuinfo in their own link line too,
            # because libxpu_rt.so may leave some cpuinfo refs
            # unresolved (shared libs allow undefined refs at .so
            # build time and resolve them at exe link).
            target_link_libraries(xpu_rt        INTERFACE cpuinfo)
            target_link_libraries(xpu_rt_static INTERFACE cpuinfo)
        endif()
        set(XPURT_HAVE_XNNPACK TRUE)
        message(STATUS "xpu_rt: XNNPACK driver enabled (UCB-BAR fork @ ${XNNPACK_ROOT_DIR})")
    else()
        message(FATAL_ERROR
            "XPURT_WITH_XNNPACK=ON but the XNNPACK CMake target was not "
            "produced by ${XNNPACK_ROOT_DIR}/CMakeLists.txt. Check that "
            "the submodule is fully populated.")
    endif()
endif()

# Export headers and artefacts relative to this CMakeLists so a parent
# build can simply ``add_subdirectory(runtime/native/libxpu_rt)``
# and link against ``xpu_rt``.

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

# Host tests run only on posix x86_64 — cpu_task tests need threads,
# the others all share the same test binary shape. When cross-compiling
# to riscv64+spike we instead build a dedicated XNNPACK bridge smoke
# test under spike+pk.
if(CG_RT_PLATFORM STREQUAL "posix" AND NOT XPURT_RISCV_SPIKE)
    set(CG_RT_TESTS
        tests/test_semaphore.c
        tests/test_event_tensor.c
        tests/test_command_buffer.c
        tests/test_integration.c
        tests/test_cpu_task.c
        tests/test_phase_g_scaffolds.c
    )
    if(CG_RT_HAVE_CUDA)
        list(APPEND CG_RT_TESTS tests/test_cuda.c)
    endif()

    foreach(test_src ${CG_RT_TESTS})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE xpu_rt)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# --- riscv64 + Spike + RVV smoke test (drives FC through the bridge) ----
# Built when cross-compiling to riscv64 with XNNPACK enabled. Newlib's
# htif_nano.specs (configured in the toolchain file) provides crt0,
# printf via HTIF, and exit via HTIF tohost — no manual startup.S or
# syscalls.c needed. The CMAKE_C_LINK_EXECUTABLE override in the
# toolchain handles the -specs / -T flags.
if(XPURT_RISCV_SPIKE AND XPURT_HAVE_XNNPACK)
    add_executable(test_xnnpack_bridge_riscv
        tests/test_xnnpack_bridge_riscv.c)
    target_link_libraries(test_xnnpack_bridge_riscv PRIVATE
        xpu_rt_static)
    target_include_directories(test_xnnpack_bridge_riscv PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(test_xnnpack_bridge_riscv PROPERTIES
        SUFFIX ".elf")
    message(STATUS "xpu_rt: built riscv64+RVV XNNPACK bridge smoke test")
endif()

# --- Bare-metal smoke test (freestanding RISC-V, runs on Spike) ---------
if(CG_RT_PLATFORM STREQUAL "bare")
    add_executable(smoke_bare
        tests/bare/startup.S
        tests/bare/syscalls.c
        tests/bare/smoke.c
    )
    set_source_files_properties(tests/bare/startup.S PROPERTIES LANGUAGE C)
    target_link_libraries(smoke_bare PRIVATE xpu_rt_static)
    target_include_directories(smoke_bare PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_options(smoke_bare PRIVATE
        -T${CMAKE_CURRENT_SOURCE_DIR}/tests/bare/link.ld
        -nostartfiles
        -static
        -Wl,-Map=smoke_bare.map
    )
    set_target_properties(smoke_bare PROPERTIES SUFFIX ".elf")
endif()
