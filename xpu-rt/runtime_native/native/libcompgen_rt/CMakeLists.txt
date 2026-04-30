cmake_minimum_required(VERSION 3.16)
project(compgen_rt C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# libcompgen_rt — the native HAL
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
message(STATUS "compgen_rt: platform backend = ${CG_RT_PLATFORM}")

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
        message(STATUS "compgen_rt: CUDA driver enabled "
                       "(toolkit ${CUDAToolkit_VERSION}, "
                       "archs ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "compgen_rt: CUDA toolkit not found — "
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
    add_library(compgen_rt SHARED ${CG_RT_LIB_SOURCES})
    target_include_directories(compgen_rt PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(compgen_rt PRIVATE ${CG_RT_PLATFORM_INCLUDE_DIR})
    target_compile_options(compgen_rt PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
    )
    if(CG_RT_HAVE_CUDA)
        # CUDA-language flags only apply to .cu translation units.
        # --use_fast_math safe for our atomic-ops + spin code.
        target_compile_options(compgen_rt PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>
        )
        # Device-link separable-compilation objects so cooperative
        # launches + cluster-launch attributes resolve.
        set_target_properties(compgen_rt PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
endif()

# Static library — usable on bare-metal / Zephyr cross-compiles.
add_library(compgen_rt_static STATIC ${CG_RT_SOURCES})
target_include_directories(compgen_rt_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(compgen_rt_static PRIVATE ${CG_RT_PLATFORM_INCLUDE_DIR})
target_compile_options(compgen_rt_static PRIVATE
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
)

if(CG_RT_PLATFORM STREQUAL "posix")
    find_package(Threads REQUIRED)
    target_link_libraries(compgen_rt        PUBLIC Threads::Threads)
    target_link_libraries(compgen_rt_static PUBLIC Threads::Threads)
endif()

if(CG_RT_HAVE_CUDA AND CG_RT_PLATFORM STREQUAL "posix")
    target_compile_definitions(compgen_rt        PRIVATE CG_RT_WITH_CUDA)
    target_compile_definitions(compgen_rt_static PRIVATE CG_RT_WITH_CUDA)
    target_include_directories(compgen_rt        PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(compgen_rt_static PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(compgen_rt        PRIVATE CUDA::cuda_driver)
    target_link_libraries(compgen_rt_static PRIVATE CUDA::cuda_driver)

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
        message(STATUS "compgen_rt: using static NVRTC at ${CG_RT_NVRTC_STATIC}")
        # --exclude-libs hides the static NVRTC's internal symbols
        # from the .so's export table so they can't be stolen by a
        # later dlopen of a different libnvrtc.
        target_link_libraries(compgen_rt PRIVATE
            -Wl,--exclude-libs=libnvrtc_static.a
            -Wl,--exclude-libs=libnvptxcompiler_static.a
            -Wl,--exclude-libs=libnvrtc-builtins_static.a
            ${CG_RT_NVRTC_STATIC}
            ${CG_RT_NVRTC_BUILTINS_STATIC}
            ${CG_RT_NVPTXCOMPILER_STATIC}
            dl stdc++
        )
        target_link_libraries(compgen_rt_static PRIVATE
            ${CG_RT_NVRTC_STATIC}
            ${CG_RT_NVRTC_BUILTINS_STATIC}
            ${CG_RT_NVPTXCOMPILER_STATIC}
            dl stdc++
        )
    else()
        message(STATUS "compgen_rt: static NVRTC not found, falling back to shared")
        target_link_libraries(compgen_rt        PRIVATE CUDA::nvrtc)
        target_link_libraries(compgen_rt_static PRIVATE CUDA::nvrtc)
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
            message(STATUS "compgen_rt: NCCL bridge enabled (header ${CG_RT_NCCL_INCLUDE_DIR}/nccl.h, lib ${CG_RT_NCCL_LIB})")
            target_include_directories(compgen_rt        PRIVATE ${CG_RT_NCCL_INCLUDE_DIR})
            target_include_directories(compgen_rt_static PRIVATE ${CG_RT_NCCL_INCLUDE_DIR})
            target_link_libraries(compgen_rt        PRIVATE ${CG_RT_NCCL_LIB})
            target_link_libraries(compgen_rt_static PRIVATE ${CG_RT_NCCL_LIB})
            target_compile_definitions(compgen_rt        PRIVATE CG_RT_WITH_NCCL)
            target_compile_definitions(compgen_rt_static PRIVATE CG_RT_WITH_NCCL)
        else()
            message(FATAL_ERROR
                "CG_RT_WITH_NCCL=ON but NCCL header or library not found. "
                "Set NCCL_HOME or install the nvidia-nccl-cu13 wheel + system "
                "CUDA toolkit headers.")
        endif()
    endif()
endif()

# Export headers and artefacts relative to this CMakeLists so a parent
# build can simply ``add_subdirectory(runtime/native/libcompgen_rt)``
# and link against ``compgen_rt``.

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

# Host tests run only on posix — cpu_task tests need threads, the
# others all share the same test binary shape.
if(CG_RT_PLATFORM STREQUAL "posix")
    set(CG_RT_TESTS
        tests/test_semaphore.c
        tests/test_event_tensor.c
        tests/test_command_buffer.c
        tests/test_integration.c
        tests/test_cpu_task.c
    )
    if(CG_RT_HAVE_CUDA)
        list(APPEND CG_RT_TESTS tests/test_cuda.c)
    endif()

    foreach(test_src ${CG_RT_TESTS})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE compgen_rt)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# --- Bare-metal smoke test (freestanding RISC-V, runs on Spike) ---------
if(CG_RT_PLATFORM STREQUAL "bare")
    add_executable(smoke_bare
        tests/bare/startup.S
        tests/bare/syscalls.c
        tests/bare/smoke.c
    )
    set_source_files_properties(tests/bare/startup.S PROPERTIES LANGUAGE C)
    target_link_libraries(smoke_bare PRIVATE compgen_rt_static)
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
