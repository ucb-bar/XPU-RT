cmake_minimum_required(VERSION 3.16)
project(compgen_runtime C)

# =============================================================================
# Merged XPU-RT + CompGen native runtime
#
# This directory contains:
#
#   1. The CompGen native runtime library (src/, include/, native/, templates/)
#      built as compgen_runtime / compgen_runtime_debug static libs. Always
#      built; no Merlin/IREE dependency. Tracing + perf instrumentation are
#      opt-in via CG_TRACE_ENABLED / CG_PERF_BACKEND.
#
#   2. The XPU-RT dispatch runners (tools/json_dispatch_runner.c,
#      tools/xpurt_scheduler_runner.c) that link against the Merlin-built
#      xpu-rt standalone archive (libxpurt_standalone.a). Opt-in — only built
#      if XPURT_STANDALONE_LIB_PATH is set.
#
# Common workflows:
#
#   # Just the CompGen native lib (host build):
#   cmake -B build -S runtime
#   cmake --build build --target compgen_runtime
#
#   # XPU-RT dispatch runners on top of Merlin/IREE:
#   cmake -B build -S runtime \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DXPURT_STANDALONE_LIB_PATH=../merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_standalone.a
#   cmake --build build --target json_dispatch_runner xpurt_scheduler_runner
#
# For Zephyr: cross-compile merlin with a Zephyr-compatible profile, then
# point XPURT_STANDALONE_LIB_PATH at the resulting libxpurt_standalone.a.
# =============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
find_package(Threads)

# ---------------------------------------------------------------------------
# CompGen native runtime — options
# ---------------------------------------------------------------------------

option(CG_TRACE_ENABLED "Enable trace instrumentation" OFF)
option(CG_TRACE_ZEPHYR  "Route tracing through Zephyr sys_trace" OFF)

set(CG_PERF_BACKEND "none" CACHE STRING
    "Performance counter backend: none, linux_perf, zephyr_timing, bare_metal_csr, cuda_cupti")
set(CG_INSTRUMENTATION_LEVEL "0" CACHE STRING
    "Instrumentation level: 0=none, 1=op, 2=tile, 3=full")

# ---------------------------------------------------------------------------
# Core runtime library (always built, zero instrumentation overhead)
# ---------------------------------------------------------------------------

set(RUNTIME_CORE_SOURCES
  src/engine.c
  src/task.c
  src/transport_local.c
  src/arena.c
  src/ops.c
)

add_library(compgen_runtime STATIC ${RUNTIME_CORE_SOURCES})

target_include_directories(compgen_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(compgen_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(Threads_FOUND)
  target_link_libraries(compgen_runtime PRIVATE Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# Debug/instrumented variant
# ---------------------------------------------------------------------------

set(RUNTIME_DEBUG_SOURCES
  ${RUNTIME_CORE_SOURCES}
  src/trace.c
  src/perf_counters.c
)

add_library(compgen_runtime_debug STATIC ${RUNTIME_DEBUG_SOURCES})

target_include_directories(compgen_runtime_debug PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(compgen_runtime_debug PRIVATE -Wall -Wextra -Wpedantic)

target_compile_definitions(compgen_runtime_debug PRIVATE
  CG_TRACE_ENABLED
  CG_INSTRUMENTATION_LEVEL=${CG_INSTRUMENTATION_LEVEL}
)

if(CG_PERF_BACKEND STREQUAL "linux_perf")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_LINUX_PERF)
elseif(CG_PERF_BACKEND STREQUAL "zephyr_timing")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_ZEPHYR_TIMING)
elseif(CG_PERF_BACKEND STREQUAL "bare_metal_csr")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_BARE_METAL_CSR)
elseif(CG_PERF_BACKEND STREQUAL "cuda_cupti")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_CUDA_CUPTI)
endif()

if(CG_TRACE_ZEPHYR)
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_TRACE_ZEPHYR)
endif()

if(Threads_FOUND)
  target_link_libraries(compgen_runtime_debug PRIVATE Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# Optional: also apply instrumentation to the main library if requested
# ---------------------------------------------------------------------------

if(CG_TRACE_ENABLED)
  target_sources(compgen_runtime PRIVATE
    src/trace.c
    src/perf_counters.c
  )
  target_compile_definitions(compgen_runtime PRIVATE
    CG_TRACE_ENABLED
    CG_INSTRUMENTATION_LEVEL=${CG_INSTRUMENTATION_LEVEL}
  )
  if(CG_PERF_BACKEND STREQUAL "linux_perf")
    target_compile_definitions(compgen_runtime PRIVATE CG_PERF_BACKEND_LINUX_PERF)
  elseif(CG_PERF_BACKEND STREQUAL "bare_metal_csr")
    target_compile_definitions(compgen_runtime PRIVATE CG_PERF_BACKEND_BARE_METAL_CSR)
  endif()
  if(CG_TRACE_ZEPHYR)
    target_compile_definitions(compgen_runtime PRIVATE CG_TRACE_ZEPHYR)
  endif()
endif()

# =============================================================================
# XPU-RT Merlin/IREE dispatch runners (opt-in via XPURT_STANDALONE_LIB_PATH)
# =============================================================================

set(XPURT_STANDALONE_LIB_PATH "" CACHE FILEPATH
    "Path to libxpurt_standalone.a (xpu-rt + IREE runtime objects) built by Merlin/IREE")
set(XPURT_FLATCC_PARSING_LIB_PATH "" CACHE FILEPATH
    "Optional path to FlatCC parsing/verifier archive (libflatcc_parsing.a) if not auto-detected")

if(XPURT_STANDALONE_LIB_PATH)
  if(NOT Threads_FOUND)
    find_package(Threads REQUIRED)
  endif()

  # Build up extra archives that must come *after* the xpurt standalone archive.
  set(_XPURT_EXTRA_ARCHIVES "")
  if(XPURT_FLATCC_PARSING_LIB_PATH)
    list(APPEND _XPURT_EXTRA_ARCHIVES "${XPURT_FLATCC_PARSING_LIB_PATH}")
  endif()

  # -------------------------------------------------------------------------
  # json_dispatch_runner — baseline topo-order dispatch runner
  # -------------------------------------------------------------------------
  add_executable(json_dispatch_runner
    tools/json_dispatch_runner.c
  )

  target_include_directories(json_dispatch_runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../merlin/samples/common"
  )

  target_link_libraries(json_dispatch_runner PRIVATE
    "-Wl,--whole-archive"
    "${XPURT_STANDALONE_LIB_PATH}"
    "-Wl,--no-whole-archive"
    ${_XPURT_EXTRA_ARCHIVES}
  )
  target_link_libraries(json_dispatch_runner PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
  )

  # -------------------------------------------------------------------------
  # xpurt_scheduler_runner — two-cluster scheduled dispatch runner
  # -------------------------------------------------------------------------
  add_executable(xpurt_scheduler_runner
    tools/xpurt_scheduler_runner.c
  )

  target_include_directories(xpurt_scheduler_runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../merlin/samples/common"
  )

  target_link_libraries(xpurt_scheduler_runner PRIVATE
    "-Wl,--whole-archive"
    "${XPURT_STANDALONE_LIB_PATH}"
    "-Wl,--no-whole-archive"
    ${_XPURT_EXTRA_ARCHIVES}
  )
  target_link_libraries(xpurt_scheduler_runner PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m
  )
else()
  message(STATUS "XPURT_STANDALONE_LIB_PATH not set; skipping XPU-RT dispatch runners.")
  message(STATUS "Building only the CompGen native runtime libraries. Set "
                 "-DXPURT_STANDALONE_LIB_PATH=<path/to/libxpurt_standalone.a> to build "
                 "json_dispatch_runner and xpurt_scheduler_runner.")
endif()
