cmake_minimum_required(VERSION 3.16)
project(compgen_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Options — instrumentation and perf backend selection
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
)

add_library(compgen_runtime STATIC ${RUNTIME_CORE_SOURCES})

target_include_directories(compgen_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(compgen_runtime PRIVATE -Wall -Wextra -Wpedantic)

# Thread support for transport_local
find_package(Threads)
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

# Always enable tracing in the debug variant
target_compile_definitions(compgen_runtime_debug PRIVATE
  CG_TRACE_ENABLED
  CG_INSTRUMENTATION_LEVEL=${CG_INSTRUMENTATION_LEVEL}
)

# Perf backend selection for debug variant
if(CG_PERF_BACKEND STREQUAL "linux_perf")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_LINUX_PERF)
elseif(CG_PERF_BACKEND STREQUAL "zephyr_timing")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_ZEPHYR_TIMING)
elseif(CG_PERF_BACKEND STREQUAL "bare_metal_csr")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_BARE_METAL_CSR)
elseif(CG_PERF_BACKEND STREQUAL "cuda_cupti")
  target_compile_definitions(compgen_runtime_debug PRIVATE CG_PERF_BACKEND_CUDA_CUPTI)
endif()

# Zephyr trace routing
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
