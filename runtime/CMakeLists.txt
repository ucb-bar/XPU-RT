cmake_minimum_required(VERSION 3.16)
project(XpuRtRuntime C)

# This directory can build custom runtime tools that link against the
# Merlin-built xpu-rt standalone archive (libxpurt_standalone.a).
#
# For most workflows, use the pre-built binaries from merlin directly:
#   merlin-baseline-async        (baseline topo-order dispatch runner)
#   merlin-dispatch-scheduler    (two-cluster scheduled dispatch runner)
#
# These are built automatically by `merlin build --profile spacemit` and
# live in merlin/build/<profile>/runtime/plugins/merlin-samples/.
#
# For custom tools, link against the standalone archive:
#   cmake -B build -S runtime \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DXPURT_STANDALONE_LIB_PATH=../merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_standalone.a
#   cmake --build build --target json_dispatch_runner
#
# For Zephyr: cross-compile merlin with a Zephyr-compatible profile, then
# point XPURT_STANDALONE_LIB_PATH at the resulting libxpurt_standalone.a.
# Use Zephyr's zephyr_library_import_from_static() or
# target_link_libraries() to link it into your Zephyr application.

set(XPURT_STANDALONE_LIB_PATH "" CACHE FILEPATH
    "Path to libxpurt_standalone.a (xpu-rt + IREE runtime objects) built by Merlin/IREE")
set(XPURT_FLATCC_PARSING_LIB_PATH "" CACHE FILEPATH
    "Optional path to FlatCC parsing/verifier archive (libflatcc_parsing.a) if not auto-detected")

if(NOT XPURT_STANDALONE_LIB_PATH)
  message(FATAL_ERROR
    "Set XPURT_STANDALONE_LIB_PATH to libxpurt_standalone.a. "
    "This is built by: merlin build --profile spacemit --config perf")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(json_dispatch_runner
  tools/json_dispatch_runner.c
)

# Header-only dependency on the xpu-rt public API inside the Merlin tree.
target_include_directories(json_dispatch_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../merlin/samples/common"
)

# Build up extra archives that must come *after* the xpurt standalone archive.
set(_XPURT_EXTRA_ARCHIVES "")

# FlatBuffers verification helpers in IREE VM use FlatCC verifier functions.
# In our build, these are in libflatcc_parsing.a (not libflatcc_runtime.a).
if(XPURT_FLATCC_PARSING_LIB_PATH)
  list(APPEND _XPURT_EXTRA_ARCHIVES "${XPURT_FLATCC_PARSING_LIB_PATH}")
endif()

# Link the standalone archive (xpu-rt + IREE objects). We whole-archive it to
# keep driver/registry objects that may otherwise be dropped by the linker.
target_link_libraries(json_dispatch_runner PRIVATE
  "-Wl,--whole-archive"
  "${XPURT_STANDALONE_LIB_PATH}"
  "-Wl,--no-whole-archive"
  ${_XPURT_EXTRA_ARCHIVES}
)

# System libs commonly needed by IREE runtime and the plugin.
target_link_libraries(json_dispatch_runner PRIVATE
  Threads::Threads
  ${CMAKE_DL_LIBS}
  m
)

#-------------------------------------------------------------------------------
# xpurt_scheduler_runner
#
# Two-cluster scheduled dispatch runner — same shape as the SpacemiT main.c
# in merlin/samples/SpacemiTX60/dispatch_scheduler/, but parameterized by a
# built-in target preset table so the same binary covers spacemit_x60,
# qrb5165, and host. Driven by:
#   ./xpurt_scheduler_runner schedule.json --target=<name> \
#       --vmfb_dir=<path> [--telemetry_jsonl=...] [--schedule_next=...]
#-------------------------------------------------------------------------------
add_executable(xpurt_scheduler_runner
  tools/xpurt_scheduler_runner.c
)

target_include_directories(xpurt_scheduler_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../merlin/samples/common"
)

# Same whole-archive incantation as json_dispatch_runner — IREE's HAL
# drivers and executable loaders register via static initializers /
# weak symbols and would be dropped by the linker without it.
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
