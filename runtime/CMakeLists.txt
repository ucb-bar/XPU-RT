cmake_minimum_required(VERSION 3.16)
project(XpuRtRuntime C)

# This directory builds a small runtime tool (`json_dispatch_runner`) that
# consumes the Merlin-built xpu-rt IREE wrapper library.
#
# Usage example (standalone archive is strongly recommended):
#   cmake -B build -S runtime \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DXPURT_STANDALONE_LIB_PATH=../merlin/build/spacemit-merlin-perf/runtime/src/iree/runtime/libxpurt_iree_plugin_standalone.a
#   cmake --build build --target json_dispatch_runner

set(XPURT_STANDALONE_LIB_PATH "" CACHE FILEPATH
    "Path to libxpurt_iree_plugin_standalone.a (plugin + IREE runtime objects) built by Merlin/IREE")
set(XPURT_LIB_PATH "" CACHE FILEPATH
    "Path to libxpurt_iree_plugin.a (plugin only). This usually cannot link out-of-tree without IREE targets.")
set(XPURT_FLATCC_PARSING_LIB_PATH "" CACHE FILEPATH
    "Optional path to FlatCC parsing/verifier archive (libflatcc_parsing.a) if not auto-detected")

set(_XPURT_LINK_LIB "")
set(_XPURT_WHOLE_ARCHIVE OFF)
if(XPURT_STANDALONE_LIB_PATH)
  set(_XPURT_LINK_LIB "${XPURT_STANDALONE_LIB_PATH}")
  set(_XPURT_WHOLE_ARCHIVE ON)
elseif(XPURT_LIB_PATH)
  message(WARNING
    "XPURT_LIB_PATH points at the plugin-only archive. Prefer XPURT_STANDALONE_LIB_PATH "
    "(libxpurt_iree_plugin_standalone.a), otherwise you may see undefined references to IREE.")
  set(_XPURT_LINK_LIB "${XPURT_LIB_PATH}")
else()
  message(FATAL_ERROR
    "Set XPURT_STANDALONE_LIB_PATH to libxpurt_iree_plugin_standalone.a (recommended). "
    "Alternatively set XPURT_LIB_PATH to libxpurt_iree_plugin.a (may not link out-of-tree).")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(json_dispatch_runner
  tools/json_dispatch_runner.c
)

# Header-only dependency on the xpu-rt public API inside the Merlin tree.
target_include_directories(json_dispatch_runner PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../merlin/xpu-rt"
)

# Build up extra archives that must come *after* the xpurt standalone archive.
set(_XPURT_EXTRA_ARCHIVES "")

# FlatBuffers verification helpers in IREE VM use FlatCC verifier functions.
# In our build, these are in libflatcc_parsing.a (not libflatcc_runtime.a).
if(XPURT_FLATCC_PARSING_LIB_PATH)
  list(APPEND _XPURT_EXTRA_ARCHIVES "${XPURT_FLATCC_PARSING_LIB_PATH}")
endif()

# Link the standalone archive (plugin + IREE objects). We whole-archive it to
# keep driver/registry objects that may otherwise be dropped by the linker.
# IMPORTANT: static archive link order matters (GNU ld), so we also append any
# extra archives *after* the standalone archive in the same call.
if(_XPURT_WHOLE_ARCHIVE)
  target_link_libraries(json_dispatch_runner PRIVATE
    "-Wl,--whole-archive"
    "${_XPURT_LINK_LIB}"
    "-Wl,--no-whole-archive"
    ${_XPURT_EXTRA_ARCHIVES}
  )
else()
  target_link_libraries(json_dispatch_runner PRIVATE
    "${_XPURT_LINK_LIB}"
    ${_XPURT_EXTRA_ARCHIVES}
  )
endif()

# System libs commonly needed by IREE runtime and the plugin.
target_link_libraries(json_dispatch_runner PRIVATE
  Threads::Threads
  ${CMAKE_DL_LIBS}
  m
)
