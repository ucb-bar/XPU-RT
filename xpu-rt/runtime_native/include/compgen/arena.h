// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Bump arena for CompGen's ahead-of-time inference runtime.
//
// Generated forward functions request activation buffers from an arena
// supplied by the caller. The arena is a bump allocator with alignment
// support and a reset operation — no free, no hidden global state,
// no dependency on libc malloc. This keeps the runtime link-compatible
// with freestanding, bare-metal, and RTOS environments without
// modification.
//
// Complements the task-graph execution engine in compgen/engine.h:
// - The engine orchestrates task dependencies on the host control path.
// - The arena services the per-invocation activation memory needs of
//   the generated straight-line kernels the tasks dispatch to.

#ifndef COMPGEN_ARENA_H_
#define COMPGEN_ARENA_H_

#include "compgen/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cg_arena_s {
  uint8_t *base;
  size_t   size;
  size_t   used;
} cg_arena_t;

// Initialise ``a`` over a caller-owned byte buffer.
// ``base`` must be 16-byte aligned; the arena does not realign.
static inline void cg_arena_init(cg_arena_t *a, void *base, size_t size) {
  a->base = (uint8_t *)base;
  a->size = size;
  a->used = 0;
}

// Reset to empty without touching contents — constant time.
static inline void cg_arena_reset(cg_arena_t *a) { a->used = 0; }

// Allocate ``n`` bytes aligned to ``align``. Returns NULL on OOM.
void *cg_arena_alloc(cg_arena_t *a, size_t n, size_t align);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPGEN_ARENA_H_
