// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "compgen/arena.h"

void *cg_arena_alloc(cg_arena_t *a, size_t n, size_t align) {
  if (a == 0 || a->base == 0) return (void *)0;
  size_t mask = align - 1u;
  size_t off = (a->used + mask) & ~mask;
  if (off + n > a->size) return (void *)0;  // OOM
  a->used = off + n;
  return a->base + off;
}
