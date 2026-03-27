// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Core type definitions for the CompGen C runtime.

#ifndef COMPGEN_TYPES_H_
#define COMPGEN_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

typedef enum cg_status_e {
  CG_STATUS_OK = 0,
  CG_STATUS_ERROR = 1,
  CG_STATUS_UNIMPLEMENTED = 2,
  CG_STATUS_OUT_OF_MEMORY = 3,
  CG_STATUS_INVALID_ARGUMENT = 4,
  CG_STATUS_FAILED_PRECONDITION = 5,
  CG_STATUS_UNAVAILABLE = 6,
  CG_STATUS_OUT_OF_RANGE = 7,
  CG_STATUS_RESOURCE_EXHAUSTED = 8,
} cg_status_t;

// Return a static string describing |status|.
static inline const char* cg_status_string(cg_status_t status) {
  switch (status) {
    case CG_STATUS_OK:
      return "OK";
    case CG_STATUS_ERROR:
      return "ERROR";
    case CG_STATUS_UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case CG_STATUS_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case CG_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case CG_STATUS_FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case CG_STATUS_UNAVAILABLE:
      return "UNAVAILABLE";
    case CG_STATUS_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case CG_STATUS_RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    default:
      return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// Basic typedefs
// ---------------------------------------------------------------------------

typedef size_t cg_size_t;
typedef int64_t cg_index_t;

// ---------------------------------------------------------------------------
// Host allocator
// ---------------------------------------------------------------------------

// Function pointer types for host memory allocation.
typedef void* (*cg_alloc_fn)(void* self, cg_size_t size);
typedef void (*cg_free_fn)(void* self, void* ptr);

// A simple host-side allocator.  |self| is forwarded to every call so that
// the allocator can carry its own state (arena pointer, stats, etc.).
typedef struct cg_allocator_s {
  void* self;
  cg_alloc_fn alloc;
  cg_free_fn free;
} cg_allocator_t;

// Convenience: allocate through an allocator.
static inline void* cg_allocator_alloc(const cg_allocator_t* a,
                                       cg_size_t size) {
  return a->alloc(a->self, size);
}

// Convenience: free through an allocator.
static inline void cg_allocator_free(const cg_allocator_t* a, void* ptr) {
  a->free(a->self, ptr);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPGEN_TYPES_H_
