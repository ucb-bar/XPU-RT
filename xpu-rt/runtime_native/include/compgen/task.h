// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Task primitives for the CompGen execution engine.
//
// Inspired by IREE's task system but intentionally much simpler: tasks form a
// DAG through completion_task pointers.  The engine schedules them in
// dependency order on a single thread.

#ifndef COMPGEN_TASK_H_
#define COMPGEN_TASK_H_

#include "compgen/types.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Task type enum
// ---------------------------------------------------------------------------

typedef enum cg_task_type_e {
  CG_TASK_NOP = 0,       // No-op (used as a join point / sentinel).
  CG_TASK_CALL = 1,      // Invoke a host function pointer.
  CG_TASK_BARRIER = 2,   // Pure synchronisation point.
  CG_TASK_FENCE = 3,     // External wait / signal.
  CG_TASK_DISPATCH = 4,  // Dispatch a workgroup computation.
} cg_task_type_t;

// ---------------------------------------------------------------------------
// Task flags
// ---------------------------------------------------------------------------

typedef uint32_t cg_task_flags_t;
#define CG_TASK_FLAG_NONE 0u

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

typedef struct cg_task_s cg_task_t;

// Cleanup function called after a task completes.
typedef void (*cg_task_cleanup_fn)(cg_task_t* task, cg_status_t status);

// ---------------------------------------------------------------------------
// Base task
// ---------------------------------------------------------------------------

struct cg_task_s {
  cg_task_type_t type;
  cg_task_flags_t flags;

  // Intrusive linked-list pointer (used by the engine's ready queue).
  cg_task_t* next_task;

  // Task to notify when this task completes.  May be NULL.
  cg_task_t* completion_task;

  // Number of incomplete predecessors.  The task is ready when this reaches 0.
  atomic_int pending_dependency_count;

  // Optional cleanup callback invoked after execution.
  cg_task_cleanup_fn cleanup_fn;
};

// ---------------------------------------------------------------------------
// Call task
// ---------------------------------------------------------------------------

// User callback for CG_TASK_CALL.
typedef cg_status_t (*cg_task_call_fn)(void* user_data);

typedef struct cg_task_call_s {
  cg_task_t base;
  cg_task_call_fn fn;
  void* user_data;
} cg_task_call_t;

// ---------------------------------------------------------------------------
// Dispatch task
// ---------------------------------------------------------------------------

#define CG_MAX_PUSH_CONSTANTS 16

typedef struct cg_task_dispatch_s {
  cg_task_t base;

  // Opaque executable pointer (e.g. a compiled kernel handle).
  void* executable;

  // 3-D workgroup grid.
  uint32_t workgroup_count[3];

  // Small set of push-constant values forwarded to the executable.
  uint32_t push_constant_count;
  uint32_t push_constants[CG_MAX_PUSH_CONSTANTS];
} cg_task_dispatch_t;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Initialise a task struct to safe defaults.
void cg_task_initialize(cg_task_t* task, cg_task_type_t type);

// Set the completion (dependent) task.
void cg_task_set_completion_task(cg_task_t* task, cg_task_t* completion);

// Returns true when the task has no remaining dependencies.
bool cg_task_is_ready(const cg_task_t* task);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPGEN_TASK_H_
