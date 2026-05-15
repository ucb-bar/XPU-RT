// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Single-threaded task execution engine for the CompGen runtime.
//
// The engine maintains a ready queue of tasks whose dependencies have been
// satisfied.  Calling cg_engine_execute_step() pops one task, executes it,
// and propagates completions.  cg_engine_wait_idle() drains the queue.

#ifndef COMPGEN_ENGINE_H_
#define COMPGEN_ENGINE_H_

#include "compgen/task.h"
#include "compgen/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Engine options
// ---------------------------------------------------------------------------

typedef struct cg_engine_options_s {
  // Reserved for future use (e.g. thread count, queue depth).
  int reserved;
} cg_engine_options_t;

// ---------------------------------------------------------------------------
// Opaque engine handle
// ---------------------------------------------------------------------------

typedef struct cg_engine_s cg_engine_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create an engine.  On success *out_engine is set and CG_STATUS_OK returned.
cg_status_t cg_engine_create(const cg_engine_options_t* options,
                             const cg_allocator_t* allocator,
                             cg_engine_t** out_engine);

// Destroy an engine and release all resources.
void cg_engine_destroy(cg_engine_t* engine);

// ---------------------------------------------------------------------------
// Submission & execution
// ---------------------------------------------------------------------------

// Submit a linked list of root tasks (connected via next_task).  Tasks whose
// pending_dependency_count is already 0 are placed directly on the ready queue.
cg_status_t cg_engine_submit(cg_engine_t* engine, cg_task_t* task_list);

// Execute a single ready task.  Returns CG_STATUS_OK if a task was executed,
// or CG_STATUS_UNIMPLEMENTED if the ready queue was empty.
cg_status_t cg_engine_execute_step(cg_engine_t* engine);

// Block until the ready queue is empty (single-threaded: loops execute_step).
cg_status_t cg_engine_wait_idle(cg_engine_t* engine);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // COMPGEN_ENGINE_H_
