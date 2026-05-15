// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "compgen/engine.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal engine state
// ---------------------------------------------------------------------------

struct cg_engine_s {
  cg_allocator_t allocator;

  // Singly-linked ready queue (FIFO).
  cg_task_t* ready_head;
  cg_task_t* ready_tail;
};

// ---------------------------------------------------------------------------
// Default allocator (wraps malloc / free)
// ---------------------------------------------------------------------------

static void* cg_default_alloc(void* self, cg_size_t size) {
  (void)self;
  return malloc(size);
}

static void cg_default_free(void* self, void* ptr) {
  (void)self;
  free(ptr);
}

static const cg_allocator_t cg_default_allocator = {
    .self = NULL,
    .alloc = cg_default_alloc,
    .free = cg_default_free,
};

// ---------------------------------------------------------------------------
// Ready-queue helpers
// ---------------------------------------------------------------------------

static void cg_engine_enqueue(cg_engine_t* engine, cg_task_t* task) {
  task->next_task = NULL;
  if (engine->ready_tail) {
    engine->ready_tail->next_task = task;
    engine->ready_tail = task;
  } else {
    engine->ready_head = task;
    engine->ready_tail = task;
  }
}

static cg_task_t* cg_engine_dequeue(cg_engine_t* engine) {
  cg_task_t* task = engine->ready_head;
  if (!task) return NULL;
  engine->ready_head = task->next_task;
  if (!engine->ready_head) {
    engine->ready_tail = NULL;
  }
  task->next_task = NULL;
  return task;
}

// ---------------------------------------------------------------------------
// Task execution (single-threaded dispatch)
// ---------------------------------------------------------------------------

static cg_status_t cg_engine_execute_task(cg_task_t* task) {
  switch (task->type) {
    case CG_TASK_NOP:
    case CG_TASK_BARRIER:
    case CG_TASK_FENCE:
      return CG_STATUS_OK;

    case CG_TASK_CALL: {
      cg_task_call_t* call = (cg_task_call_t*)task;
      if (call->fn) {
        return call->fn(call->user_data);
      }
      return CG_STATUS_OK;
    }

    case CG_TASK_DISPATCH:
      // Dispatch is a placeholder: real kernel launch would happen here.
      return CG_STATUS_OK;

    default:
      return CG_STATUS_UNIMPLEMENTED;
  }
}

// After a task completes, propagate to its completion task.
static void cg_engine_propagate(cg_engine_t* engine, cg_task_t* task,
                                cg_status_t status) {
  if (task->cleanup_fn) {
    task->cleanup_fn(task, status);
  }

  cg_task_t* completion = task->completion_task;
  if (completion) {
    int prev = atomic_fetch_sub(&completion->pending_dependency_count, 1);
    if (prev == 1) {
      // All dependencies satisfied -- task is now ready.
      cg_engine_enqueue(engine, completion);
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cg_status_t cg_engine_create(const cg_engine_options_t* options,
                             const cg_allocator_t* allocator,
                             cg_engine_t** out_engine) {
  (void)options;
  if (!out_engine) return CG_STATUS_ERROR;

  const cg_allocator_t* alloc = allocator ? allocator : &cg_default_allocator;

  cg_engine_t* engine =
      (cg_engine_t*)cg_allocator_alloc(alloc, sizeof(cg_engine_t));
  if (!engine) return CG_STATUS_OUT_OF_MEMORY;

  memset(engine, 0, sizeof(*engine));
  engine->allocator = *alloc;

  *out_engine = engine;
  return CG_STATUS_OK;
}

void cg_engine_destroy(cg_engine_t* engine) {
  if (!engine) return;
  cg_allocator_t alloc = engine->allocator;
  cg_allocator_free(&alloc, engine);
}

cg_status_t cg_engine_submit(cg_engine_t* engine, cg_task_t* task_list) {
  if (!engine) return CG_STATUS_ERROR;

  cg_task_t* task = task_list;
  while (task) {
    cg_task_t* next = task->next_task;
    if (cg_task_is_ready(task)) {
      cg_engine_enqueue(engine, task);
    }
    task = next;
  }
  return CG_STATUS_OK;
}

cg_status_t cg_engine_execute_step(cg_engine_t* engine) {
  if (!engine) return CG_STATUS_ERROR;

  cg_task_t* task = cg_engine_dequeue(engine);
  if (!task) return CG_STATUS_UNIMPLEMENTED;

  cg_status_t status = cg_engine_execute_task(task);
  cg_engine_propagate(engine, task, status);
  return status;
}

cg_status_t cg_engine_wait_idle(cg_engine_t* engine) {
  if (!engine) return CG_STATUS_ERROR;

  while (engine->ready_head) {
    cg_status_t status = cg_engine_execute_step(engine);
    if (status != CG_STATUS_OK && status != CG_STATUS_UNIMPLEMENTED) {
      return status;
    }
  }
  return CG_STATUS_OK;
}
