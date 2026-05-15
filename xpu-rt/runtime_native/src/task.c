// Copyright 2026 CompGen Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "compgen/task.h"

#include <string.h>

void cg_task_initialize(cg_task_t* task, cg_task_type_t type) {
  memset(task, 0, sizeof(*task));
  task->type = type;
  atomic_store(&task->pending_dependency_count, 0);
}

void cg_task_set_completion_task(cg_task_t* task, cg_task_t* completion) {
  task->completion_task = completion;
  // The completion task gains one more pending dependency.
  atomic_fetch_add(&completion->pending_dependency_count, 1);
}

bool cg_task_is_ready(const cg_task_t* task) {
  return atomic_load(&task->pending_dependency_count) == 0;
}
