/*
 * Shared CPU command-buffer executor.
 *
 * Runs a finalised ``cg_rt_command_buffer_t`` on the host. Both
 * cpu_sync and cpu_task call this; the difference between the two
 * drivers is only *when* and *where* this runs (caller thread vs
 * worker thread). Semaphore waits / signals are orchestrated by the
 * driver's queue_submit, not here.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

static cg_rt_status_t execute_copy(const cg_rt_command_t *cmd) {
    if (cmd->copy.src->data == NULL || cmd->copy.dst->data == NULL) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    memcpy((char *)cmd->copy.dst->data + cmd->copy.dst_offset,
           (const char *)cmd->copy.src->data + cmd->copy.src_offset,
           cmd->copy.size_bytes);
    return CG_RT_OK;
}

static cg_rt_status_t execute_fill(const cg_rt_command_t *cmd) {
    if (cmd->fill.dst->data == NULL) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    uint8_t *dst = (uint8_t *)cmd->fill.dst->data + cmd->fill.dst_offset;
    size_t n = cmd->fill.size_bytes;
    const uint8_t *pat_bytes = (const uint8_t *)&cmd->fill.pattern;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = pat_bytes[i & 3];
    }
    return CG_RT_OK;
}

static cg_rt_status_t execute_dispatch(const cg_rt_command_t *cmd) {
    cg_rt_executable_t *exe = cmd->dispatch.executable;
    if (exe == NULL || exe->entry_point == NULL) {
        return CG_RT_ERR_FAILED_PRECOND;
    }

    size_t n = cmd->dispatch.n_bindings;
    void **binding_ptrs = NULL;
    size_t *binding_sizes = NULL;
    if (n > 0) {
        binding_ptrs = malloc(n * sizeof(*binding_ptrs));
        binding_sizes = malloc(n * sizeof(*binding_sizes));
        if (binding_ptrs == NULL || binding_sizes == NULL) {
            free(binding_ptrs);
            free(binding_sizes);
            return CG_RT_ERR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < n; ++i) {
            binding_ptrs[i] = cmd->dispatch.bindings[i]->data;
            binding_sizes[i] = cmd->dispatch.bindings[i]->size;
        }
    }

    int rc = exe->entry_point(cmd->dispatch.push_constants,
                              cmd->dispatch.pc_size,
                              binding_ptrs,
                              binding_sizes,
                              n);
    free(binding_ptrs);
    free(binding_sizes);
    if (rc != 0) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_execute_cpu_command_buffer(cg_rt_command_buffer_t *cb) {
    if (cb == NULL) return CG_RT_ERR_INVALID_ARGUMENT;
    if (cb->state != CG_RT_CB_STATE_EXECUTABLE) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    for (size_t i = 0; i < cb->num_commands; ++i) {
        cg_rt_status_t rc = CG_RT_OK;
        switch (cb->commands[i].op) {
        case CG_RT_CMD_OP_COPY:     rc = execute_copy(&cb->commands[i]);     break;
        case CG_RT_CMD_OP_FILL:     rc = execute_fill(&cb->commands[i]);     break;
        case CG_RT_CMD_OP_DISPATCH: rc = execute_dispatch(&cb->commands[i]); break;
        case CG_RT_CMD_OP_BARRIER:  /* no-op; cb is replayed in-order */      break;
        default: rc = CG_RT_ERR_UNSUPPORTED; break;
        }
        if (rc != CG_RT_OK) return rc;
    }
    return CG_RT_OK;
}
