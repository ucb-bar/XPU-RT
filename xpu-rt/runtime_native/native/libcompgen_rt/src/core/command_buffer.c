/*
 * Command buffer — records a list of ops; cpu_sync replays them
 * synchronously at submit time.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 8

static cg_rt_status_t ensure_capacity(cg_rt_command_buffer_t *cb) {
    if (cb->num_commands < cb->capacity) {
        return CG_RT_OK;
    }
    size_t new_cap = (cb->capacity == 0) ? INITIAL_CAPACITY : cb->capacity * 2;
    cg_rt_command_t *new_commands = realloc(cb->commands, new_cap * sizeof(*new_commands));
    if (new_commands == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    cb->commands = new_commands;
    cb->capacity = new_cap;
    return CG_RT_OK;
}

static cg_rt_status_t require_recording(cg_rt_command_buffer_t *cb) {
    if (cb->state != CG_RT_CB_STATE_RECORDING) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    return CG_RT_OK;
}

static void free_command(cg_rt_command_t *cmd) {
    if (cmd->op == CG_RT_CMD_OP_DISPATCH) {
        free(cmd->dispatch.push_constants);
        free(cmd->dispatch.bindings);
    }
}

cg_rt_status_t cg_rt_command_buffer_create(cg_rt_device_t          *device,
                                           cg_rt_command_buffer_t **out_command_buffer) {
    if (device == NULL || out_command_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_command_buffer_t *cb = calloc(1, sizeof(*cb));
    if (cb == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    cb->device = device;
    cb->state = CG_RT_CB_STATE_NEW;
    *out_command_buffer = cb;
    return CG_RT_OK;
}

void cg_rt_command_buffer_destroy(cg_rt_command_buffer_t *command_buffer) {
    if (command_buffer == NULL) {
        return;
    }
    for (size_t i = 0; i < command_buffer->num_commands; ++i) {
        free_command(&command_buffer->commands[i]);
    }
    free(command_buffer->commands);
    free(command_buffer);
}

cg_rt_status_t cg_rt_command_buffer_begin(cg_rt_command_buffer_t *command_buffer) {
    if (command_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (command_buffer->state != CG_RT_CB_STATE_NEW) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    command_buffer->state = CG_RT_CB_STATE_RECORDING;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_command_buffer_end(cg_rt_command_buffer_t *command_buffer) {
    if (command_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (command_buffer->state != CG_RT_CB_STATE_RECORDING) {
        return CG_RT_ERR_FAILED_PRECOND;
    }
    command_buffer->state = CG_RT_CB_STATE_EXECUTABLE;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_command_buffer_copy(cg_rt_command_buffer_t *command_buffer,
                                         cg_rt_buffer_t         *src,
                                         size_t                  src_offset,
                                         cg_rt_buffer_t         *dst,
                                         size_t                  dst_offset,
                                         size_t                  size_bytes) {
    if (command_buffer == NULL || src == NULL || dst == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_status_t rc = require_recording(command_buffer);
    if (rc != CG_RT_OK) return rc;

    /* Bounds check at record time — catches errors early. */
    if (src_offset > src->size || size_bytes > src->size - src_offset ||
        dst_offset > dst->size || size_bytes > dst->size - dst_offset) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    rc = ensure_capacity(command_buffer);
    if (rc != CG_RT_OK) return rc;

    cg_rt_command_t *cmd = &command_buffer->commands[command_buffer->num_commands++];
    cmd->op = CG_RT_CMD_OP_COPY;
    cmd->copy.src = src;
    cmd->copy.src_offset = src_offset;
    cmd->copy.dst = dst;
    cmd->copy.dst_offset = dst_offset;
    cmd->copy.size_bytes = size_bytes;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_command_buffer_fill(cg_rt_command_buffer_t *command_buffer,
                                         cg_rt_buffer_t         *dst,
                                         size_t                  dst_offset,
                                         size_t                  size_bytes,
                                         uint32_t                pattern) {
    if (command_buffer == NULL || dst == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_status_t rc = require_recording(command_buffer);
    if (rc != CG_RT_OK) return rc;
    if (dst_offset > dst->size || size_bytes > dst->size - dst_offset) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    rc = ensure_capacity(command_buffer);
    if (rc != CG_RT_OK) return rc;

    cg_rt_command_t *cmd = &command_buffer->commands[command_buffer->num_commands++];
    cmd->op = CG_RT_CMD_OP_FILL;
    cmd->fill.dst = dst;
    cmd->fill.dst_offset = dst_offset;
    cmd->fill.size_bytes = size_bytes;
    cmd->fill.pattern = pattern;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_command_buffer_dispatch(cg_rt_command_buffer_t *command_buffer,
                                             cg_rt_executable_t     *executable,
                                             const void             *push_constants,
                                             size_t                  pc_size,
                                             cg_rt_buffer_t        **bindings,
                                             size_t                  n_bindings) {
    if (command_buffer == NULL || executable == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if ((push_constants == NULL) != (pc_size == 0)) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if ((bindings == NULL) != (n_bindings == 0)) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_status_t rc = require_recording(command_buffer);
    if (rc != CG_RT_OK) return rc;
    rc = ensure_capacity(command_buffer);
    if (rc != CG_RT_OK) return rc;

    /* Copy push constants so the caller can free them after record.
     * Dispatch stores buffer pointers (not copies — buffers outlive
     * the command record per documented contract). */
    void *pc_copy = NULL;
    if (pc_size > 0) {
        pc_copy = malloc(pc_size);
        if (pc_copy == NULL) return CG_RT_ERR_OUT_OF_MEMORY;
        memcpy(pc_copy, push_constants, pc_size);
    }

    cg_rt_buffer_t **bindings_copy = NULL;
    if (n_bindings > 0) {
        bindings_copy = malloc(n_bindings * sizeof(*bindings_copy));
        if (bindings_copy == NULL) {
            free(pc_copy);
            return CG_RT_ERR_OUT_OF_MEMORY;
        }
        memcpy(bindings_copy, bindings, n_bindings * sizeof(*bindings_copy));
    }

    cg_rt_command_t *cmd = &command_buffer->commands[command_buffer->num_commands++];
    cmd->op = CG_RT_CMD_OP_DISPATCH;
    cmd->dispatch.executable = executable;
    cmd->dispatch.push_constants = pc_copy;
    cmd->dispatch.pc_size = pc_size;
    cmd->dispatch.bindings = bindings_copy;
    cmd->dispatch.n_bindings = n_bindings;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_command_buffer_barrier(cg_rt_command_buffer_t *command_buffer) {
    if (command_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_status_t rc = require_recording(command_buffer);
    if (rc != CG_RT_OK) return rc;
    rc = ensure_capacity(command_buffer);
    if (rc != CG_RT_OK) return rc;
    cg_rt_command_t *cmd = &command_buffer->commands[command_buffer->num_commands++];
    cmd->op = CG_RT_CMD_OP_BARRIER;
    return CG_RT_OK;
}
