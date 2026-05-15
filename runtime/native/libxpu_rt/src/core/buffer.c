/*
 * Buffer implementation — host-backed for cpu_sync / cpu_task.
 *
 * Device-local and unified spaces are treated as host-backed here
 * because cpu_sync lives entirely in host memory. A discrete-GPU
 * driver replaces these with its own allocator but preserves the
 * public handle shape.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

cg_rt_status_t cg_rt_buffer_alloc(cg_rt_device_t       *device,
                                  size_t                size_bytes,
                                  cg_rt_memory_space_t  memory_space,
                                  uint32_t              usage_flags,
                                  cg_rt_buffer_t      **out_buffer) {
    if (device == NULL || out_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (size_bytes == 0) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_buffer_t *buffer = calloc(1, sizeof(*buffer));
    if (buffer == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }

    /* Allocation path: defer to the driver's optional ``data_alloc``
     * hook (CUDA uses cuMemAllocManaged); fall back to calloc for
     * the host drivers. The zero-init on the fallback gives tests a
     * deterministic starting state. */
    cg_rt_status_t rc = CG_RT_OK;
    if (device->vtable != NULL && device->vtable->data_alloc != NULL) {
        rc = device->vtable->data_alloc(device, size_bytes, memory_space, &buffer->data);
    } else {
        buffer->data = calloc(1, size_bytes);
        if (buffer->data == NULL) rc = CG_RT_ERR_OUT_OF_MEMORY;
    }
    if (rc != CG_RT_OK) {
        free(buffer);
        return rc;
    }
    buffer->size = size_bytes;
    buffer->memory_space = memory_space;
    buffer->usage_flags = usage_flags;
    buffer->mapped = false;
    buffer->device = device;
    *out_buffer = buffer;
    return CG_RT_OK;
}

void cg_rt_buffer_destroy(cg_rt_buffer_t *buffer) {
    if (buffer == NULL) {
        return;
    }
    if (buffer->data != NULL) {
        if (buffer->device != NULL && buffer->device->vtable != NULL &&
            buffer->device->vtable->data_free != NULL) {
            buffer->device->vtable->data_free(buffer->device, buffer->data);
        } else {
            free(buffer->data);
        }
    }
    free(buffer);
}

size_t cg_rt_buffer_size(const cg_rt_buffer_t *buffer) {
    return (buffer == NULL) ? 0 : buffer->size;
}

cg_rt_status_t cg_rt_buffer_map(cg_rt_buffer_t  *buffer,
                                size_t           offset,
                                size_t           size,
                                void           **out_ptr) {
    if (buffer == NULL || out_ptr == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (offset > buffer->size || size > buffer->size - offset) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (buffer->mapped) {
        /* cpu_sync permits re-mapping because it's direct host memory;
         * a discrete driver would reject this. Keep the idempotent
         * behaviour explicit so code is portable. */
    }
    buffer->mapped = true;
    *out_ptr = (char *)buffer->data + offset;
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_buffer_unmap(cg_rt_buffer_t *buffer) {
    if (buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    buffer->mapped = false;
    return CG_RT_OK;
}
