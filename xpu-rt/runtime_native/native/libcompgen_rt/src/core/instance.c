/*
 * Instance factory + driver registry + shared helpers.
 *
 * The registry is static (no dynamic plugin loading yet). Each
 * driver's vtable symbol is referenced here directly so new drivers
 * are added by listing their vtable in ``kDrivers``.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

static const cg_rt_driver_vtable_t *kDrivers[] = {
    &cg_rt_cpu_sync_vtable,
#ifndef CG_RT_PLATFORM_BARE
    &cg_rt_cpu_task_vtable,
#endif
#ifdef CG_RT_WITH_CUDA
    &cg_rt_cuda_vtable,
#endif
};

static const size_t kNumDrivers = sizeof(kDrivers) / sizeof(kDrivers[0]);

const char *cg_rt_status_string(cg_rt_status_t status) {
    switch (status) {
    case CG_RT_OK:                   return "ok";
    case CG_RT_ERR_INVALID_ARGUMENT: return "invalid-argument";
    case CG_RT_ERR_OUT_OF_MEMORY:    return "out-of-memory";
    case CG_RT_ERR_UNSUPPORTED:      return "unsupported";
    case CG_RT_ERR_NOT_FOUND:        return "not-found";
    case CG_RT_ERR_TIMED_OUT:        return "timed-out";
    case CG_RT_ERR_FAILED_PRECOND:   return "failed-precondition";
    case CG_RT_ERR_ABORTED:          return "aborted";
    default:                         return "unknown";
    }
}

cg_rt_status_t cg_rt_instance_create(const char        *driver_name,
                                     cg_rt_instance_t **out_instance) {
    if (out_instance == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    const char *effective_name = (driver_name == NULL) ? "cpu_sync" : driver_name;

    const cg_rt_driver_vtable_t *selected = NULL;
    for (size_t i = 0; i < kNumDrivers; ++i) {
        if (strcmp(kDrivers[i]->name, effective_name) == 0) {
            selected = kDrivers[i];
            break;
        }
    }
    if (selected == NULL) {
        return CG_RT_ERR_NOT_FOUND;
    }

    cg_rt_instance_t *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    instance->vtable = selected;
    /* Populate device_count from the driver. Single-device drivers
     * (cpu_sync) return 1; multi-device drivers (cuda) query the
     * driver runtime — e.g. ``cuDeviceGetCount`` — and return the
     * actual count. NULL fallback keeps the legacy hardcoded-1
     * behaviour for any driver that hasn't filled it in. */
    if (selected->query_device_count != NULL) {
        instance->device_count = selected->query_device_count();
    } else {
        instance->device_count = 1;
    }
    *out_instance = instance;
    return CG_RT_OK;
}

void cg_rt_instance_destroy(cg_rt_instance_t *instance) {
    free(instance);
}

cg_rt_status_t cg_rt_instance_query_devices(cg_rt_instance_t *instance,
                                            cg_rt_device_t  **out_devices,
                                            size_t           *inout_count) {
    if (instance == NULL || inout_count == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    /* Count-only query: caller passes NULL ``out_devices`` and a zero
     * initial count to learn how many slots to allocate. */
    if (out_devices == NULL) {
        *inout_count = instance->device_count;
        return CG_RT_OK;
    }
    /* This runtime's device handles are created on demand via
     * ``cg_rt_device_open`` — there's no pre-existing device array to
     * return pointers into. Callers that want handles must open them
     * individually. This matches IREE's device_open pattern. */
    return CG_RT_ERR_UNSUPPORTED;
}

cg_rt_status_t cg_rt_device_open(cg_rt_instance_t *instance,
                                 uint32_t          device_index,
                                 cg_rt_device_t  **out_device) {
    if (instance == NULL || instance->vtable == NULL ||
        instance->vtable->device_open == NULL || out_device == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (device_index >= instance->device_count) {
        return CG_RT_ERR_NOT_FOUND;
    }
    return instance->vtable->device_open(instance, device_index, out_device);
}

void cg_rt_device_close(cg_rt_device_t *device) {
    if (device == NULL || device->vtable == NULL ||
        device->vtable->device_close == NULL) {
        return;
    }
    device->vtable->device_close(device);
}

cg_rt_status_t cg_rt_device_query_traits(cg_rt_device_t        *device,
                                         cg_rt_device_traits_t *out_traits) {
    if (device == NULL || out_traits == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    *out_traits = device->traits;
    return CG_RT_OK;
}
