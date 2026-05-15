/*
 * cpu_sync driver — blocking, single-threaded reference driver.
 *
 * Every queue submit runs synchronously on the caller thread: all
 * waits block the calling thread until they complete, then the
 * command buffer replays inline, then the signal semaphores fire.
 * This is the simplest possible driver that still exercises every
 * public primitive in libcompgen_rt, and it is the reference
 * implementation bare-metal + Zephyr builds will derive from.
 *
 * Concurrency: different queues on the same device can be submitted
 * concurrently from different threads — each queue's submit holds a
 * per-queue mutex, so order-within-queue is preserved but parallelism
 * across queues is permitted (so that Python-level threaded callers
 * can exercise traits.max_concurrent_queues).
 */

#include "../../core/internal.h"

#include <stdlib.h>
#include <string.h>

#define CPU_SYNC_NUM_QUEUES 4

static cg_rt_status_t cpu_sync_device_open(cg_rt_instance_t *instance,
                                           uint32_t          device_index,
                                           cg_rt_device_t  **out_device);
static void cpu_sync_device_close(cg_rt_device_t *device);
static cg_rt_status_t cpu_sync_queue_submit(cg_rt_device_t               *device,
                                            uint32_t                      queue_index,
                                            const cg_rt_semaphore_point_t *wait,
                                            size_t                        n_wait,
                                            const cg_rt_semaphore_point_t *signal,
                                            size_t                        n_signal,
                                            cg_rt_command_buffer_t       *command_buffer);

static uint32_t cpu_sync_query_device_count(void) {
    return 1;  /* cpu_sync exposes a single logical device */
}

const cg_rt_driver_vtable_t cg_rt_cpu_sync_vtable = {
    .name               = "cpu_sync",
    .device_open        = cpu_sync_device_open,
    .device_close       = cpu_sync_device_close,
    .query_device_count = cpu_sync_query_device_count,
    .queue_submit       = cpu_sync_queue_submit,
};

static void fill_cpu_traits(cg_rt_device_traits_t *t) {
    memset(t, 0, sizeof(*t));
    t->device_class = CG_RT_DEVICE_CLASS_CPU;
    strncpy(t->vendor, "host", sizeof(t->vendor) - 1);
    strncpy(t->name,   "cpu_sync", sizeof(t->name) - 1);
    t->has_native_timeline_semaphores = 1; /* host-emulated but correct */
    t->has_global_atomics             = 1;
    t->has_shared_memory_atomics      = 1;
    t->supports_persistent_kernels    = 1;
    t->supports_cooperative_launch    = 0;
    t->supports_command_buffers       = 1;
    t->supports_graph_capture         = 0;
    t->supports_event_tensors         = 1;
    t->is_bare_metal                  = 0;
    t->has_rtos_support               = 0;
    t->max_device_memory_bytes        = 0; /* unlimited — host RAM */
    t->supports_host_pinned           = 1;
    t->supports_peer_access           = 0;
    t->max_concurrent_queues          = CPU_SYNC_NUM_QUEUES;
    t->max_workgroup_size             = 1;
}

static cg_rt_status_t cpu_sync_device_open(cg_rt_instance_t *instance,
                                           uint32_t          device_index,
                                           cg_rt_device_t  **out_device) {
    cg_rt_device_t *dev = calloc(1, sizeof(*dev));
    if (dev == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    dev->vtable = instance->vtable;
    dev->device_index = device_index;
    fill_cpu_traits(&dev->traits);

    dev->num_queues = CPU_SYNC_NUM_QUEUES;
    dev->queue_mutexes = calloc(dev->num_queues, sizeof(cg_rt_platform_mutex_t));
    if (dev->queue_mutexes == NULL) {
        free(dev);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < dev->num_queues; ++i) {
        if (cg_rt_platform_mutex_init(&dev->queue_mutexes[i]) != 0) {
            for (size_t j = 0; j < i; ++j) {
                cg_rt_platform_mutex_destroy(&dev->queue_mutexes[j]);
            }
            free(dev->queue_mutexes);
            free(dev);
            return CG_RT_ERR_UNKNOWN;
        }
    }

    *out_device = dev;
    return CG_RT_OK;
}

static void cpu_sync_device_close(cg_rt_device_t *device) {
    if (device == NULL) return;
    if (device->queue_mutexes != NULL) {
        for (size_t i = 0; i < device->num_queues; ++i) {
            cg_rt_platform_mutex_destroy(&device->queue_mutexes[i]);
        }
        free(device->queue_mutexes);
    }
    free(device);
}

/* ------------------------------------------------------------------ */
/* Queue submit — synchronous. Uses the shared CPU executor.           */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cpu_sync_queue_submit(cg_rt_device_t               *device,
                                            uint32_t                      queue_index,
                                            const cg_rt_semaphore_point_t *wait,
                                            size_t                        n_wait,
                                            const cg_rt_semaphore_point_t *signal,
                                            size_t                        n_signal,
                                            cg_rt_command_buffer_t       *command_buffer) {
    if (command_buffer == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (queue_index >= device->num_queues) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if ((wait == NULL) != (n_wait == 0)) return CG_RT_ERR_INVALID_ARGUMENT;
    if ((signal == NULL) != (n_signal == 0)) return CG_RT_ERR_INVALID_ARGUMENT;

    cg_rt_platform_mutex_lock(&device->queue_mutexes[queue_index]);

    /* Wait phase — block until each wait reaches its target value.
     * A failed wait fails all signal semaphores and returns early. */
    for (size_t i = 0; i < n_wait; ++i) {
        cg_rt_status_t rc = cg_rt_semaphore_wait(wait[i].semaphore,
                                                 wait[i].value,
                                                 CG_RT_TIMEOUT_INFINITE);
        if (rc != CG_RT_OK) {
            for (size_t j = 0; j < n_signal; ++j) {
                cg_rt_semaphore_fail(signal[j].semaphore, rc);
            }
            cg_rt_platform_mutex_unlock(&device->queue_mutexes[queue_index]);
            return rc;
        }
    }

    /* Execute phase. */
    cg_rt_status_t exec_rc = cg_rt_execute_cpu_command_buffer(command_buffer);
    if (exec_rc != CG_RT_OK) {
        for (size_t j = 0; j < n_signal; ++j) {
            cg_rt_semaphore_fail(signal[j].semaphore, exec_rc);
        }
        cg_rt_platform_mutex_unlock(&device->queue_mutexes[queue_index]);
        return exec_rc;
    }

    /* Signal phase — the wait/signal contract guarantees signals fire
     * only after the command buffer completes successfully. */
    for (size_t i = 0; i < n_signal; ++i) {
        cg_rt_status_t rc = cg_rt_semaphore_signal(signal[i].semaphore,
                                                   signal[i].value);
        if (rc != CG_RT_OK) {
            cg_rt_platform_mutex_unlock(&device->queue_mutexes[queue_index]);
            return rc;
        }
    }

    cg_rt_platform_mutex_unlock(&device->queue_mutexes[queue_index]);
    return CG_RT_OK;
}
