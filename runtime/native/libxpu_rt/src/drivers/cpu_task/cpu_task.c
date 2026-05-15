/*
 * cpu_task driver — pthread-backed asynchronous queues.
 *
 * One worker thread per logical queue, FIFO ordering on each queue,
 * but fully concurrent across queues. ``queue_submit`` returns
 * immediately after enqueuing; the worker performs the wait, execute,
 * and signal phases on its own thread. This is the CPU mirror of a
 * real GPU driver's stream — it exercises the full async submission
 * surface without any discrete-device runtime dependency.
 *
 * Ownership: submitters give up their ``wait``/``signal`` *arrays*
 * (we copy them into the entry) but retain ownership of the
 * ``command_buffer`` and ``semaphore`` handles — those must outlive
 * the submission. This matches Vulkan / IREE's queue-submit contract.
 *
 * Shutdown: ``device_close`` sets a ``shutdown`` flag, broadcasts all
 * queue condvars, and joins every worker. Any entries still in a
 * queue at shutdown have their signal semaphores *failed* with
 * CG_RT_ERR_ABORTED so waiters don't deadlock.
 */

#include "../../core/internal.h"

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define CPU_TASK_NUM_QUEUES 4

/* ------------------------------------------------------------------ */
/* Per-queue entry + queue state                                       */
/* ------------------------------------------------------------------ */

typedef struct submit_entry {
    cg_rt_semaphore_point_t *wait;
    size_t                   n_wait;
    cg_rt_semaphore_point_t *signal;
    size_t                   n_signal;
    cg_rt_command_buffer_t  *command_buffer;
    struct submit_entry     *next;
} submit_entry_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    submit_entry_t *head;
    submit_entry_t *tail;
    bool            shutdown;
    pthread_t       worker;
    bool            worker_started;
} cpu_task_queue_t;

typedef struct {
    cpu_task_queue_t queues[CPU_TASK_NUM_QUEUES];
} cpu_task_state_t;

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static submit_entry_t *entry_alloc(const cg_rt_semaphore_point_t *wait,
                                   size_t                        n_wait,
                                   const cg_rt_semaphore_point_t *signal,
                                   size_t                        n_signal,
                                   cg_rt_command_buffer_t       *cb) {
    submit_entry_t *e = calloc(1, sizeof(*e));
    if (e == NULL) return NULL;
    if (n_wait > 0) {
        e->wait = malloc(n_wait * sizeof(*e->wait));
        if (e->wait == NULL) {
            free(e);
            return NULL;
        }
        memcpy(e->wait, wait, n_wait * sizeof(*e->wait));
        e->n_wait = n_wait;
    }
    if (n_signal > 0) {
        e->signal = malloc(n_signal * sizeof(*e->signal));
        if (e->signal == NULL) {
            free(e->wait);
            free(e);
            return NULL;
        }
        memcpy(e->signal, signal, n_signal * sizeof(*e->signal));
        e->n_signal = n_signal;
    }
    e->command_buffer = cb;
    return e;
}

static void entry_free(submit_entry_t *e) {
    if (e == NULL) return;
    free(e->wait);
    free(e->signal);
    free(e);
}

static void fail_entry_signals(submit_entry_t *e, cg_rt_status_t status) {
    for (size_t i = 0; i < e->n_signal; ++i) {
        cg_rt_semaphore_fail(e->signal[i].semaphore, status);
    }
}

/* ------------------------------------------------------------------ */
/* Worker loop                                                          */
/* ------------------------------------------------------------------ */

static void *worker_main(void *arg) {
    cpu_task_queue_t *q = arg;

    for (;;) {
        /* Pull next entry or observe shutdown. */
        pthread_mutex_lock(&q->mutex);
        while (q->head == NULL && !q->shutdown) {
            pthread_cond_wait(&q->cond, &q->mutex);
        }
        if (q->head == NULL && q->shutdown) {
            pthread_mutex_unlock(&q->mutex);
            break;
        }
        submit_entry_t *e = q->head;
        q->head = e->next;
        if (q->head == NULL) q->tail = NULL;
        pthread_mutex_unlock(&q->mutex);

        /* Wait phase — failure fails all signals and drops the entry. */
        cg_rt_status_t rc = CG_RT_OK;
        for (size_t i = 0; i < e->n_wait; ++i) {
            rc = cg_rt_semaphore_wait(e->wait[i].semaphore,
                                      e->wait[i].value,
                                      CG_RT_TIMEOUT_INFINITE);
            if (rc != CG_RT_OK) break;
        }
        if (rc != CG_RT_OK) {
            fail_entry_signals(e, rc);
            entry_free(e);
            continue;
        }

        /* Execute phase. */
        rc = cg_rt_execute_cpu_command_buffer(e->command_buffer);
        if (rc != CG_RT_OK) {
            fail_entry_signals(e, rc);
            entry_free(e);
            continue;
        }

        /* Signal phase. */
        for (size_t i = 0; i < e->n_signal; ++i) {
            cg_rt_semaphore_signal(e->signal[i].semaphore,
                                   e->signal[i].value);
        }
        entry_free(e);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Vtable entrypoints                                                   */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cpu_task_device_open(cg_rt_instance_t *instance,
                                           uint32_t          device_index,
                                           cg_rt_device_t  **out_device);
static void cpu_task_device_close(cg_rt_device_t *device);
static cg_rt_status_t cpu_task_queue_submit(cg_rt_device_t               *device,
                                            uint32_t                      queue_index,
                                            const cg_rt_semaphore_point_t *wait,
                                            size_t                        n_wait,
                                            const cg_rt_semaphore_point_t *signal,
                                            size_t                        n_signal,
                                            cg_rt_command_buffer_t       *command_buffer);

static uint32_t cpu_task_query_device_count(void) {
    return 1;  /* cpu_task is single-logical-device like cpu_sync */
}

const cg_rt_driver_vtable_t cg_rt_cpu_task_vtable = {
    .name               = "cpu_task",
    .device_open        = cpu_task_device_open,
    .device_close       = cpu_task_device_close,
    .query_device_count = cpu_task_query_device_count,
    .queue_submit       = cpu_task_queue_submit,
};

static void fill_cpu_task_traits(cg_rt_device_traits_t *t) {
    memset(t, 0, sizeof(*t));
    t->device_class = CG_RT_DEVICE_CLASS_CPU;
    strncpy(t->vendor, "host", sizeof(t->vendor) - 1);
    strncpy(t->name,   "cpu_task", sizeof(t->name) - 1);
    t->has_native_timeline_semaphores = 1;
    t->has_global_atomics             = 1;
    t->has_shared_memory_atomics      = 1;
    t->supports_persistent_kernels    = 1;
    t->supports_cooperative_launch    = 0;
    t->supports_command_buffers       = 1;
    t->supports_graph_capture         = 0;
    t->supports_event_tensors         = 1;
    t->is_bare_metal                  = 0;
    t->has_rtos_support               = 0;
    t->max_device_memory_bytes        = 0;
    t->supports_host_pinned           = 1;
    t->supports_peer_access           = 0;
    t->max_concurrent_queues          = CPU_TASK_NUM_QUEUES;
    t->max_workgroup_size             = 1;
}

static cg_rt_status_t cpu_task_device_open(cg_rt_instance_t *instance,
                                           uint32_t          device_index,
                                           cg_rt_device_t  **out_device) {
    cg_rt_device_t *dev = calloc(1, sizeof(*dev));
    if (dev == NULL) return CG_RT_ERR_OUT_OF_MEMORY;
    dev->vtable = instance->vtable;
    dev->device_index = device_index;
    fill_cpu_task_traits(&dev->traits);
    dev->num_queues = CPU_TASK_NUM_QUEUES;

    cpu_task_state_t *state = calloc(1, sizeof(*state));
    if (state == NULL) {
        free(dev);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    dev->driver_state = state;

    /* Initialise each queue's mutex + cond and spin up its worker. */
    for (size_t i = 0; i < CPU_TASK_NUM_QUEUES; ++i) {
        if (pthread_mutex_init(&state->queues[i].mutex, NULL) != 0) {
            goto fail;
        }
        if (pthread_cond_init(&state->queues[i].cond, NULL) != 0) {
            pthread_mutex_destroy(&state->queues[i].mutex);
            goto fail;
        }
        if (pthread_create(&state->queues[i].worker, NULL,
                           worker_main, &state->queues[i]) != 0) {
            pthread_cond_destroy(&state->queues[i].cond);
            pthread_mutex_destroy(&state->queues[i].mutex);
            goto fail;
        }
        state->queues[i].worker_started = true;
    }

    *out_device = dev;
    return CG_RT_OK;

fail:
    /* Tear down any started workers. */
    for (size_t i = 0; i < CPU_TASK_NUM_QUEUES; ++i) {
        if (state->queues[i].worker_started) {
            pthread_mutex_lock(&state->queues[i].mutex);
            state->queues[i].shutdown = true;
            pthread_cond_broadcast(&state->queues[i].cond);
            pthread_mutex_unlock(&state->queues[i].mutex);
            pthread_join(state->queues[i].worker, NULL);
            pthread_cond_destroy(&state->queues[i].cond);
            pthread_mutex_destroy(&state->queues[i].mutex);
        }
    }
    free(state);
    free(dev);
    return CG_RT_ERR_UNKNOWN;
}

static void cpu_task_device_close(cg_rt_device_t *device) {
    if (device == NULL) return;
    cpu_task_state_t *state = device->driver_state;
    if (state != NULL) {
        /* Signal shutdown on every queue. */
        for (size_t i = 0; i < CPU_TASK_NUM_QUEUES; ++i) {
            pthread_mutex_lock(&state->queues[i].mutex);
            state->queues[i].shutdown = true;
            pthread_cond_broadcast(&state->queues[i].cond);
            pthread_mutex_unlock(&state->queues[i].mutex);
        }
        /* Join workers. */
        for (size_t i = 0; i < CPU_TASK_NUM_QUEUES; ++i) {
            if (state->queues[i].worker_started) {
                pthread_join(state->queues[i].worker, NULL);
            }
        }
        /* Fail any still-queued entries. This shouldn't happen in a
         * well-behaved caller, but if the device is torn down while
         * entries are in flight we mark their signals as aborted so
         * waiters don't hang. */
        for (size_t i = 0; i < CPU_TASK_NUM_QUEUES; ++i) {
            submit_entry_t *e = state->queues[i].head;
            while (e != NULL) {
                fail_entry_signals(e, CG_RT_ERR_ABORTED);
                submit_entry_t *next = e->next;
                entry_free(e);
                e = next;
            }
            pthread_cond_destroy(&state->queues[i].cond);
            pthread_mutex_destroy(&state->queues[i].mutex);
        }
        free(state);
    }
    free(device);
}

static cg_rt_status_t cpu_task_queue_submit(cg_rt_device_t               *device,
                                            uint32_t                      queue_index,
                                            const cg_rt_semaphore_point_t *wait,
                                            size_t                        n_wait,
                                            const cg_rt_semaphore_point_t *signal,
                                            size_t                        n_signal,
                                            cg_rt_command_buffer_t       *command_buffer) {
    if (command_buffer == NULL || device == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (queue_index >= device->num_queues) return CG_RT_ERR_INVALID_ARGUMENT;
    if ((wait == NULL) != (n_wait == 0)) return CG_RT_ERR_INVALID_ARGUMENT;
    if ((signal == NULL) != (n_signal == 0)) return CG_RT_ERR_INVALID_ARGUMENT;

    cpu_task_state_t *state = device->driver_state;
    cpu_task_queue_t *q = &state->queues[queue_index];

    submit_entry_t *e = entry_alloc(wait, n_wait, signal, n_signal, command_buffer);
    if (e == NULL) return CG_RT_ERR_OUT_OF_MEMORY;

    pthread_mutex_lock(&q->mutex);
    if (q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        fail_entry_signals(e, CG_RT_ERR_ABORTED);
        entry_free(e);
        return CG_RT_ERR_ABORTED;
    }
    if (q->tail == NULL) {
        q->head = q->tail = e;
    } else {
        q->tail->next = e;
        q->tail = e;
    }
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
    return CG_RT_OK;
}
