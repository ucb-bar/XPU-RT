/*
 * Event tensor — N-dimensional array of atomic counters.
 *
 * Implements the paper primitive (arXiv:2604.13327, MLSys '26, §3.2):
 *   notify(idx, d): atomic_fetch_sub(E[idx], d)
 *   wait(idx):      spin / block until E[idx] <= 0
 *
 * Notify is a single atomic op on the hot path; host-side waits use
 * the platform condvar via ``platform.h`` so the same code compiles
 * on POSIX (pthreads) and bare-metal (spin-polled deadline). The
 * broadcast is issued only at the zero-crossing to keep contention
 * minimal.
 */

#include "internal.h"

#include <stdlib.h>
#include <string.h>

static size_t compute_num_cells(const int64_t *shape, size_t rank) {
    size_t n = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (shape[i] <= 0) {
            return 0;
        }
        n *= (size_t)shape[i];
    }
    return n;
}

cg_rt_status_t cg_rt_event_tensor_create(cg_rt_device_t        *device,
                                         size_t                 rank,
                                         const int64_t         *shape,
                                         cg_rt_event_dtype_t    dtype,
                                         int64_t                initial_value,
                                         cg_rt_event_tensor_t **out_event_tensor) {
    if (out_event_tensor == NULL || (rank > 0 && shape == NULL)) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (dtype != CG_RT_EVENT_DTYPE_I32 && dtype != CG_RT_EVENT_DTYPE_I64) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    size_t num_cells = (rank == 0) ? 1 : compute_num_cells(shape, rank);
    if (num_cells == 0) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    cg_rt_event_tensor_t *et = calloc(1, sizeof(*et));
    if (et == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    et->rank = rank;
    et->num_cells = num_cells;
    et->dtype = dtype;
    if (rank > 0) {
        et->shape = malloc(rank * sizeof(int64_t));
        if (et->shape == NULL) {
            free(et);
            return CG_RT_ERR_OUT_OF_MEMORY;
        }
        memcpy(et->shape, shape, rank * sizeof(int64_t));
    }
    et->cells = calloc(num_cells, sizeof(_Atomic int64_t));
    if (et->cells == NULL) {
        free(et->shape);
        free(et);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < num_cells; ++i) {
        atomic_store(&et->cells[i], initial_value);
    }
    if (cg_rt_platform_mutex_init(&et->mutex) != 0) {
        free(et->cells);
        free(et->shape);
        free(et);
        return CG_RT_ERR_UNKNOWN;
    }
    if (cg_rt_platform_cond_init(&et->cond) != 0) {
        cg_rt_platform_mutex_destroy(&et->mutex);
        free(et->cells);
        free(et->shape);
        free(et);
        return CG_RT_ERR_UNKNOWN;
    }
    (void)device;
    *out_event_tensor = et;
    return CG_RT_OK;
}

void cg_rt_event_tensor_destroy(cg_rt_event_tensor_t *event_tensor) {
    if (event_tensor == NULL) {
        return;
    }
    cg_rt_platform_cond_destroy(&event_tensor->cond);
    cg_rt_platform_mutex_destroy(&event_tensor->mutex);
    free(event_tensor->cells);
    free(event_tensor->shape);
    free(event_tensor);
}

size_t cg_rt_event_tensor_num_cells(const cg_rt_event_tensor_t *event_tensor) {
    return (event_tensor == NULL) ? 0 : event_tensor->num_cells;
}

cg_rt_status_t cg_rt_event_tensor_notify(cg_rt_event_tensor_t *event_tensor,
                                         size_t                linear_idx,
                                         int64_t               decrement) {
    if (event_tensor == NULL || linear_idx >= event_tensor->num_cells) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    int64_t prev = atomic_fetch_sub(&event_tensor->cells[linear_idx], decrement);
    int64_t now  = prev - decrement;
    if (prev > 0 && now <= 0) {
        cg_rt_platform_mutex_lock(&event_tensor->mutex);
        cg_rt_platform_cond_broadcast(&event_tensor->cond);
        cg_rt_platform_mutex_unlock(&event_tensor->mutex);
    }
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_event_tensor_wait(cg_rt_event_tensor_t *event_tensor,
                                       size_t                linear_idx,
                                       uint64_t              timeout_ns) {
    if (event_tensor == NULL || linear_idx >= event_tensor->num_cells) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    /* Fast path: already ready. */
    if (atomic_load(&event_tensor->cells[linear_idx]) <= 0) {
        return CG_RT_OK;
    }
    if (timeout_ns == CG_RT_TIMEOUT_POLL) {
        return CG_RT_ERR_TIMED_OUT;
    }

    cg_rt_platform_mutex_lock(&event_tensor->mutex);

    if (timeout_ns == CG_RT_TIMEOUT_INFINITE) {
        while (atomic_load(&event_tensor->cells[linear_idx]) > 0) {
            cg_rt_platform_cond_wait(&event_tensor->cond, &event_tensor->mutex);
        }
        cg_rt_platform_mutex_unlock(&event_tensor->mutex);
        return CG_RT_OK;
    }

    cg_rt_status_t rc = CG_RT_OK;
    while (atomic_load(&event_tensor->cells[linear_idx]) > 0) {
        cg_rt_status_t prc =
            cg_rt_platform_cond_timedwait_ns(&event_tensor->cond,
                                             &event_tensor->mutex,
                                             timeout_ns);
        if (prc == CG_RT_ERR_TIMED_OUT) {
            if (atomic_load(&event_tensor->cells[linear_idx]) <= 0) {
                rc = CG_RT_OK;
            } else {
                rc = CG_RT_ERR_TIMED_OUT;
            }
            break;
        } else if (prc != CG_RT_OK) {
            rc = prc;
            break;
        }
    }
    cg_rt_platform_mutex_unlock(&event_tensor->mutex);
    return rc;
}

cg_rt_status_t cg_rt_event_tensor_query(cg_rt_event_tensor_t *event_tensor,
                                        size_t                linear_idx,
                                        int64_t              *out_value) {
    if (event_tensor == NULL || out_value == NULL ||
        linear_idx >= event_tensor->num_cells) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    *out_value = atomic_load(&event_tensor->cells[linear_idx]);
    return CG_RT_OK;
}

cg_rt_status_t cg_rt_event_tensor_reset(cg_rt_event_tensor_t *event_tensor,
                                        int64_t               value) {
    if (event_tensor == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_platform_mutex_lock(&event_tensor->mutex);
    for (size_t i = 0; i < event_tensor->num_cells; ++i) {
        atomic_store(&event_tensor->cells[i], value);
    }
    cg_rt_platform_cond_broadcast(&event_tensor->cond);
    cg_rt_platform_mutex_unlock(&event_tensor->mutex);
    return CG_RT_OK;
}
