/*
 * Timeline semaphore — atomic uint64 payload + condvar for waiters.
 *
 * Mirrors IREE's host-emulated timeline semaphore. The payload
 * advances monotonically; a separate ``failure_code`` channel
 * propagates asynchronous failures to waiters. Locks and condvars go
 * through the platform shim (``platform.h``) so the same source
 * compiles against POSIX (pthreads) and bare-metal (no OS).
 */

#include "internal.h"

#include <stdlib.h>

cg_rt_status_t cg_rt_semaphore_create(cg_rt_device_t     *device,
                                      uint64_t            initial_value,
                                      cg_rt_semaphore_t **out_semaphore) {
    if (out_semaphore == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_semaphore_t *sem = calloc(1, sizeof(*sem));
    if (sem == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    if (cg_rt_platform_mutex_init(&sem->mutex) != 0) {
        free(sem);
        return CG_RT_ERR_UNKNOWN;
    }
    if (cg_rt_platform_cond_init(&sem->cond) != 0) {
        cg_rt_platform_mutex_destroy(&sem->mutex);
        free(sem);
        return CG_RT_ERR_UNKNOWN;
    }
    atomic_store(&sem->value, initial_value);
    atomic_store(&sem->failure_code, 0);
    (void)device;
    *out_semaphore = sem;
    return CG_RT_OK;
}

void cg_rt_semaphore_destroy(cg_rt_semaphore_t *semaphore) {
    if (semaphore == NULL) {
        return;
    }
    cg_rt_platform_cond_destroy(&semaphore->cond);
    cg_rt_platform_mutex_destroy(&semaphore->mutex);
    free(semaphore);
}

cg_rt_status_t cg_rt_semaphore_signal(cg_rt_semaphore_t *semaphore,
                                      uint64_t           value) {
    if (semaphore == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (atomic_load(&semaphore->failure_code) != 0) {
        return CG_RT_ERR_ABORTED;
    }

    cg_rt_platform_mutex_lock(&semaphore->mutex);
    uint64_t current = atomic_load(&semaphore->value);
    /* Monotonic: refuse to go backwards. Equal is idempotent no-op. */
    if (value > current) {
        atomic_store(&semaphore->value, value);
        cg_rt_platform_cond_broadcast(&semaphore->cond);
    }
    cg_rt_platform_mutex_unlock(&semaphore->mutex);
    return CG_RT_OK;
}

/* Helper: evaluate the wait predicate under the semaphore's lock.
 * Returns OK if the target has been reached, ABORTED if the
 * semaphore has failed, or TIMED_OUT as a "not ready yet" sentinel
 * (the real final return uses this to decide whether to park). */
static cg_rt_status_t wait_predicate(cg_rt_semaphore_t *sem, uint64_t target) {
    int32_t fc = atomic_load(&sem->failure_code);
    if (fc != 0) return CG_RT_ERR_ABORTED;
    uint64_t cur = atomic_load(&sem->value);
    if (cur >= target) return CG_RT_OK;
    return CG_RT_ERR_TIMED_OUT;
}

cg_rt_status_t cg_rt_semaphore_wait(cg_rt_semaphore_t *semaphore,
                                    uint64_t           value,
                                    uint64_t           timeout_ns) {
    if (semaphore == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }

    /* Fast path: already signalled / already failed. */
    int32_t fc = atomic_load(&semaphore->failure_code);
    if (fc != 0) {
        return CG_RT_ERR_ABORTED;
    }
    if (atomic_load(&semaphore->value) >= value) {
        return CG_RT_OK;
    }
    if (timeout_ns == CG_RT_TIMEOUT_POLL) {
        return CG_RT_ERR_TIMED_OUT;
    }

    cg_rt_platform_mutex_lock(&semaphore->mutex);

    cg_rt_status_t rc = wait_predicate(semaphore, value);
    if (rc == CG_RT_OK || rc == CG_RT_ERR_ABORTED) {
        cg_rt_platform_mutex_unlock(&semaphore->mutex);
        return rc;
    }

    if (timeout_ns == CG_RT_TIMEOUT_INFINITE) {
        while ((rc = wait_predicate(semaphore, value)) == CG_RT_ERR_TIMED_OUT) {
            cg_rt_platform_cond_wait(&semaphore->cond, &semaphore->mutex);
        }
    } else {
        /* Re-check after each platform timedwait so a signal that raced
         * the deadline still wins. */
        while ((rc = wait_predicate(semaphore, value)) == CG_RT_ERR_TIMED_OUT) {
            cg_rt_status_t prc =
                cg_rt_platform_cond_timedwait_ns(&semaphore->cond,
                                                 &semaphore->mutex,
                                                 timeout_ns);
            if (prc == CG_RT_ERR_TIMED_OUT) {
                rc = wait_predicate(semaphore, value);
                if (rc == CG_RT_ERR_TIMED_OUT) break;
            } else if (prc != CG_RT_OK) {
                cg_rt_platform_mutex_unlock(&semaphore->mutex);
                return prc;
            }
        }
    }

    cg_rt_platform_mutex_unlock(&semaphore->mutex);
    return rc;
}

cg_rt_status_t cg_rt_semaphore_query(cg_rt_semaphore_t *semaphore,
                                     uint64_t          *out_value) {
    if (semaphore == NULL || out_value == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    int32_t fc = atomic_load(&semaphore->failure_code);
    if (fc != 0) {
        *out_value = atomic_load(&semaphore->value);
        return CG_RT_ERR_ABORTED;
    }
    *out_value = atomic_load(&semaphore->value);
    return CG_RT_OK;
}

void cg_rt_semaphore_fail(cg_rt_semaphore_t *semaphore,
                          cg_rt_status_t     status) {
    if (semaphore == NULL || status >= 0) {
        return;
    }
    cg_rt_platform_mutex_lock(&semaphore->mutex);
    /* Only the first failure sticks; re-failing is a no-op. */
    int32_t expected = 0;
    atomic_compare_exchange_strong(&semaphore->failure_code, &expected, status);
    cg_rt_platform_cond_broadcast(&semaphore->cond);
    cg_rt_platform_mutex_unlock(&semaphore->mutex);
}
