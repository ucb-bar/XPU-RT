/*
 * POSIX platform backend — pthread-based.
 *
 * Preserves the exact behaviour the library had before the platform
 * layer existed. Realtime-clock deadlines match the condvar
 * ``pthread_cond_timedwait`` semantics.
 */

#include "../../core/platform.h"

#include <errno.h>
#include <time.h>

int cg_rt_platform_mutex_init(cg_rt_platform_mutex_t *m) {
    return pthread_mutex_init(m, NULL);
}

void cg_rt_platform_mutex_destroy(cg_rt_platform_mutex_t *m) {
    pthread_mutex_destroy(m);
}

void cg_rt_platform_mutex_lock(cg_rt_platform_mutex_t *m) {
    pthread_mutex_lock(m);
}

void cg_rt_platform_mutex_unlock(cg_rt_platform_mutex_t *m) {
    pthread_mutex_unlock(m);
}

int cg_rt_platform_cond_init(cg_rt_platform_cond_t *c) {
    return pthread_cond_init(c, NULL);
}

void cg_rt_platform_cond_destroy(cg_rt_platform_cond_t *c) {
    pthread_cond_destroy(c);
}

void cg_rt_platform_cond_wait(cg_rt_platform_cond_t  *c,
                              cg_rt_platform_mutex_t *m) {
    pthread_cond_wait(c, m);
}

cg_rt_status_t cg_rt_platform_cond_timedwait_ns(cg_rt_platform_cond_t  *c,
                                                cg_rt_platform_mutex_t *m,
                                                uint64_t                timeout_ns) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    uint64_t secs = timeout_ns / UINT64_C(1000000000);
    uint64_t nsec = timeout_ns % UINT64_C(1000000000);
    deadline.tv_sec += (time_t)secs;
    deadline.tv_nsec += (long)nsec;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    int ec = pthread_cond_timedwait(c, m, &deadline);
    if (ec == 0) return CG_RT_OK;
    if (ec == ETIMEDOUT) return CG_RT_ERR_TIMED_OUT;
    return CG_RT_ERR_UNKNOWN;
}

void cg_rt_platform_cond_broadcast(cg_rt_platform_cond_t *c) {
    pthread_cond_broadcast(c);
}

uint64_t cg_rt_platform_monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
