/*
 * libcompgen_rt — Zephyr platform glue.
 *
 * Maps the platform abstraction (cg_rt_platform_mutex_t /
 * cg_rt_platform_cond_t / monotonic_ns) onto Zephyr's kernel
 * primitives:
 *
 *   - k_mutex_init / k_mutex_lock / k_mutex_unlock for mutexes.
 *   - k_condvar_init / k_condvar_wait / k_condvar_broadcast for
 *     condvars.
 *   - k_uptime_ticks() + k_ticks_to_ns_floor64 for the monotonic
 *     clock.
 *
 * The implementation is compile-guarded by ``CG_RT_PLATFORM_ZEPHYR``.
 * Outside Zephyr builds the file compiles to no symbols (the guard
 * elides every definition), so libcompgen_rt remains a single
 * portable translation unit across all platforms.
 *
 * Realness: declared at realness_level = read_only.
 * The CI path is Zephyr's ``native_posix`` simulator + the existing
 * libcompgen_rt cpu_sync conformance suite. Hardware-backed
 * verification (Saturn OPU, FreeRTOS-style boards) lands as
 * follow-up.
 */

#include "../../core/internal.h"

#ifdef CG_RT_PLATFORM_ZEPHYR

#include <zephyr/kernel.h>

int cg_rt_platform_mutex_init(cg_rt_platform_mutex_t *m) {
    struct k_mutex *km = (struct k_mutex *)m->storage;
    k_mutex_init(km);
    return 0;
}

void cg_rt_platform_mutex_destroy(cg_rt_platform_mutex_t *m) {
    (void)m;  /* Zephyr mutex objects don't need explicit teardown. */
}

void cg_rt_platform_mutex_lock(cg_rt_platform_mutex_t *m) {
    struct k_mutex *km = (struct k_mutex *)m->storage;
    k_mutex_lock(km, K_FOREVER);
}

void cg_rt_platform_mutex_unlock(cg_rt_platform_mutex_t *m) {
    struct k_mutex *km = (struct k_mutex *)m->storage;
    k_mutex_unlock(km);
}

int cg_rt_platform_cond_init(cg_rt_platform_cond_t *c) {
    struct k_condvar *kc = (struct k_condvar *)c->storage;
    k_condvar_init(kc);
    return 0;
}

void cg_rt_platform_cond_destroy(cg_rt_platform_cond_t *c) {
    (void)c;
}

void cg_rt_platform_cond_wait(cg_rt_platform_cond_t  *c,
                              cg_rt_platform_mutex_t *m) {
    struct k_condvar *kc = (struct k_condvar *)c->storage;
    struct k_mutex   *km = (struct k_mutex   *)m->storage;
    k_condvar_wait(kc, km, K_FOREVER);
}

cg_rt_status_t cg_rt_platform_cond_timedwait_ns(cg_rt_platform_cond_t  *c,
                                                cg_rt_platform_mutex_t *m,
                                                uint64_t                timeout_ns) {
    struct k_condvar *kc = (struct k_condvar *)c->storage;
    struct k_mutex   *km = (struct k_mutex   *)m->storage;
    k_timeout_t to = K_NSEC(timeout_ns);
    int rc = k_condvar_wait(kc, km, to);
    return (rc == 0) ? CG_RT_OK : CG_RT_ERR_TIMED_OUT;
}

void cg_rt_platform_cond_broadcast(cg_rt_platform_cond_t *c) {
    struct k_condvar *kc = (struct k_condvar *)c->storage;
    k_condvar_broadcast(kc);
}

uint64_t cg_rt_platform_monotonic_ns(void) {
    return (uint64_t)k_ticks_to_ns_floor64(k_uptime_ticks());
}

#endif /* CG_RT_PLATFORM_ZEPHYR */
