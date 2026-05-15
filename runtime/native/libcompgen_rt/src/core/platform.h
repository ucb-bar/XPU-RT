/*
 * Platform abstraction — a tiny shim around mutex + condvar + clock.
 *
 * Every primitive in libcompgen_rt that historically used pthreads
 * goes through this interface instead. Platform backends (posix,
 * bare, future zephyr/freertos) supply concrete types via
 * ``cg_rt_platform_defs.h`` — one backend is selected at build time.
 *
 * The interface is deliberately minimal: mutex lock/unlock, condvar
 * wait + broadcast + timed_wait, monotonic + realtime nanosecond
 * clocks. Bare-metal backends that lack threading implement trivial
 * no-op mutexes (single-threaded by assumption) and a polling
 * timed_wait built on a deadline check.
 *
 * Usage pattern:
 *
 *     cg_rt_platform_mutex_t m;
 *     cg_rt_platform_cond_t  c;
 *     cg_rt_platform_mutex_init(&m);
 *     cg_rt_platform_cond_init(&c);
 *     cg_rt_platform_mutex_lock(&m);
 *     while (!ready()) cg_rt_platform_cond_wait(&c, &m);
 *     cg_rt_platform_mutex_unlock(&m);
 *     cg_rt_platform_cond_broadcast(&c);
 *     cg_rt_platform_cond_destroy(&c);
 *     cg_rt_platform_mutex_destroy(&m);
 */

#ifndef COMPGEN_RT_PLATFORM_H_
#define COMPGEN_RT_PLATFORM_H_

#include <stddef.h>
#include <stdint.h>

#include "compgen_rt/compgen_rt.h"

/* Each backend provides the concrete types + their sizes here. */
#include "compgen_rt_platform_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Return values: 0 = success, non-zero = backend-specific error. */
int  cg_rt_platform_mutex_init(cg_rt_platform_mutex_t *m);
void cg_rt_platform_mutex_destroy(cg_rt_platform_mutex_t *m);
void cg_rt_platform_mutex_lock(cg_rt_platform_mutex_t *m);
void cg_rt_platform_mutex_unlock(cg_rt_platform_mutex_t *m);

int  cg_rt_platform_cond_init(cg_rt_platform_cond_t *c);
void cg_rt_platform_cond_destroy(cg_rt_platform_cond_t *c);

/* Wait forever on the condvar. Caller must hold ``m``. */
void cg_rt_platform_cond_wait(cg_rt_platform_cond_t  *c,
                              cg_rt_platform_mutex_t *m);

/*
 * Timed wait. ``timeout_ns`` is a RELATIVE timeout (ns from now).
 * Returns 0 on signal, CG_RT_ERR_TIMED_OUT on deadline expiry.
 * Callers are expected to loop on the condition and re-check the
 * predicate on wake-up.
 */
cg_rt_status_t cg_rt_platform_cond_timedwait_ns(cg_rt_platform_cond_t  *c,
                                                cg_rt_platform_mutex_t *m,
                                                uint64_t                timeout_ns);

void cg_rt_platform_cond_broadcast(cg_rt_platform_cond_t *c);

/* Monotonic clock in nanoseconds. Only used for deadline math; need
 * not be wall-clock accurate. */
uint64_t cg_rt_platform_monotonic_ns(void);

#ifdef __cplusplus
}
#endif

#endif /* COMPGEN_RT_PLATFORM_H_ */
