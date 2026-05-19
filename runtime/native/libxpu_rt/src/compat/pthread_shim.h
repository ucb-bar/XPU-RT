/*
 * Minimal <pthread.h> shim for bare-metal newlib+pk cross-compiles.
 *
 * XNNPACK calls pthread_once / PTHREAD_ONCE_INIT unconditionally
 * (init-once.h:38) even when the threadpool itself is the single-
 * threaded shim. newlib bare-metal does not provide <pthread.h>, so
 * we synthesize just enough of it for the single-threaded path:
 *
 *   - pthread_once_t is an int.
 *   - PTHREAD_ONCE_INIT is 0.
 *   - pthread_once(once, fn) runs fn exactly once across all callers
 *     in a single-threaded program — atomically by construction.
 *
 * This header is force-included on every XNNPACK translation unit via
 * `-include` when XPURT_RISCV_BARE is set. It is a no-op header on
 * any platform that already has <pthread.h>.
 */

#ifndef XPU_RT_PTHREAD_SHIM_H
#define XPU_RT_PTHREAD_SHIM_H

#if defined(XPURT_RISCV_BARE) && XPURT_RISCV_BARE && !defined(_PTHREAD_H)

typedef int pthread_once_t;
#define PTHREAD_ONCE_INIT 0

static inline int pthread_once(pthread_once_t* once, void (*fn)(void)) {
    if (*once == 0) {
        fn();
        *once = 1;
    }
    return 0;
}

/* Sentinels for the other pthread types XNNPACK may name in
 * declarations even when the codepath is never taken on bare-metal. */
typedef int pthread_mutex_t;
typedef int pthread_t;
typedef int pthread_cond_t;
typedef int pthread_attr_t;

#define PTHREAD_MUTEX_INITIALIZER 0
#define PTHREAD_COND_INITIALIZER  0

static inline int pthread_mutex_lock(pthread_mutex_t* m)   { (void)m; return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t* m) { (void)m; return 0; }
static inline int pthread_mutex_init(pthread_mutex_t* m, const void* a) {
    (void)m; (void)a; return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t* m) { (void)m; return 0; }

#endif /* XPURT_RISCV_BARE */

#endif /* XPU_RT_PTHREAD_SHIM_H */
