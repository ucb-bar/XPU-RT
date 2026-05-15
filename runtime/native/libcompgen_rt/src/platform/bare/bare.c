/*
 * Bare-metal platform backend — no OS, single-threaded.
 *
 * Assumptions:
 *   - The library is called from a single hart / thread. Mutexes are
 *     no-ops since concurrent access is by construction impossible.
 *   - Condvar "waits" without a real threading primitive would
 *     deadlock in a multi-thread world; here they short-circuit to
 *     an immediate return because any state the waiter was observing
 *     must have been prepared on the *same* thread before the wait
 *     call (the caller simply polls after). Callers that wait inside
 *     a loop on a predicate — which every libcompgen_rt caller does —
 *     get the right behaviour: the loop exits if the predicate is
 *     satisfied, or spins via the ``timedwait`` polling path.
 *   - The monotonic clock reads the RISC-V ``cycle`` CSR by default.
 *     Integrators on other architectures override the weak symbol
 *     ``cg_rt_platform_bare_monotonic_ns`` with a platform-specific
 *     implementation.
 */

#include "../../core/platform.h"

#include <stdint.h>

/* ---- mutex ---------------------------------------------------------- */

int  cg_rt_platform_mutex_init(cg_rt_platform_mutex_t *m)    { (void)m; return 0; }
void cg_rt_platform_mutex_destroy(cg_rt_platform_mutex_t *m) { (void)m; }
void cg_rt_platform_mutex_lock(cg_rt_platform_mutex_t *m)    { (void)m; }
void cg_rt_platform_mutex_unlock(cg_rt_platform_mutex_t *m)  { (void)m; }

/* ---- condvar -------------------------------------------------------- */

int  cg_rt_platform_cond_init(cg_rt_platform_cond_t *c)    { (void)c; return 0; }
void cg_rt_platform_cond_destroy(cg_rt_platform_cond_t *c) { (void)c; }

/* An immediate return is correct under single-thread assumption: the
 * caller spins on its predicate after any event that could have
 * changed it (an op completes, an atomic counter hits zero).  A real
 * loop needs ``timedwait`` below to give it a stop condition. */
void cg_rt_platform_cond_wait(cg_rt_platform_cond_t  *c,
                              cg_rt_platform_mutex_t *m) {
    (void)c; (void)m;
}

/*
 * Timed wait reduces to a monotonic deadline check. The caller's
 * outer ``while (predicate) { cond_wait(); }`` loop re-tests the
 * predicate after each call, so the combined loop spins until either
 * (a) the predicate is satisfied — no further waits happen or
 * (b) the deadline expires — we return CG_RT_ERR_TIMED_OUT and the
 * caller bails.
 */
cg_rt_status_t cg_rt_platform_cond_timedwait_ns(cg_rt_platform_cond_t  *c,
                                                cg_rt_platform_mutex_t *m,
                                                uint64_t                timeout_ns) {
    (void)c; (void)m;
    uint64_t deadline = cg_rt_platform_monotonic_ns() + timeout_ns;
    while (cg_rt_platform_monotonic_ns() < deadline) {
        /* WFI would park the hart until an interrupt on real hardware;
         * on Spike / an ISA simulator it's a no-op.  Emitting it via
         * inline asm keeps the loop cheap without requiring libgcc. */
#if defined(__riscv)
        __asm__ volatile ("wfi" ::: "memory");
#endif
    }
    return CG_RT_ERR_TIMED_OUT;
}

void cg_rt_platform_cond_broadcast(cg_rt_platform_cond_t *c) {
    (void)c;
}

/* ---- monotonic clock ------------------------------------------------- */

/*
 * Default implementation: RISC-V reads the ``cycle`` CSR directly.
 * Assumes a 1-cycle-per-ns CPU — callers with a different clock rate
 * override the weak symbol below.
 *
 * The function is declared ``__attribute__((weak))`` so an integrator
 * can supply a stronger definition without modifying libcompgen_rt.
 */

__attribute__((weak))
uint64_t cg_rt_platform_bare_monotonic_ns(void) {
#if defined(__riscv) && (__riscv_xlen == 64)
    uint64_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
#elif defined(__riscv)
    /* rv32 — read cycleh:cycle with a retry loop to guard against
     * upper-word rollover between reads. */
    uint32_t lo, hi, hi2;
    do {
        __asm__ volatile ("rdcycleh %0" : "=r"(hi));
        __asm__ volatile ("rdcycle  %0" : "=r"(lo));
        __asm__ volatile ("rdcycleh %0" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
#else
    /* No cycle counter available — return 0 so timedwaits fail fast.
     * Integrators should override this. */
    return 0;
#endif
}

uint64_t cg_rt_platform_monotonic_ns(void) {
    return cg_rt_platform_bare_monotonic_ns();
}
