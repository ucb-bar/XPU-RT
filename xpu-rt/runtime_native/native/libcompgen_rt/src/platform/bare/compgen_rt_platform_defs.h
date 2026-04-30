/*
 * Bare-metal backend definitions — no OS, no threading.
 *
 * The library, under this backend, runs single-threaded on the
 * caller's context. Mutexes are no-ops (nothing else can race), and
 * condvars don't block — waiters that would have parked on a condvar
 * instead spin on the backing predicate via polling inside the
 * condvar's ``timedwait_ns``.
 *
 * The integrator must supply one function somewhere in their link:
 *
 *     extern uint64_t cg_rt_platform_bare_monotonic_ns(void);
 *
 * on RISC-V this is typically ``rdcycle`` scaled by the CPU clock
 * rate (see ``bare.c`` for a default implementation that reads the
 * ``cycle`` CSR and assumes a 1 ns tick — override via the weak
 * symbol if your clock differs).
 *
 * Mutex and condvar types are empty bytes (padding so the struct is
 * non-zero in C); the actual state lives in the caller's data
 * structures using atomics.
 */

#ifndef COMPGEN_RT_PLATFORM_DEFS_BARE_H_
#define COMPGEN_RT_PLATFORM_DEFS_BARE_H_

#include <stdint.h>

typedef struct {
    uint8_t _pad;
} cg_rt_platform_mutex_t;

typedef struct {
    uint8_t _pad;
} cg_rt_platform_cond_t;

#endif
