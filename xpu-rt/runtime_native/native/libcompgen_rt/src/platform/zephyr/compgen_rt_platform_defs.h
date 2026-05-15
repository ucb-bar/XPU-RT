/*
 * Zephyr backend definitions.
 *
 * Maps the platform abstraction onto Zephyr's k_mutex / k_condvar.
 * The actual implementation lives in ``zephyr_platform.c`` and is
 * compile-guarded by ``CG_RT_PLATFORM_ZEPHYR``.
 *
 * On the native_posix Zephyr simulator the layer behaves identically
 * to the posix backend; on real RTOS targets ``k_thread`` workers
 * handle queue execution and ``k_sem`` underlies timeline semaphore
 * implementations. CI uses native_posix.
 */

#ifndef COMPGEN_RT_PLATFORM_DEFS_ZEPHYR_H_
#define COMPGEN_RT_PLATFORM_DEFS_ZEPHYR_H_

#include <stdint.h>

#ifdef CG_RT_PLATFORM_ZEPHYR
/* On real Zephyr builds the public types alias Zephyr primitives.
 * The header would normally pull in <zephyr/kernel.h>; here we keep
 * the type-only header as a placeholder so the runtime compiles in
 * isolation and the Zephyr include enters at the .c file in
 * platform/zephyr/zephyr_platform.c. */
typedef struct {
    /* sizeof(struct k_mutex) is ~48 bytes on Zephyr 3.x; reserve a
     * generous padded slot so the structure layout is stable. */
    uint8_t  storage[64];
} cg_rt_platform_mutex_t;

typedef struct {
    uint8_t  storage[64];
} cg_rt_platform_cond_t;

#else
/* Off-Zephyr builds — keep the type non-zero so structs containing
 * it still compile (the actual definitions never run). */
typedef struct {
    uint8_t _pad;
} cg_rt_platform_mutex_t;

typedef struct {
    uint8_t _pad;
} cg_rt_platform_cond_t;
#endif

#endif /* COMPGEN_RT_PLATFORM_DEFS_ZEPHYR_H_ */
