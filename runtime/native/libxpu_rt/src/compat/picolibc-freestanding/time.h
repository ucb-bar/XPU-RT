/*
 * Stub fragments of POSIX <time.h> for picolibc on bare-metal riscv64.
 *
 * Includes picolibc's real <time.h> (which gives us `time_t`,
 * `struct tm`, `time()`, `localtime()`, etc. — all there), then adds
 * the few clock_* symbols XNNPACK references unconditionally in its
 * runtime profiling path. picolibc's <time.h> on its own omits
 * clock_gettime / CLOCK_MONOTONIC under bare-metal because there is
 * no underlying clock service.
 *
 * The stubs report a monotonically-increasing fake timestamp (zero
 * delta), which is fine because we never enable XNNPACK profiling.
 */

#ifndef XPU_RT_COMPAT_TIME_H
#define XPU_RT_COMPAT_TIME_H 1

#include_next <time.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME  0
#endif
#ifndef CLOCK_UPTIME_RAW
#define CLOCK_UPTIME_RAW 8
#endif

#ifndef _CLOCKID_T_DECLARED
typedef int clockid_t;
#define _CLOCKID_T_DECLARED
#endif

static inline int clock_gettime(clockid_t id, struct timespec* ts) {
    (void)id;
    if (ts) { ts->tv_sec = 0; ts->tv_nsec = 0; }
    return 0;
}

/* macOS-style alternate that XNNPACK references inside a #ifdef. */
static inline uint64_t clock_gettime_nsec_np(clockid_t id) {
    (void)id;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* XPU_RT_COMPAT_TIME_H */
