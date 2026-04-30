/*
 * POSIX backend definitions — pthread-based mutex + condvar, realtime
 * clock for the deadline. Included via the ``platform.h`` header on
 * any build where ``-I.../platform/posix`` wins the include-path
 * race; CMake guarantees this for Linux/macOS.
 */

#ifndef COMPGEN_RT_PLATFORM_DEFS_POSIX_H_
#define COMPGEN_RT_PLATFORM_DEFS_POSIX_H_

#include <pthread.h>

typedef pthread_mutex_t cg_rt_platform_mutex_t;
typedef pthread_cond_t  cg_rt_platform_cond_t;

#endif
