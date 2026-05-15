/*
 * XPU-RT CPU Reference Driver — Internal Header
 *
 * Defines the concrete struct layouts behind the opaque HAL handles for
 * the CPU (host) reference implementation.  This header is private to the
 * HAL implementation; consumers should include <xpu_rt/hal.h> instead.
 */

#ifndef XPU_RT_CPU_DRIVER_H
#define XPU_RT_CPU_DRIVER_H

#include "hal_internal.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A CPU buffer is a thin wrapper around a heap allocation.
 */
struct xpu_rt_buffer_s {
    void*  data;
    size_t size;
};

/**
 * A CPU "executable" holds a dlopen handle and the resolved entry-point
 * function pointer.
 *
 * For the CPU reference driver the executable format is a shared object
 * (.so) loaded via dlopen.  The entry point is resolved via dlsym.
 */
typedef void (*xpu_rt_cpu_kernel_fn)(const void* args, size_t args_size);

struct xpu_rt_executable_s {
    void*                  dl_handle;   /* dlopen handle  */
    xpu_rt_cpu_kernel_fn  entry;       /* resolved entry */
};

#ifdef __cplusplus
}
#endif

#endif /* XPU_RT_CPU_DRIVER_H */
