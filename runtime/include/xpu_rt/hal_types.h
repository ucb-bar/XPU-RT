/*
 * XPU-RT HAL Types
 *
 * Opaque handle types, status codes, and info-key enumerations used across
 * the entire HAL surface.  This header is intentionally free of function
 * declarations so that lightweight consumers (e.g. generated code that only
 * needs the types) do not pull in the full vtable definition.
 */

#ifndef XPU_RT_HAL_TYPES_H
#define XPU_RT_HAL_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum xpu_rt_status_t {
    XPU_RT_STATUS_OK = 0,
    XPU_RT_STATUS_ERROR = 1,
    XPU_RT_STATUS_OUT_OF_MEMORY = 2,
    XPU_RT_STATUS_UNIMPLEMENTED = 3,
    XPU_RT_STATUS_INVALID_ARGUMENT = 4,
    XPU_RT_STATUS_NOT_FOUND = 5,
} xpu_rt_status_t;

/* ------------------------------------------------------------------ */
/* Device info query keys                                              */
/* ------------------------------------------------------------------ */

typedef enum xpu_rt_device_info_key_t {
    XPU_RT_DEVICE_INFO_DEVICE_TYPE = 0,
    XPU_RT_DEVICE_INFO_MEMORY_TOTAL = 1,
    XPU_RT_DEVICE_INFO_COMPUTE_UNITS = 2,
    XPU_RT_DEVICE_INFO_MAX_DISPATCH_CONCURRENCY = 3,
    XPU_RT_DEVICE_INFO_ADDRESS_SPACE_COUNT = 4,
    XPU_RT_DEVICE_INFO_SUPPORTS_ASYNC_DMA = 5,
    XPU_RT_DEVICE_INFO_MAX_ALLOC_SIZE = 6,
} xpu_rt_device_info_key_t;

/* ------------------------------------------------------------------ */
/* Opaque handle types                                                 */
/* ------------------------------------------------------------------ */

typedef struct xpu_rt_device_s*     xpu_rt_device_t;
typedef struct xpu_rt_buffer_s*     xpu_rt_buffer_t;
typedef struct xpu_rt_executable_s* xpu_rt_executable_t;

#ifdef __cplusplus
}
#endif

#endif /* XPU_RT_HAL_TYPES_H */
