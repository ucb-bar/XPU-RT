/*
 * XPU-RT HAL — Device Vtable API
 *
 * Every HAL driver populates a `xpu_rt_device_vtable_t` and stores a pointer
 * to it inside the opaque `xpu_rt_device_s` structure.  Code that consumes
 * the HAL dispatches through these function pointers, making the API
 * target-agnostic.
 *
 * See docs/HAL_DESIGN.md for the full specification.
 */

#ifndef XPU_RT_HAL_H
#define XPU_RT_HAL_H

#include "xpu_rt/hal_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Device vtable                                                       */
/* ------------------------------------------------------------------ */

typedef struct xpu_rt_device_vtable_t {
    /* -- Lifecycle ------------------------------------------------- */
    void (*destroy)(xpu_rt_device_t device);

    /* -- Buffer management ----------------------------------------- */
    xpu_rt_status_t (*buffer_alloc)(xpu_rt_device_t device,
                                     size_t size,
                                     xpu_rt_buffer_t* out);

    void (*buffer_free)(xpu_rt_device_t device,
                        xpu_rt_buffer_t buffer);

    xpu_rt_status_t (*buffer_map)(xpu_rt_device_t device,
                                   xpu_rt_buffer_t buffer,
                                   void** out_ptr);

    void (*buffer_unmap)(xpu_rt_device_t device,
                         xpu_rt_buffer_t buffer);

    xpu_rt_status_t (*buffer_copy)(xpu_rt_device_t device,
                                    xpu_rt_buffer_t src,
                                    xpu_rt_buffer_t dst,
                                    size_t size);

    /* -- Dispatch -------------------------------------------------- */
    xpu_rt_status_t (*dispatch)(xpu_rt_device_t device,
                                 xpu_rt_executable_t exe,
                                 const void* args,
                                 size_t args_size);

    /* -- Synchronization ------------------------------------------- */
    xpu_rt_status_t (*sync)(xpu_rt_device_t device);

    /* -- Query ----------------------------------------------------- */
    xpu_rt_status_t (*query_i64)(xpu_rt_device_t device,
                                  xpu_rt_device_info_key_t key,
                                  int64_t* out);
} xpu_rt_device_vtable_t;

/* ------------------------------------------------------------------ */
/* Convenience inline dispatchers                                      */
/* ------------------------------------------------------------------ */

/*
 * Each inline function below dereferences the vtable stored inside the
 * device handle.  The concrete `xpu_rt_device_s` layout is defined in
 * hal.c (or the driver that creates the device).  We forward-declare a
 * helper here to retrieve the vtable from an opaque device pointer.
 */

const xpu_rt_device_vtable_t* xpu_rt_device_get_vtable(xpu_rt_device_t device);

static inline void xpu_rt_device_destroy(xpu_rt_device_t device) {
    xpu_rt_device_get_vtable(device)->destroy(device);
}

static inline xpu_rt_status_t xpu_rt_buffer_alloc(xpu_rt_device_t device,
                                                     size_t size,
                                                     xpu_rt_buffer_t* out) {
    return xpu_rt_device_get_vtable(device)->buffer_alloc(device, size, out);
}

static inline void xpu_rt_buffer_free(xpu_rt_device_t device,
                                        xpu_rt_buffer_t buffer) {
    xpu_rt_device_get_vtable(device)->buffer_free(device, buffer);
}

static inline xpu_rt_status_t xpu_rt_buffer_map(xpu_rt_device_t device,
                                                   xpu_rt_buffer_t buffer,
                                                   void** out_ptr) {
    return xpu_rt_device_get_vtable(device)->buffer_map(device, buffer, out_ptr);
}

static inline void xpu_rt_buffer_unmap(xpu_rt_device_t device,
                                         xpu_rt_buffer_t buffer) {
    xpu_rt_device_get_vtable(device)->buffer_unmap(device, buffer);
}

static inline xpu_rt_status_t xpu_rt_buffer_copy(xpu_rt_device_t device,
                                                    xpu_rt_buffer_t src,
                                                    xpu_rt_buffer_t dst,
                                                    size_t size) {
    return xpu_rt_device_get_vtable(device)->buffer_copy(device, src, dst, size);
}

static inline xpu_rt_status_t xpu_rt_dispatch(xpu_rt_device_t device,
                                                 xpu_rt_executable_t exe,
                                                 const void* args,
                                                 size_t args_size) {
    return xpu_rt_device_get_vtable(device)->dispatch(device, exe, args, args_size);
}

static inline xpu_rt_status_t xpu_rt_device_sync(xpu_rt_device_t device) {
    return xpu_rt_device_get_vtable(device)->sync(device);
}

static inline xpu_rt_status_t xpu_rt_device_query_i64(xpu_rt_device_t device,
                                                         xpu_rt_device_info_key_t key,
                                                         int64_t* out) {
    return xpu_rt_device_get_vtable(device)->query_i64(device, key, out);
}

/* ------------------------------------------------------------------ */
/* CPU reference driver constructor                                    */
/* ------------------------------------------------------------------ */

/**
 * Create a CPU reference device backed by host malloc / memcpy / dlopen.
 *
 * The returned device must eventually be destroyed via
 * `xpu_rt_device_destroy()`.
 */
xpu_rt_status_t xpu_rt_cpu_device_create(xpu_rt_device_t* out_device);

#ifdef __cplusplus
}
#endif

#endif /* XPU_RT_HAL_H */
