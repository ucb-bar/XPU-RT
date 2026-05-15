/*
 * CompGen HAL — Device Vtable API
 *
 * Every HAL driver populates a `compgen_device_vtable_t` and stores a pointer
 * to it inside the opaque `compgen_device_s` structure.  Code that consumes
 * the HAL dispatches through these function pointers, making the API
 * target-agnostic.
 *
 * See docs/HAL_DESIGN.md for the full specification.
 */

#ifndef COMPGEN_HAL_H
#define COMPGEN_HAL_H

#include "compgen/hal_types.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Device vtable                                                       */
/* ------------------------------------------------------------------ */

typedef struct compgen_device_vtable_t {
    /* -- Lifecycle ------------------------------------------------- */
    void (*destroy)(compgen_device_t device);

    /* -- Buffer management ----------------------------------------- */
    compgen_status_t (*buffer_alloc)(compgen_device_t device,
                                     size_t size,
                                     compgen_buffer_t* out);

    void (*buffer_free)(compgen_device_t device,
                        compgen_buffer_t buffer);

    compgen_status_t (*buffer_map)(compgen_device_t device,
                                   compgen_buffer_t buffer,
                                   void** out_ptr);

    void (*buffer_unmap)(compgen_device_t device,
                         compgen_buffer_t buffer);

    compgen_status_t (*buffer_copy)(compgen_device_t device,
                                    compgen_buffer_t src,
                                    compgen_buffer_t dst,
                                    size_t size);

    /* -- Dispatch -------------------------------------------------- */
    compgen_status_t (*dispatch)(compgen_device_t device,
                                 compgen_executable_t exe,
                                 const void* args,
                                 size_t args_size);

    /* -- Synchronization ------------------------------------------- */
    compgen_status_t (*sync)(compgen_device_t device);

    /* -- Query ----------------------------------------------------- */
    compgen_status_t (*query_i64)(compgen_device_t device,
                                  compgen_device_info_key_t key,
                                  int64_t* out);
} compgen_device_vtable_t;

/* ------------------------------------------------------------------ */
/* Convenience inline dispatchers                                      */
/* ------------------------------------------------------------------ */

/*
 * Each inline function below dereferences the vtable stored inside the
 * device handle.  The concrete `compgen_device_s` layout is defined in
 * hal.c (or the driver that creates the device).  We forward-declare a
 * helper here to retrieve the vtable from an opaque device pointer.
 */

const compgen_device_vtable_t* compgen_device_get_vtable(compgen_device_t device);

static inline void compgen_device_destroy(compgen_device_t device) {
    compgen_device_get_vtable(device)->destroy(device);
}

static inline compgen_status_t compgen_buffer_alloc(compgen_device_t device,
                                                     size_t size,
                                                     compgen_buffer_t* out) {
    return compgen_device_get_vtable(device)->buffer_alloc(device, size, out);
}

static inline void compgen_buffer_free(compgen_device_t device,
                                        compgen_buffer_t buffer) {
    compgen_device_get_vtable(device)->buffer_free(device, buffer);
}

static inline compgen_status_t compgen_buffer_map(compgen_device_t device,
                                                   compgen_buffer_t buffer,
                                                   void** out_ptr) {
    return compgen_device_get_vtable(device)->buffer_map(device, buffer, out_ptr);
}

static inline void compgen_buffer_unmap(compgen_device_t device,
                                         compgen_buffer_t buffer) {
    compgen_device_get_vtable(device)->buffer_unmap(device, buffer);
}

static inline compgen_status_t compgen_buffer_copy(compgen_device_t device,
                                                    compgen_buffer_t src,
                                                    compgen_buffer_t dst,
                                                    size_t size) {
    return compgen_device_get_vtable(device)->buffer_copy(device, src, dst, size);
}

static inline compgen_status_t compgen_dispatch(compgen_device_t device,
                                                 compgen_executable_t exe,
                                                 const void* args,
                                                 size_t args_size) {
    return compgen_device_get_vtable(device)->dispatch(device, exe, args, args_size);
}

static inline compgen_status_t compgen_device_sync(compgen_device_t device) {
    return compgen_device_get_vtable(device)->sync(device);
}

static inline compgen_status_t compgen_device_query_i64(compgen_device_t device,
                                                         compgen_device_info_key_t key,
                                                         int64_t* out) {
    return compgen_device_get_vtable(device)->query_i64(device, key, out);
}

/* ------------------------------------------------------------------ */
/* CPU reference driver constructor                                    */
/* ------------------------------------------------------------------ */

/**
 * Create a CPU reference device backed by host malloc / memcpy / dlopen.
 *
 * The returned device must eventually be destroyed via
 * `compgen_device_destroy()`.
 */
compgen_status_t compgen_cpu_device_create(compgen_device_t* out_device);

#ifdef __cplusplus
}
#endif

#endif /* COMPGEN_HAL_H */
