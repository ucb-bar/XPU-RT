/*
 * CompGen CPU Reference Driver
 *
 * Provides a complete HAL implementation backed by host memory (malloc),
 * memcpy for copies, and dlopen/dlsym for kernel dispatch.  This driver
 * is the simplest possible HAL and is used for:
 *
 *   1. Unit testing the rest of the runtime without real hardware.
 *   2. Golden-model execution for verification.
 *   3. As a template for writing new target drivers.
 *
 * All operations are synchronous; `sync` is a no-op.
 */

#include "cpu_driver.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Forward declarations of vtable methods                              */
/* ------------------------------------------------------------------ */

static void             cpu_destroy(compgen_device_t device);
static compgen_status_t cpu_buffer_alloc(compgen_device_t device,
                                          size_t size,
                                          compgen_buffer_t* out);
static void             cpu_buffer_free(compgen_device_t device,
                                         compgen_buffer_t buffer);
static compgen_status_t cpu_buffer_map(compgen_device_t device,
                                        compgen_buffer_t buffer,
                                        void** out_ptr);
static void             cpu_buffer_unmap(compgen_device_t device,
                                          compgen_buffer_t buffer);
static compgen_status_t cpu_buffer_copy(compgen_device_t device,
                                         compgen_buffer_t src,
                                         compgen_buffer_t dst,
                                         size_t size);
static compgen_status_t cpu_dispatch(compgen_device_t device,
                                      compgen_executable_t exe,
                                      const void* args,
                                      size_t args_size);
static compgen_status_t cpu_sync(compgen_device_t device);
static compgen_status_t cpu_query_i64(compgen_device_t device,
                                       compgen_device_info_key_t key,
                                       int64_t* out);

/* ------------------------------------------------------------------ */
/* Static vtable instance                                              */
/* ------------------------------------------------------------------ */

static const compgen_device_vtable_t cpu_vtable = {
    .destroy      = cpu_destroy,
    .buffer_alloc = cpu_buffer_alloc,
    .buffer_free  = cpu_buffer_free,
    .buffer_map   = cpu_buffer_map,
    .buffer_unmap = cpu_buffer_unmap,
    .buffer_copy  = cpu_buffer_copy,
    .dispatch     = cpu_dispatch,
    .sync         = cpu_sync,
    .query_i64    = cpu_query_i64,
};

/* ------------------------------------------------------------------ */
/* Device lifecycle                                                    */
/* ------------------------------------------------------------------ */

compgen_status_t
compgen_cpu_device_create(compgen_device_t* out_device)
{
    if (!out_device) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    struct compgen_device_s* dev = calloc(1, sizeof(*dev));
    if (!dev) {
        return COMPGEN_STATUS_OUT_OF_MEMORY;
    }

    dev->vtable = &cpu_vtable;
    *out_device = dev;
    return COMPGEN_STATUS_OK;
}

static void
cpu_destroy(compgen_device_t device)
{
    free(device);
}

/* ------------------------------------------------------------------ */
/* Buffer management                                                   */
/* ------------------------------------------------------------------ */

static compgen_status_t
cpu_buffer_alloc(compgen_device_t device,
                 size_t size,
                 compgen_buffer_t* out)
{
    (void)device;

    if (!out) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }
    if (size == 0) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    struct compgen_buffer_s* buf = calloc(1, sizeof(*buf));
    if (!buf) {
        return COMPGEN_STATUS_OUT_OF_MEMORY;
    }

    buf->data = malloc(size);
    if (!buf->data) {
        free(buf);
        return COMPGEN_STATUS_OUT_OF_MEMORY;
    }

    buf->size = size;
    *out = buf;
    return COMPGEN_STATUS_OK;
}

static void
cpu_buffer_free(compgen_device_t device, compgen_buffer_t buffer)
{
    (void)device;

    if (!buffer) {
        return;
    }
    free(buffer->data);
    free(buffer);
}

static compgen_status_t
cpu_buffer_map(compgen_device_t device,
               compgen_buffer_t buffer,
               void** out_ptr)
{
    (void)device;

    if (!buffer || !out_ptr) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    /* CPU buffers are always host-accessible; just return the pointer. */
    *out_ptr = buffer->data;
    return COMPGEN_STATUS_OK;
}

static void
cpu_buffer_unmap(compgen_device_t device, compgen_buffer_t buffer)
{
    /* No-op on CPU — the pointer is always valid. */
    (void)device;
    (void)buffer;
}

static compgen_status_t
cpu_buffer_copy(compgen_device_t device,
                compgen_buffer_t src,
                compgen_buffer_t dst,
                size_t size)
{
    (void)device;

    if (!src || !dst) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }
    if (size > src->size || size > dst->size) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    memcpy(dst->data, src->data, size);
    return COMPGEN_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

static compgen_status_t
cpu_dispatch(compgen_device_t device,
             compgen_executable_t exe,
             const void* args,
             size_t args_size)
{
    (void)device;

    if (!exe || !exe->entry) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    exe->entry(args, args_size);
    return COMPGEN_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Synchronization                                                     */
/* ------------------------------------------------------------------ */

static compgen_status_t
cpu_sync(compgen_device_t device)
{
    /* CPU execution is synchronous — nothing to wait for. */
    (void)device;
    return COMPGEN_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Query                                                               */
/* ------------------------------------------------------------------ */

static compgen_status_t
cpu_query_i64(compgen_device_t device,
              compgen_device_info_key_t key,
              int64_t* out)
{
    (void)device;

    if (!out) {
        return COMPGEN_STATUS_INVALID_ARGUMENT;
    }

    switch (key) {
    case COMPGEN_DEVICE_INFO_DEVICE_TYPE:
        *out = 0; /* 0 = CPU */
        return COMPGEN_STATUS_OK;

    case COMPGEN_DEVICE_INFO_MEMORY_TOTAL: {
        long pages     = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            *out = (int64_t)pages * (int64_t)page_size;
        } else {
            *out = 0;
        }
        return COMPGEN_STATUS_OK;
    }

    case COMPGEN_DEVICE_INFO_COMPUTE_UNITS: {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        *out = (ncpu > 0) ? (int64_t)ncpu : 1;
        return COMPGEN_STATUS_OK;
    }

    case COMPGEN_DEVICE_INFO_MAX_DISPATCH_CONCURRENCY:
        *out = 1; /* single-threaded reference driver */
        return COMPGEN_STATUS_OK;

    case COMPGEN_DEVICE_INFO_ADDRESS_SPACE_COUNT:
        *out = 1; /* host DRAM only */
        return COMPGEN_STATUS_OK;

    case COMPGEN_DEVICE_INFO_SUPPORTS_ASYNC_DMA:
        *out = 0; /* no async DMA on CPU */
        return COMPGEN_STATUS_OK;

    case COMPGEN_DEVICE_INFO_MAX_ALLOC_SIZE:
        *out = (int64_t)((size_t)-1 >> 1); /* SIZE_MAX / 2 as a safe i64 */
        return COMPGEN_STATUS_OK;

    default:
        return COMPGEN_STATUS_NOT_FOUND;
    }
}
