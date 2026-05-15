/*
 * XPU-RT CPU Reference Driver
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

static void             cpu_destroy(xpu_rt_device_t device);
static xpu_rt_status_t cpu_buffer_alloc(xpu_rt_device_t device,
                                          size_t size,
                                          xpu_rt_buffer_t* out);
static void             cpu_buffer_free(xpu_rt_device_t device,
                                         xpu_rt_buffer_t buffer);
static xpu_rt_status_t cpu_buffer_map(xpu_rt_device_t device,
                                        xpu_rt_buffer_t buffer,
                                        void** out_ptr);
static void             cpu_buffer_unmap(xpu_rt_device_t device,
                                          xpu_rt_buffer_t buffer);
static xpu_rt_status_t cpu_buffer_copy(xpu_rt_device_t device,
                                         xpu_rt_buffer_t src,
                                         xpu_rt_buffer_t dst,
                                         size_t size);
static xpu_rt_status_t cpu_dispatch(xpu_rt_device_t device,
                                      xpu_rt_executable_t exe,
                                      const void* args,
                                      size_t args_size);
static xpu_rt_status_t cpu_sync(xpu_rt_device_t device);
static xpu_rt_status_t cpu_query_i64(xpu_rt_device_t device,
                                       xpu_rt_device_info_key_t key,
                                       int64_t* out);

/* ------------------------------------------------------------------ */
/* Static vtable instance                                              */
/* ------------------------------------------------------------------ */

static const xpu_rt_device_vtable_t cpu_vtable = {
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

xpu_rt_status_t
xpu_rt_cpu_device_create(xpu_rt_device_t* out_device)
{
    if (!out_device) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    struct xpu_rt_device_s* dev = calloc(1, sizeof(*dev));
    if (!dev) {
        return XPU_RT_STATUS_OUT_OF_MEMORY;
    }

    dev->vtable = &cpu_vtable;
    *out_device = dev;
    return XPU_RT_STATUS_OK;
}

static void
cpu_destroy(xpu_rt_device_t device)
{
    free(device);
}

/* ------------------------------------------------------------------ */
/* Buffer management                                                   */
/* ------------------------------------------------------------------ */

static xpu_rt_status_t
cpu_buffer_alloc(xpu_rt_device_t device,
                 size_t size,
                 xpu_rt_buffer_t* out)
{
    (void)device;

    if (!out) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }
    if (size == 0) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    struct xpu_rt_buffer_s* buf = calloc(1, sizeof(*buf));
    if (!buf) {
        return XPU_RT_STATUS_OUT_OF_MEMORY;
    }

    buf->data = malloc(size);
    if (!buf->data) {
        free(buf);
        return XPU_RT_STATUS_OUT_OF_MEMORY;
    }

    buf->size = size;
    *out = buf;
    return XPU_RT_STATUS_OK;
}

static void
cpu_buffer_free(xpu_rt_device_t device, xpu_rt_buffer_t buffer)
{
    (void)device;

    if (!buffer) {
        return;
    }
    free(buffer->data);
    free(buffer);
}

static xpu_rt_status_t
cpu_buffer_map(xpu_rt_device_t device,
               xpu_rt_buffer_t buffer,
               void** out_ptr)
{
    (void)device;

    if (!buffer || !out_ptr) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    /* CPU buffers are always host-accessible; just return the pointer. */
    *out_ptr = buffer->data;
    return XPU_RT_STATUS_OK;
}

static void
cpu_buffer_unmap(xpu_rt_device_t device, xpu_rt_buffer_t buffer)
{
    /* No-op on CPU — the pointer is always valid. */
    (void)device;
    (void)buffer;
}

static xpu_rt_status_t
cpu_buffer_copy(xpu_rt_device_t device,
                xpu_rt_buffer_t src,
                xpu_rt_buffer_t dst,
                size_t size)
{
    (void)device;

    if (!src || !dst) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }
    if (size > src->size || size > dst->size) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    memcpy(dst->data, src->data, size);
    return XPU_RT_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

static xpu_rt_status_t
cpu_dispatch(xpu_rt_device_t device,
             xpu_rt_executable_t exe,
             const void* args,
             size_t args_size)
{
    (void)device;

    if (!exe || !exe->entry) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    exe->entry(args, args_size);
    return XPU_RT_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Synchronization                                                     */
/* ------------------------------------------------------------------ */

static xpu_rt_status_t
cpu_sync(xpu_rt_device_t device)
{
    /* CPU execution is synchronous — nothing to wait for. */
    (void)device;
    return XPU_RT_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Query                                                               */
/* ------------------------------------------------------------------ */

static xpu_rt_status_t
cpu_query_i64(xpu_rt_device_t device,
              xpu_rt_device_info_key_t key,
              int64_t* out)
{
    (void)device;

    if (!out) {
        return XPU_RT_STATUS_INVALID_ARGUMENT;
    }

    switch (key) {
    case XPU_RT_DEVICE_INFO_DEVICE_TYPE:
        *out = 0; /* 0 = CPU */
        return XPU_RT_STATUS_OK;

    case XPU_RT_DEVICE_INFO_MEMORY_TOTAL: {
        long pages     = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            *out = (int64_t)pages * (int64_t)page_size;
        } else {
            *out = 0;
        }
        return XPU_RT_STATUS_OK;
    }

    case XPU_RT_DEVICE_INFO_COMPUTE_UNITS: {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        *out = (ncpu > 0) ? (int64_t)ncpu : 1;
        return XPU_RT_STATUS_OK;
    }

    case XPU_RT_DEVICE_INFO_MAX_DISPATCH_CONCURRENCY:
        *out = 1; /* single-threaded reference driver */
        return XPU_RT_STATUS_OK;

    case XPU_RT_DEVICE_INFO_ADDRESS_SPACE_COUNT:
        *out = 1; /* host DRAM only */
        return XPU_RT_STATUS_OK;

    case XPU_RT_DEVICE_INFO_SUPPORTS_ASYNC_DMA:
        *out = 0; /* no async DMA on CPU */
        return XPU_RT_STATUS_OK;

    case XPU_RT_DEVICE_INFO_MAX_ALLOC_SIZE:
        *out = (int64_t)((size_t)-1 >> 1); /* SIZE_MAX / 2 as a safe i64 */
        return XPU_RT_STATUS_OK;

    default:
        return XPU_RT_STATUS_NOT_FOUND;
    }
}
