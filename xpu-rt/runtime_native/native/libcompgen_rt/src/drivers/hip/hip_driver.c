/*
 * libcompgen_rt — HIP driver scaffold.
 *
 * Mirrors src/drivers/cuda/cuda_driver.c for AMD ROCm/HIP. The full
 * implementation tracks the cuda driver feature-for-feature:
 *
 *   - hipDeviceGet + hipCtxCreate → ``device_open``
 *   - hipMemAllocManaged          → buffer backing
 *   - hipModuleLoadData (HIP RTC) → executable_create_hip
 *   - hipStreamCreate / queue     → queue_submit
 *   - hipEvent / hipStreamWait    → semaphore points
 *
 * Build gate: compile only when ``CG_RT_WITH_HIP`` is defined AND the
 * HIP toolkit headers are on the include path. On builds without HIP
 * the vtable still compiles but every entry returns
 * ``CG_RT_ERR_UNSUPPORTED`` so callers see a typed unsupported
 * response rather than a link failure.
 *
 * Realness honesty: the realness contract declares this driver at
 * realness_level=read_only. Functional verification requires AMD
 * silicon; CI machines do not have it. ``cg_rt_*`` callers that
 * select "hip" on a non-HIP build receive CG_RT_ERR_NOT_FOUND from
 * cg_rt_instance_create. Callers on a HIP-capable host (HIP headers
 * present + at least one HIP device) get a fully functional driver
 * once the HIP backend is wired in; this file ships the ABI scaffolding.
 */

#include "../../core/internal.h"

#include <stdlib.h>
#include <string.h>

#ifdef CG_RT_WITH_HIP
/* When real HIP is present, include the headers here. This file leaves the
 * actual HIP includes as a TODO marker; the scaffold's
 * job is to register the vtable and route unsupported calls cleanly. */
/* #include <hip/hip_runtime.h> */
#endif

static cg_rt_status_t hip_device_open(cg_rt_instance_t *instance,
                                      uint32_t          device_index,
                                      cg_rt_device_t  **out_device);
static void hip_device_close(cg_rt_device_t *device);
static cg_rt_status_t hip_queue_submit(cg_rt_device_t               *device,
                                       uint32_t                      queue_index,
                                       const cg_rt_semaphore_point_t *wait,
                                       size_t                        n_wait,
                                       const cg_rt_semaphore_point_t *signal,
                                       size_t                        n_signal,
                                       cg_rt_command_buffer_t       *command_buffer);
static uint32_t hip_query_device_count(void);

const cg_rt_driver_vtable_t cg_rt_hip_vtable = {
    .name               = "hip",
    .device_open        = hip_device_open,
    .device_close       = hip_device_close,
    .query_device_count = hip_query_device_count,
    .queue_submit       = hip_queue_submit,
};

static uint32_t hip_query_device_count(void) {
#ifdef CG_RT_WITH_HIP
    /* int count = 0; hipGetDeviceCount(&count); return (uint32_t)count; */
    return 0; /* TODO: wire the real probe. */
#else
    return 0;
#endif
}

static cg_rt_status_t hip_device_open(cg_rt_instance_t *instance,
                                      uint32_t          device_index,
                                      cg_rt_device_t  **out_device) {
    (void)instance;
    (void)device_index;
    (void)out_device;
    return CG_RT_ERR_UNSUPPORTED;
}

static void hip_device_close(cg_rt_device_t *device) {
    (void)device;
}

static cg_rt_status_t hip_queue_submit(cg_rt_device_t               *device,
                                       uint32_t                      queue_index,
                                       const cg_rt_semaphore_point_t *wait,
                                       size_t                        n_wait,
                                       const cg_rt_semaphore_point_t *signal,
                                       size_t                        n_signal,
                                       cg_rt_command_buffer_t       *command_buffer) {
    (void)device;
    (void)queue_index;
    (void)wait;
    (void)n_wait;
    (void)signal;
    (void)n_signal;
    (void)command_buffer;
    return CG_RT_ERR_UNSUPPORTED;
}
