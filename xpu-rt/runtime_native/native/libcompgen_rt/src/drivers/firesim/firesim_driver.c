/*
 * libcompgen_rt — FireSim driver scaffold.
 *
 * RTL co-simulation bridge. The intended implementation talks to a
 * FireSim simulator over the FireSim-published bridge protocol
 * (MMIO + DMA pattern; see chipyard/firesim docs for the wire-level
 * spec). This file ships the ABI scaffolding; the full implementation is
 * a research-tier follow-up that requires the FireSim toolchain
 * (chipyard + verilator/vcs + the FireSim runtime libs).
 *
 * Build gate: ``CG_RT_WITH_FIRESIM``. Otherwise every vtable entry
 * returns CG_RT_ERR_UNSUPPORTED.
 *
 * Realness: declared at realness_level =
 * schema_only.  No functional CI path is plausible without
 * non-trivial chipyard infrastructure; the scaffold is in the tree
 * so future hardware bring-up plugs into the existing vtable
 * registry without changing the public ABI.
 */

#include "../../core/internal.h"

#include <stdlib.h>
#include <string.h>

static cg_rt_status_t firesim_device_open(cg_rt_instance_t *instance,
                                          uint32_t          device_index,
                                          cg_rt_device_t  **out_device);
static void firesim_device_close(cg_rt_device_t *device);
static cg_rt_status_t firesim_queue_submit(cg_rt_device_t               *device,
                                           uint32_t                      queue_index,
                                           const cg_rt_semaphore_point_t *wait,
                                           size_t                        n_wait,
                                           const cg_rt_semaphore_point_t *signal,
                                           size_t                        n_signal,
                                           cg_rt_command_buffer_t       *command_buffer);
static uint32_t firesim_query_device_count(void);

const cg_rt_driver_vtable_t cg_rt_firesim_vtable = {
    .name               = "firesim",
    .device_open        = firesim_device_open,
    .device_close       = firesim_device_close,
    .query_device_count = firesim_query_device_count,
    .queue_submit       = firesim_queue_submit,
};

static uint32_t firesim_query_device_count(void) {
    return 0;
}

static cg_rt_status_t firesim_device_open(cg_rt_instance_t *instance,
                                          uint32_t          device_index,
                                          cg_rt_device_t  **out_device) {
    (void)instance;
    (void)device_index;
    (void)out_device;
    return CG_RT_ERR_UNSUPPORTED;
}

static void firesim_device_close(cg_rt_device_t *device) {
    (void)device;
}

static cg_rt_status_t firesim_queue_submit(cg_rt_device_t               *device,
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
