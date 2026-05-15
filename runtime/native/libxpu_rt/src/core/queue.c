/*
 * Public ``cg_rt_queue_submit`` trampoline — dispatches to the
 * device's driver vtable.
 */

#include "internal.h"

cg_rt_status_t cg_rt_queue_submit(cg_rt_device_t               *device,
                                  uint32_t                      queue_index,
                                  const cg_rt_semaphore_point_t *wait,
                                  size_t                        n_wait,
                                  const cg_rt_semaphore_point_t *signal,
                                  size_t                        n_signal,
                                  cg_rt_command_buffer_t       *command_buffer) {
    if (device == NULL || device->vtable == NULL ||
        device->vtable->queue_submit == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    return device->vtable->queue_submit(device, queue_index,
                                        wait, n_wait,
                                        signal, n_signal,
                                        command_buffer);
}
