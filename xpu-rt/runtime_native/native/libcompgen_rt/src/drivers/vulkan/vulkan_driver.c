/*
 * libcompgen_rt — Vulkan driver scaffold.
 *
 * SPIR-V compute-shader path. Real implementation will use:
 *
 *   - vkCreateInstance + vkEnumeratePhysicalDevices → ``device_open``
 *   - vkAllocateMemory + VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT for
 *     unified buffer backing.
 *   - vkCreateShaderModule on SPIR-V binaries for executables.
 *   - vkCreateTimelineSemaphore for libcompgen_rt timeline semaphore
 *     ↔ Vulkan timeline binding (§12 Dream 3 — the event tensor's
 *     canonical sync abstraction maps directly).
 *   - vkCmdDispatch + vkQueueSubmit + vkSemaphoreSignalKHR for the
 *     command-buffer / queue surface.
 *
 * Build gate: ``CG_RT_WITH_VULKAN``. On non-Vulkan builds every
 * vtable entry returns CG_RT_ERR_UNSUPPORTED.
 *
 * Realness: declared at realness_level =
 * read_only. The intended functional CI uses lavapipe (software
 * Vulkan); hardware_backed verification requires a real Vulkan
 * device.
 */

#include "../../core/internal.h"

#include <stdlib.h>
#include <string.h>

#ifdef CG_RT_WITH_VULKAN
/* #include <vulkan/vulkan.h> */
#endif

static cg_rt_status_t vulkan_device_open(cg_rt_instance_t *instance,
                                         uint32_t          device_index,
                                         cg_rt_device_t  **out_device);
static void vulkan_device_close(cg_rt_device_t *device);
static cg_rt_status_t vulkan_queue_submit(cg_rt_device_t               *device,
                                          uint32_t                      queue_index,
                                          const cg_rt_semaphore_point_t *wait,
                                          size_t                        n_wait,
                                          const cg_rt_semaphore_point_t *signal,
                                          size_t                        n_signal,
                                          cg_rt_command_buffer_t       *command_buffer);
static uint32_t vulkan_query_device_count(void);

const cg_rt_driver_vtable_t cg_rt_vulkan_vtable = {
    .name               = "vulkan",
    .device_open        = vulkan_device_open,
    .device_close       = vulkan_device_close,
    .query_device_count = vulkan_query_device_count,
    .queue_submit       = vulkan_queue_submit,
};

static uint32_t vulkan_query_device_count(void) {
    return 0;
}

static cg_rt_status_t vulkan_device_open(cg_rt_instance_t *instance,
                                         uint32_t          device_index,
                                         cg_rt_device_t  **out_device) {
    (void)instance;
    (void)device_index;
    (void)out_device;
    return CG_RT_ERR_UNSUPPORTED;
}

static void vulkan_device_close(cg_rt_device_t *device) {
    (void)device;
}

static cg_rt_status_t vulkan_queue_submit(cg_rt_device_t               *device,
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
