/*
 * Executable — CPU entry-point wrapper.
 *
 * For cpu_sync + cpu_task the executable is simply a function
 * pointer. CUDA / HIP drivers will extend this with an opaque module
 * + export ordinal. The abstraction stays the same: dispatch gets an
 * executable handle and looks up the implementation.
 */

#include "internal.h"

#include <stdlib.h>

cg_rt_status_t cg_rt_executable_create_cpu(cg_rt_device_t      *device,
                                           cg_rt_cpu_kernel_fn  entry_point,
                                           cg_rt_executable_t **out_executable) {
    if (entry_point == NULL || out_executable == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cg_rt_executable_t *exe = calloc(1, sizeof(*exe));
    if (exe == NULL) {
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    exe->entry_point = entry_point;
    (void)device;
    *out_executable = exe;
    return CG_RT_OK;
}

void cg_rt_executable_destroy(cg_rt_executable_t *executable) {
    if (executable == NULL) return;
    if (executable->destroy_impl != NULL) {
        /* Driver-specific cleanup (e.g. cuModuleUnload) then free. */
        executable->destroy_impl(executable);
        return;
    }
    free(executable);
}
