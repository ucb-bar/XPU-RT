/*
 * CUDA driver — backs the libcompgen_rt HAL with NVIDIA GPU execution.
 *
 * Design notes:
 *   - Uses the CUDA Driver API (cuInit / cuDevicePrimaryCtxRetain /
 *     cuStreamCreate / cuMemAllocManaged) rather than the Runtime API.
 *     This avoids conflicting with a host process that has already
 *     initialised cudart (e.g. PyTorch) and gives us finer control
 *     over contexts and streams.
 *   - Buffers use ``cuMemAllocManaged`` so the same pointer is valid
 *     on host and device. That lets ``cg_rt_buffer_map`` return the
 *     direct pointer without a staging copy — mirrors cpu_sync's
 *     behaviour and keeps the public API uniform.
 *   - Command-buffer execution is translated into CUDA stream
 *     operations: copies → cuMemcpyAsync, fills → cuMemsetD32Async,
 *     dispatch → cuLaunchKernel. Queue submit is synchronous in this
 *     first cut (waits the host semaphores, replays on a stream,
 *     cuStreamSynchronize, signals). This is the "CUDA-as-cpu_sync"
 *     tier; a follow-up adds async cuEvent-backed semaphores.
 *   - Executables are NVRTC-compiled CUDA C sources loaded through
 *     ``cuModuleLoadData`` with a kernel entry name. The caller owns
 *     the source string; the driver retains the compiled module.
 *
 * Built only when ``CG_RT_WITH_CUDA=ON``.  When the CUDA libraries
 * cannot be found at CMake configure time the driver is omitted and
 * ``cg_rt_instance_create("cuda", ...)`` returns CG_RT_ERR_NOT_FOUND.
 */

#include "../../core/internal.h"

#include <cuda.h>
#include <nvrtc.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Error helpers                                                        */
/* ------------------------------------------------------------------ */

#define CU_CHECK(expr)                                                       \
    do {                                                                     \
        CUresult _r = (expr);                                                \
        if (_r != CUDA_SUCCESS) {                                            \
            const char *_name = NULL;                                        \
            cuGetErrorName(_r, &_name);                                      \
            fprintf(stderr, "cuda: %s failed: %s\n", #expr,                  \
                    _name ? _name : "?");                                    \
            return CG_RT_ERR_UNKNOWN;                                        \
        }                                                                    \
    } while (0)

#define NVRTC_CHECK(expr)                                                    \
    do {                                                                     \
        nvrtcResult _r = (expr);                                             \
        if (_r != NVRTC_SUCCESS) {                                           \
            fprintf(stderr, "nvrtc: %s failed: %s\n", #expr,                 \
                    nvrtcGetErrorString(_r));                                \
            return CG_RT_ERR_UNKNOWN;                                        \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ */
/* Driver-private structs                                               */
/* ------------------------------------------------------------------ */

#define CUDA_NUM_QUEUES 4

typedef struct {
    CUdevice   dev;
    CUcontext  ctx;
    CUstream   streams[CUDA_NUM_QUEUES];
    int        compute_cap_major;
    int        compute_cap_minor;
    size_t     mem_total_bytes;
} cuda_state_t;

/* Executable backing for CUDA: a loaded module + a function handle.
 * We stash this inside the ``entry_point`` of the generic executable
 * by storing a pointer to a small adaptor struct; when cpu_sync/task
 * look at entry_point they never execute it because the CUDA driver
 * uses its own dispatch path. */
typedef struct {
    CUmodule   module;
    CUfunction func;
} cuda_executable_impl_t;

/* ------------------------------------------------------------------ */
/* Forward declarations                                                 */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cuda_device_open(cg_rt_instance_t *instance,
                                       uint32_t          device_index,
                                       cg_rt_device_t  **out_device);
static void cuda_device_close(cg_rt_device_t *device);

static uint32_t cuda_query_device_count(void) {
    /* cuInit is idempotent. Returns 0 on any failure so the public
     * bounds check rejects every index — caller will get NOT_FOUND
     * cleanly instead of a stale stale count. */
    if (cuInit(0) != CUDA_SUCCESS) return 0;
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count < 0) return 0;
    return (uint32_t)count;
}

static cg_rt_status_t cuda_data_alloc(cg_rt_device_t       *device,
                                      size_t                size_bytes,
                                      cg_rt_memory_space_t  memory_space,
                                      void                **out_ptr);
static void cuda_data_free(cg_rt_device_t *device, void *ptr);

static cg_rt_status_t cuda_queue_submit(cg_rt_device_t               *device,
                                        uint32_t                      queue_index,
                                        const cg_rt_semaphore_point_t *wait,
                                        size_t                        n_wait,
                                        const cg_rt_semaphore_point_t *signal,
                                        size_t                        n_signal,
                                        cg_rt_command_buffer_t       *command_buffer);

const cg_rt_driver_vtable_t cg_rt_cuda_vtable = {
    .name               = "cuda",
    .device_open        = cuda_device_open,
    .device_close       = cuda_device_close,
    .query_device_count = cuda_query_device_count,
    .queue_submit       = cuda_queue_submit,
    .data_alloc   = cuda_data_alloc,
    .data_free    = cuda_data_free,
};

/* ------------------------------------------------------------------ */
/* Device lifecycle                                                     */
/* ------------------------------------------------------------------ */

static void fill_cuda_traits(cg_rt_device_traits_t *t,
                             const cuda_state_t    *state,
                             const char            *device_name) {
    memset(t, 0, sizeof(*t));
    t->device_class = CG_RT_DEVICE_CLASS_GPU;
    strncpy(t->vendor, "nvidia", sizeof(t->vendor) - 1);
    strncpy(t->name, device_name, sizeof(t->name) - 1);
    /* CUDA events give us native (GPU-side) timeline semantics. Phase
     * C.1 still uses the host timeline semaphore for correctness —
     * this trait signals the capability is there for drivers that
     * want to emit GPU-side waits. */
    t->has_native_timeline_semaphores = 1;
    t->has_global_atomics             = 1;
    t->has_shared_memory_atomics      = 1;
    /* Persistent kernels require sm_70+ for grid-wide synchronisation
     * via cooperative launch; we gate on compute cap >= 6 here which
     * covers TITAN RTX (sm_75) and newer. */
    t->supports_persistent_kernels    = (state->compute_cap_major >= 6) ? 1 : 0;
    t->supports_cooperative_launch    = (state->compute_cap_major >= 6) ? 1 : 0;
    t->supports_command_buffers       = 1;
    t->supports_graph_capture         = 1;  /* cuGraph is available on 10+ */
    t->supports_event_tensors         = t->has_global_atomics &&
                                        t->supports_persistent_kernels;
    t->is_bare_metal                  = 0;
    t->has_rtos_support               = 0;
    t->max_device_memory_bytes        = state->mem_total_bytes;
    t->supports_host_pinned           = 1;
    t->supports_peer_access           = 0;
    t->max_concurrent_queues          = CUDA_NUM_QUEUES;
    /* Max threads per block — CUDA hardware limit, 1024 on every
     * compute cap we target. */
    t->max_workgroup_size             = 1024;
}

static cg_rt_status_t cuda_device_open(cg_rt_instance_t *instance,
                                       uint32_t          device_index,
                                       cg_rt_device_t  **out_device) {
    (void)instance;
    CU_CHECK(cuInit(0));

    int count = 0;
    CU_CHECK(cuDeviceGetCount(&count));
    if ((int)device_index >= count) return CG_RT_ERR_NOT_FOUND;

    cuda_state_t *state = calloc(1, sizeof(*state));
    if (state == NULL) return CG_RT_ERR_OUT_OF_MEMORY;

    CU_CHECK(cuDeviceGet(&state->dev, (int)device_index));
    CU_CHECK(cuDevicePrimaryCtxRetain(&state->ctx, state->dev));
    CU_CHECK(cuCtxPushCurrent(state->ctx));

    /* Query compute capability + total memory for traits. */
    CU_CHECK(cuDeviceGetAttribute(&state->compute_cap_major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  state->dev));
    CU_CHECK(cuDeviceGetAttribute(&state->compute_cap_minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  state->dev));
    CU_CHECK(cuDeviceTotalMem(&state->mem_total_bytes, state->dev));

    /* Create a stream per logical queue. Streams are lightweight;
     * we pay for them up front so queue_submit has nothing to lazy-
     * init on the hot path. */
    for (int i = 0; i < CUDA_NUM_QUEUES; ++i) {
        CU_CHECK(cuStreamCreate(&state->streams[i], CU_STREAM_NON_BLOCKING));
    }

    char device_name[64] = {0};
    cuDeviceGetName(device_name, sizeof(device_name) - 1, state->dev);

    cg_rt_device_t *dev = calloc(1, sizeof(*dev));
    if (dev == NULL) {
        for (int i = 0; i < CUDA_NUM_QUEUES; ++i) cuStreamDestroy(state->streams[i]);
        cuDevicePrimaryCtxRelease(state->dev);
        free(state);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    dev->vtable = &cg_rt_cuda_vtable;
    dev->device_index = device_index;
    dev->driver_state = state;
    dev->num_queues = CUDA_NUM_QUEUES;
    fill_cuda_traits(&dev->traits, state, device_name);

    /* Keep ctx current for the life of the device. Callers running
     * libcompgen_rt on multiple CUDA devices need to push/pop
     * themselves; a future refactor will store per-thread state. */
    CU_CHECK(cuCtxPopCurrent(NULL));

    *out_device = dev;
    return CG_RT_OK;
}

static void cuda_device_close(cg_rt_device_t *device) {
    if (device == NULL) return;
    cuda_state_t *state = device->driver_state;
    if (state != NULL) {
        cuCtxPushCurrent(state->ctx);
        for (int i = 0; i < CUDA_NUM_QUEUES; ++i) {
            cuStreamDestroy(state->streams[i]);
        }
        cuCtxPopCurrent(NULL);
        cuDevicePrimaryCtxRelease(state->dev);
        free(state);
    }
    free(device);
}

/* ------------------------------------------------------------------ */
/* Buffers (managed memory — host-pointer-accessible)                   */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cuda_data_alloc(cg_rt_device_t       *device,
                                      size_t                size_bytes,
                                      cg_rt_memory_space_t  memory_space,
                                      void                **out_ptr) {
    (void)memory_space; /* all backed by managed memory for now */
    cuda_state_t *state = device->driver_state;
    CU_CHECK(cuCtxPushCurrent(state->ctx));

    CUdeviceptr dptr = 0;
    CUresult rc = cuMemAllocManaged(&dptr, size_bytes, CU_MEM_ATTACH_GLOBAL);
    if (rc != CUDA_SUCCESS) {
        cuCtxPopCurrent(NULL);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    /* Zero-init for deterministic tests. Managed memory is accessible
     * by the host so memset is legal without staging. */
    memset((void *)(uintptr_t)dptr, 0, size_bytes);
    cuCtxPopCurrent(NULL);
    *out_ptr = (void *)(uintptr_t)dptr;
    return CG_RT_OK;
}

static void cuda_data_free(cg_rt_device_t *device, void *ptr) {
    if (ptr == NULL) return;
    cuda_state_t *state = device->driver_state;
    cuCtxPushCurrent(state->ctx);
    cuMemFree((CUdeviceptr)(uintptr_t)ptr);
    cuCtxPopCurrent(NULL);
}

/* ------------------------------------------------------------------ */
/* Command-buffer execution on a CUDA stream                            */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cuda_execute_copy(CUstream stream, const cg_rt_command_t *cmd) {
    CUdeviceptr dst_dptr = (CUdeviceptr)(uintptr_t)cmd->copy.dst->data + cmd->copy.dst_offset;
    CUdeviceptr src_dptr = (CUdeviceptr)(uintptr_t)cmd->copy.src->data + cmd->copy.src_offset;
    CU_CHECK(cuMemcpyAsync(dst_dptr, src_dptr, cmd->copy.size_bytes, stream));
    return CG_RT_OK;
}

static cg_rt_status_t cuda_execute_fill(CUstream stream, const cg_rt_command_t *cmd) {
    CUdeviceptr dst_dptr = (CUdeviceptr)(uintptr_t)cmd->fill.dst->data + cmd->fill.dst_offset;
    /* cuMemsetD32Async expects a count in uint32 elements when used
     * with a 32-bit pattern. We prefer the element-wise setter when
     * size is a multiple of 4 (exact pattern match) and fall back to
     * a byte-wise memset + host-side initialisation otherwise. */
    if ((cmd->fill.size_bytes % 4) == 0) {
        CU_CHECK(cuMemsetD32Async(dst_dptr, cmd->fill.pattern,
                                  cmd->fill.size_bytes / 4, stream));
    } else {
        /* Rare path. Stage the pattern on host and cuMemcpyAsync.
         * We don't bother optimising this — no compiler emits odd
         * fills in practice. */
        const uint8_t *pb = (const uint8_t *)&cmd->fill.pattern;
        uint8_t *host_buf = malloc(cmd->fill.size_bytes);
        if (host_buf == NULL) return CG_RT_ERR_OUT_OF_MEMORY;
        for (size_t i = 0; i < cmd->fill.size_bytes; ++i) {
            host_buf[i] = pb[i & 3];
        }
        CUresult rc = cuMemcpyHtoDAsync(dst_dptr, host_buf, cmd->fill.size_bytes, stream);
        /* Force the async copy to finish before we free the host
         * buffer — the command buffer is being replayed on a single
         * stream anyway so stream-sync here costs nothing we weren't
         * going to pay at the end of queue_submit. */
        cuStreamSynchronize(stream);
        free(host_buf);
        if (rc != CUDA_SUCCESS) return CG_RT_ERR_UNKNOWN;
    }
    return CG_RT_OK;
}

static cg_rt_status_t cuda_execute_dispatch(CUstream stream, const cg_rt_command_t *cmd) {
    cg_rt_executable_t *exe = cmd->dispatch.executable;
    if (exe == NULL || exe->driver_impl == NULL) return CG_RT_ERR_FAILED_PRECOND;
    cuda_executable_impl_t *impl = exe->driver_impl;

    /* Build the kernel argument vector. For CUDA driver API,
     * ``kernelParams`` is an array of pointers to kernel arguments.
     * The dispatch contract is:
     *   - push_constants[0..4) : uint32 grid_x
     *   - push_constants[4..8) : uint32 grid_y  (default 1)
     *   - push_constants[8..12): uint32 grid_z  (default 1)
     *   - push_constants[12..16): uint32 block_x
     *   - push_constants[16..20): uint32 block_y (default 1)
     *   - push_constants[20..24): uint32 block_z (default 1)
     *   - remaining bytes: the user push constants (unused here —
     *     reserved for future kernel scalars).
     * Buffer bindings are passed as device-pointer scalars in the
     * conventional order. Callers must build the push-constant block
     * to match. This mirrors how IREE's cuda backend shapes the
     * launch descriptor from the dispatch's bindings + pc.
     */
    if (cmd->dispatch.pc_size < 24) return CG_RT_ERR_INVALID_ARGUMENT;
    const uint32_t *cfg = (const uint32_t *)cmd->dispatch.push_constants;
    unsigned int grid_x = cfg[0], grid_y = cfg[1], grid_z = cfg[2];
    unsigned int block_x = cfg[3], block_y = cfg[4], block_z = cfg[5];
    if (grid_y == 0) grid_y = 1;
    if (grid_z == 0) grid_z = 1;
    if (block_y == 0) block_y = 1;
    if (block_z == 0) block_z = 1;

    /* Build argument pointer vector. Each binding becomes a
     * CUdeviceptr passed by value to the kernel. */
    size_t n = cmd->dispatch.n_bindings;
    CUdeviceptr *dptrs = NULL;
    void **kernel_args = NULL;
    if (n > 0) {
        dptrs = malloc(n * sizeof(*dptrs));
        kernel_args = malloc(n * sizeof(*kernel_args));
        if (dptrs == NULL || kernel_args == NULL) {
            free(dptrs); free(kernel_args);
            return CG_RT_ERR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < n; ++i) {
            dptrs[i] = (CUdeviceptr)(uintptr_t)cmd->dispatch.bindings[i]->data;
            kernel_args[i] = &dptrs[i];
        }
    }

    CUresult rc = cuLaunchKernel(impl->func,
                                 grid_x, grid_y, grid_z,
                                 block_x, block_y, block_z,
                                 /*sharedMemBytes=*/0,
                                 stream,
                                 kernel_args,
                                 /*extra=*/NULL);
    free(dptrs); free(kernel_args);
    if (rc != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(rc, &name);
        fprintf(stderr, "cuLaunchKernel failed: %s\n", name ? name : "?");
        return CG_RT_ERR_UNKNOWN;
    }
    return CG_RT_OK;
}

static cg_rt_status_t cuda_replay_command_buffer(CUstream stream,
                                                 cg_rt_command_buffer_t *cb) {
    if (cb->state != CG_RT_CB_STATE_EXECUTABLE) return CG_RT_ERR_FAILED_PRECOND;
    for (size_t i = 0; i < cb->num_commands; ++i) {
        cg_rt_status_t rc = CG_RT_OK;
        switch (cb->commands[i].op) {
        case CG_RT_CMD_OP_COPY:     rc = cuda_execute_copy(stream, &cb->commands[i]); break;
        case CG_RT_CMD_OP_FILL:     rc = cuda_execute_fill(stream, &cb->commands[i]); break;
        case CG_RT_CMD_OP_DISPATCH: rc = cuda_execute_dispatch(stream, &cb->commands[i]); break;
        case CG_RT_CMD_OP_BARRIER:
            /* Barrier: a stream-scoped wait. cuStreamSynchronize here
             * is overkill (it waits for all prior work) but matches
             * the intent — subsequent ops see all prior effects.
             * A later refinement can emit a lighter stream event. */
            cuStreamSynchronize(stream);
            break;
        default: rc = CG_RT_ERR_UNSUPPORTED; break;
        }
        if (rc != CG_RT_OK) return rc;
    }
    return CG_RT_OK;
}

/* ------------------------------------------------------------------ */
/* Queue submit                                                         */
/* ------------------------------------------------------------------ */

static cg_rt_status_t cuda_queue_submit(cg_rt_device_t               *device,
                                        uint32_t                      queue_index,
                                        const cg_rt_semaphore_point_t *wait,
                                        size_t                        n_wait,
                                        const cg_rt_semaphore_point_t *signal,
                                        size_t                        n_signal,
                                        cg_rt_command_buffer_t       *command_buffer) {
    if (queue_index >= device->num_queues) return CG_RT_ERR_INVALID_ARGUMENT;
    if ((wait == NULL) != (n_wait == 0)) return CG_RT_ERR_INVALID_ARGUMENT;
    if ((signal == NULL) != (n_signal == 0)) return CG_RT_ERR_INVALID_ARGUMENT;

    cuda_state_t *state = device->driver_state;
    CUstream stream = state->streams[queue_index];
    CU_CHECK(cuCtxPushCurrent(state->ctx));

    /* Host-side wait phase. */
    for (size_t i = 0; i < n_wait; ++i) {
        cg_rt_status_t rc = cg_rt_semaphore_wait(wait[i].semaphore,
                                                 wait[i].value,
                                                 CG_RT_TIMEOUT_INFINITE);
        if (rc != CG_RT_OK) {
            for (size_t j = 0; j < n_signal; ++j) {
                cg_rt_semaphore_fail(signal[j].semaphore, rc);
            }
            cuCtxPopCurrent(NULL);
            return rc;
        }
    }

    /* Replay onto the stream then synchronise. Phase C.2 will record
     * a cuEvent here and detach the host signal from the stream so
     * queue_submit can return asynchronously. */
    cg_rt_status_t exec_rc = cuda_replay_command_buffer(stream, command_buffer);
    if (exec_rc == CG_RT_OK) {
        CUresult sr = cuStreamSynchronize(stream);
        if (sr != CUDA_SUCCESS) exec_rc = CG_RT_ERR_UNKNOWN;
    }
    if (exec_rc != CG_RT_OK) {
        for (size_t j = 0; j < n_signal; ++j) {
            cg_rt_semaphore_fail(signal[j].semaphore, exec_rc);
        }
        cuCtxPopCurrent(NULL);
        return exec_rc;
    }

    for (size_t i = 0; i < n_signal; ++i) {
        cg_rt_semaphore_signal(signal[i].semaphore, signal[i].value);
    }
    cuCtxPopCurrent(NULL);
    return CG_RT_OK;
}

/* ------------------------------------------------------------------ */
/* NVRTC — compile CUDA C source into a CUfunction handle               */
/* ------------------------------------------------------------------ */

/* Public factory used by the Python binding to turn a CUDA C source
 * into a launchable executable. Lives in the CUDA driver's public
 * surface, separate from the CPU executable factory. */
cg_rt_status_t cg_rt_executable_create_cuda_ptx(cg_rt_device_t      *device,
                                                const char          *cuda_c_source,
                                                const char          *kernel_name,
                                                cg_rt_executable_t **out_executable) {
    if (device == NULL || cuda_c_source == NULL ||
        kernel_name == NULL || out_executable == NULL) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    if (device->vtable != &cg_rt_cuda_vtable) {
        return CG_RT_ERR_INVALID_ARGUMENT;
    }
    cuda_state_t *state = device->driver_state;

    /* NVRTC compile. */
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog,
                                   cuda_c_source,
                                   "compgen_rt_kernel.cu",
                                   /*numHeaders=*/0,
                                   /*headers=*/NULL,
                                   /*includeNames=*/NULL));

    /* Compile for the device's actual compute capability. */
    char arch_flag[32];
    snprintf(arch_flag, sizeof(arch_flag), "-arch=compute_%d%d",
             state->compute_cap_major, state->compute_cap_minor);
    const char *opts[] = {arch_flag};
    nvrtcResult compile_rc = nvrtcCompileProgram(prog, 1, opts);
    if (compile_rc != NVRTC_SUCCESS) {
        /* Surface the build log so kernel errors are debuggable. */
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        char *log = malloc(log_size + 1);
        if (log != NULL) {
            nvrtcGetProgramLog(prog, log);
            log[log_size] = '\0';
            fprintf(stderr, "nvrtc compile failed:\n%s\n", log);
            free(log);
        }
        nvrtcDestroyProgram(&prog);
        return CG_RT_ERR_FAILED_PRECOND;
    }

    size_t ptx_size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    char *ptx = malloc(ptx_size);
    if (ptx == NULL) {
        nvrtcDestroyProgram(&prog);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx));
    nvrtcDestroyProgram(&prog);

    CU_CHECK(cuCtxPushCurrent(state->ctx));

    cuda_executable_impl_t *impl = calloc(1, sizeof(*impl));
    if (impl == NULL) {
        free(ptx);
        cuCtxPopCurrent(NULL);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }

    CUresult cr = cuModuleLoadData(&impl->module, ptx);
    free(ptx);
    if (cr != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(cr, &name);
        fprintf(stderr, "cuModuleLoadData failed: %s (ctx=%p)\n",
                name ? name : "?", (void *)state->ctx);
        free(impl);
        cuCtxPopCurrent(NULL);
        return CG_RT_ERR_UNKNOWN;
    }
    cr = cuModuleGetFunction(&impl->func, impl->module, kernel_name);
    if (cr != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(cr, &name);
        fprintf(stderr, "cuModuleGetFunction(%s) failed: %s\n",
                kernel_name, name ? name : "?");
        cuModuleUnload(impl->module);
        free(impl);
        cuCtxPopCurrent(NULL);
        return CG_RT_ERR_NOT_FOUND;
    }
    cuCtxPopCurrent(NULL);

    /* Stash the impl pointer in entry_point. Safe because CUDA
     * dispatch only runs through cuda_execute_dispatch, which reads
     * the pointer back as the real type. */
    cg_rt_executable_t *exe = calloc(1, sizeof(*exe));
    if (exe == NULL) {
        cuCtxPushCurrent(state->ctx);
        cuModuleUnload(impl->module);
        cuCtxPopCurrent(NULL);
        free(impl);
        return CG_RT_ERR_OUT_OF_MEMORY;
    }
    exe->driver_impl = impl;
    /* Hook the generic destroy so cuModuleUnload runs before the
     * executable handle itself is freed. */
    extern void cg_rt_cuda_executable_destroy_impl(cg_rt_executable_t *);
    exe->destroy_impl = cg_rt_cuda_executable_destroy_impl;
    *out_executable = exe;
    return CG_RT_OK;
}

void cg_rt_cuda_executable_destroy_impl(cg_rt_executable_t *executable) {
    if (executable == NULL) return;
    cuda_executable_impl_t *impl = executable->driver_impl;
    if (impl != NULL) {
        /* We don't own a context reference here — the caller must
         * ensure the device is still alive. This mirrors how Vulkan
         * shader modules are destroyed before the VkDevice. */
        cuModuleUnload(impl->module);
        free(impl);
    }
    free(executable);
}
