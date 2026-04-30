/*
 * libcompgen_rt — CompGen native HAL.
 *
 * Portable C11 runtime layer for multi-target deployment (CPU, GPU,
 * accelerators, bare-metal). The API surface mirrors IREE's HAL in
 * structure: a driver/device/queue hierarchy with timeline semaphores
 * and recorded command buffers.  Event tensors and persistent-launch
 * support the event-tensor megakernel abstraction (Jin et al., MLSys
 * '26).
 *
 * Thread safety:
 *   - Handles are opaque. Unless otherwise noted, a single handle must
 *     not be touched from multiple threads concurrently. Semaphores
 *     and event tensors are the exception: signal/wait/query are safe
 *     to call from any thread.
 *   - Creation and destruction of any handle must be serialised by
 *     the caller.
 *
 * Error model:
 *   - Every fallible call returns ``cg_rt_status_t``. Zero is success;
 *     negative codes are errors. ``cg_rt_status_string`` decodes codes.
 *   - Timeline semaphores carry a separate failure channel
 *     (``cg_rt_semaphore_fail``) that propagates to all waiters and
 *     downstream dependents.
 */

#ifndef COMPGEN_RT_H_
#define COMPGEN_RT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Versioning                                                          */
/* ------------------------------------------------------------------ */

#define CG_RT_VERSION_MAJOR 0
#define CG_RT_VERSION_MINOR 1
#define CG_RT_VERSION_PATCH 0

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef int32_t cg_rt_status_t;

#define CG_RT_OK                     0
#define CG_RT_ERR_INVALID_ARGUMENT  -1
#define CG_RT_ERR_OUT_OF_MEMORY     -2
#define CG_RT_ERR_UNSUPPORTED       -3
#define CG_RT_ERR_NOT_FOUND         -4
#define CG_RT_ERR_TIMED_OUT         -5
#define CG_RT_ERR_FAILED_PRECOND    -6
#define CG_RT_ERR_ABORTED           -7
#define CG_RT_ERR_UNKNOWN          -99

const char *cg_rt_status_string(cg_rt_status_t status);

/* Timeout sentinels (nanoseconds). */
#define CG_RT_TIMEOUT_POLL      UINT64_C(0)
#define CG_RT_TIMEOUT_INFINITE  UINT64_MAX

/* ------------------------------------------------------------------ */
/* Opaque handle types                                                 */
/* ------------------------------------------------------------------ */

typedef struct cg_rt_instance       cg_rt_instance_t;
typedef struct cg_rt_device         cg_rt_device_t;
typedef struct cg_rt_buffer         cg_rt_buffer_t;
typedef struct cg_rt_semaphore      cg_rt_semaphore_t;
typedef struct cg_rt_command_buffer cg_rt_command_buffer_t;
typedef struct cg_rt_executable     cg_rt_executable_t;
typedef struct cg_rt_event_tensor   cg_rt_event_tensor_t;

/* ------------------------------------------------------------------ */
/* Device traits                                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    CG_RT_DEVICE_CLASS_CPU     = 1,
    CG_RT_DEVICE_CLASS_GPU     = 2,
    CG_RT_DEVICE_CLASS_NPU     = 3,
    CG_RT_DEVICE_CLASS_ACCEL   = 4,
} cg_rt_device_class_t;

/*
 * DeviceTraits — capability record queried at device-open time.
 *
 * Consumers prefer trait-based routing over vendor-name branching:
 *
 *     cg_rt_device_traits_t traits;
 *     cg_rt_device_query_traits(dev, &traits);
 *     if (traits.supports_event_tensors) { ... }
 */
typedef struct {
    cg_rt_device_class_t device_class;
    char                 vendor[32];
    char                 name[64];

    /* Core capabilities (1 = supported, 0 = not). */
    uint8_t has_native_timeline_semaphores;
    uint8_t has_global_atomics;
    uint8_t has_shared_memory_atomics;
    uint8_t supports_persistent_kernels;
    uint8_t supports_cooperative_launch;
    uint8_t supports_command_buffers;
    uint8_t supports_graph_capture;
    uint8_t supports_event_tensors;
    uint8_t is_bare_metal;
    uint8_t has_rtos_support;

    /* Memory model. */
    uint64_t max_device_memory_bytes;
    uint8_t  supports_host_pinned;
    uint8_t  supports_peer_access;

    /* Parallelism. */
    uint32_t max_concurrent_queues;
    uint32_t max_workgroup_size;
} cg_rt_device_traits_t;

/* ------------------------------------------------------------------ */
/* Instance + device lifecycle                                         */
/* ------------------------------------------------------------------ */

/*
 * Instance factory.
 *
 * ``driver_name`` selects a registered driver ("cpu_sync", "cpu_task",
 * "cuda", ...). Pass NULL for the default driver (currently
 * "cpu_sync"). Returns CG_RT_ERR_NOT_FOUND when the driver is unknown.
 */
cg_rt_status_t cg_rt_instance_create(const char        *driver_name,
                                     cg_rt_instance_t **out_instance);

void cg_rt_instance_destroy(cg_rt_instance_t *instance);

/*
 * Enumerate available devices. Pass NULL ``out_devices`` with
 * ``*inout_count = 0`` to query the count.
 */
cg_rt_status_t cg_rt_instance_query_devices(cg_rt_instance_t *instance,
                                            cg_rt_device_t  **out_devices,
                                            size_t           *inout_count);

cg_rt_status_t cg_rt_device_open(cg_rt_instance_t *instance,
                                 uint32_t          device_index,
                                 cg_rt_device_t  **out_device);

void cg_rt_device_close(cg_rt_device_t *device);

cg_rt_status_t cg_rt_device_query_traits(cg_rt_device_t        *device,
                                         cg_rt_device_traits_t *out_traits);

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */

typedef enum {
    CG_RT_MEMORY_SPACE_HOST   = 1,   /* host pageable */
    CG_RT_MEMORY_SPACE_DEVICE = 2,   /* device-local */
    CG_RT_MEMORY_SPACE_UNIFIED = 3,  /* unified / shared */
} cg_rt_memory_space_t;

typedef enum {
    CG_RT_BUFFER_USAGE_NONE     = 0,
    CG_RT_BUFFER_USAGE_TRANSFER = 1u << 0,
    CG_RT_BUFFER_USAGE_DISPATCH = 1u << 1,
    CG_RT_BUFFER_USAGE_INDIRECT = 1u << 2,
} cg_rt_buffer_usage_t;

cg_rt_status_t cg_rt_buffer_alloc(cg_rt_device_t       *device,
                                  size_t                size_bytes,
                                  cg_rt_memory_space_t  memory_space,
                                  uint32_t              usage_flags,
                                  cg_rt_buffer_t      **out_buffer);

void cg_rt_buffer_destroy(cg_rt_buffer_t *buffer);

size_t cg_rt_buffer_size(const cg_rt_buffer_t *buffer);

/*
 * Map the buffer for host access. On cpu_sync this returns the direct
 * host pointer (no copy). On discrete-memory drivers this may return a
 * staging pointer backed by an implicit transfer on ``unmap``.
 */
cg_rt_status_t cg_rt_buffer_map(cg_rt_buffer_t  *buffer,
                                size_t           offset,
                                size_t           size,
                                void           **out_ptr);

cg_rt_status_t cg_rt_buffer_unmap(cg_rt_buffer_t *buffer);

/* ------------------------------------------------------------------ */
/* Timeline semaphores                                                 */
/* ------------------------------------------------------------------ */

/*
 * Timeline semaphores carry a monotonically non-decreasing uint64
 * payload. Signalling with a lower value is a no-op; signalling with a
 * strictly greater value is the only way to advance the timeline.
 * Waits are idempotent on completed values. An external "failure" bit
 * is tracked separately; once failed the semaphore cannot be advanced
 * and all waiters return ``CG_RT_ERR_ABORTED``.
 */
cg_rt_status_t cg_rt_semaphore_create(cg_rt_device_t     *device,
                                      uint64_t            initial_value,
                                      cg_rt_semaphore_t **out_semaphore);

void cg_rt_semaphore_destroy(cg_rt_semaphore_t *semaphore);

cg_rt_status_t cg_rt_semaphore_signal(cg_rt_semaphore_t *semaphore,
                                      uint64_t           value);

cg_rt_status_t cg_rt_semaphore_wait(cg_rt_semaphore_t *semaphore,
                                    uint64_t           value,
                                    uint64_t           timeout_ns);

cg_rt_status_t cg_rt_semaphore_query(cg_rt_semaphore_t *semaphore,
                                     uint64_t          *out_value);

void cg_rt_semaphore_fail(cg_rt_semaphore_t *semaphore,
                          cg_rt_status_t     status);

/* ------------------------------------------------------------------ */
/* Executables                                                         */
/* ------------------------------------------------------------------ */

/*
 * CPU executable entry point. The kernel sees:
 *   - ``push_constants``: opaque byte buffer of size ``pc_size``.
 *   - ``bindings``: array of mapped host pointers to each bound
 *     buffer. Size is ``n_bindings``. Caller-owned; the callee may
 *     not free them.
 *   - ``binding_sizes``: parallel array with each binding's byte size
 *     (useful for bounds-checked kernels).
 *
 * Return 0 for success, non-zero to indicate kernel failure (the
 * failure is forwarded to the signal semaphore).
 */
typedef int (*cg_rt_cpu_kernel_fn)(const void *push_constants,
                                   size_t      pc_size,
                                   void       **bindings,
                                   const size_t *binding_sizes,
                                   size_t      n_bindings);

cg_rt_status_t cg_rt_executable_create_cpu(cg_rt_device_t      *device,
                                           cg_rt_cpu_kernel_fn  entry_point,
                                           cg_rt_executable_t **out_executable);

/*
 * CUDA-specific executable factory. Compiles ``cuda_c_source`` through
 * NVRTC for the target device's compute capability and loads the
 * resulting PTX module.  ``kernel_name`` is the unmangled extern "C"
 * entry point inside the source.
 *
 * The launch descriptor built into ``cg_rt_command_buffer_dispatch``'s
 * ``push_constants`` block for a CUDA executable must be at least
 * 24 bytes laid out as six uint32s:
 *     [grid_x, grid_y, grid_z, block_x, block_y, block_z]
 * Any bytes after the first 24 are currently unused and reserved for
 * future kernel scalars.
 *
 * Only available when libcompgen_rt was built with CUDA support. On
 * non-CUDA builds this symbol is absent from the .so; callers should
 * query ``available()`` via the Python binding or ``dlsym`` before
 * invoking.
 */
cg_rt_status_t cg_rt_executable_create_cuda_ptx(cg_rt_device_t      *device,
                                                const char          *cuda_c_source,
                                                const char          *kernel_name,
                                                cg_rt_executable_t **out_executable);

void cg_rt_executable_destroy(cg_rt_executable_t *executable);

/* ------------------------------------------------------------------ */
/* Command buffers                                                     */
/* ------------------------------------------------------------------ */

cg_rt_status_t cg_rt_command_buffer_create(cg_rt_device_t          *device,
                                           cg_rt_command_buffer_t **out_command_buffer);

void cg_rt_command_buffer_destroy(cg_rt_command_buffer_t *command_buffer);

cg_rt_status_t cg_rt_command_buffer_begin(cg_rt_command_buffer_t *command_buffer);

cg_rt_status_t cg_rt_command_buffer_end(cg_rt_command_buffer_t *command_buffer);

cg_rt_status_t cg_rt_command_buffer_copy(cg_rt_command_buffer_t *command_buffer,
                                         cg_rt_buffer_t         *src,
                                         size_t                  src_offset,
                                         cg_rt_buffer_t         *dst,
                                         size_t                  dst_offset,
                                         size_t                  size_bytes);

cg_rt_status_t cg_rt_command_buffer_fill(cg_rt_command_buffer_t *command_buffer,
                                         cg_rt_buffer_t         *dst,
                                         size_t                  dst_offset,
                                         size_t                  size_bytes,
                                         uint32_t                pattern);

cg_rt_status_t cg_rt_command_buffer_dispatch(cg_rt_command_buffer_t *command_buffer,
                                             cg_rt_executable_t     *executable,
                                             const void             *push_constants,
                                             size_t                  pc_size,
                                             cg_rt_buffer_t        **bindings,
                                             size_t                  n_bindings);

cg_rt_status_t cg_rt_command_buffer_barrier(cg_rt_command_buffer_t *command_buffer);

/* ------------------------------------------------------------------ */
/* Queue submission                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    cg_rt_semaphore_t *semaphore;
    uint64_t           value;
} cg_rt_semaphore_point_t;

/*
 * Submit a command buffer on a device queue. Blocks the queue worker
 * until all ``wait`` semaphores reach their target value, then executes
 * the command buffer, then signals all ``signal`` semaphores with their
 * respective values. On cpu_sync the entire call is synchronous — the
 * function returns only after execution completes.
 *
 * ``queue_index`` selects among ``traits.max_concurrent_queues``.
 */
cg_rt_status_t cg_rt_queue_submit(cg_rt_device_t               *device,
                                  uint32_t                      queue_index,
                                  const cg_rt_semaphore_point_t *wait,
                                  size_t                        n_wait,
                                  const cg_rt_semaphore_point_t *signal,
                                  size_t                        n_signal,
                                  cg_rt_command_buffer_t       *command_buffer);

/* ------------------------------------------------------------------ */
/* Event tensors (paper support)                                       */
/* ------------------------------------------------------------------ */

typedef enum {
    CG_RT_EVENT_DTYPE_I32 = 1,
    CG_RT_EVENT_DTYPE_I64 = 2,
} cg_rt_event_dtype_t;

cg_rt_status_t cg_rt_event_tensor_create(cg_rt_device_t        *device,
                                         size_t                 rank,
                                         const int64_t         *shape,
                                         cg_rt_event_dtype_t    dtype,
                                         int64_t                initial_value,
                                         cg_rt_event_tensor_t **out_event_tensor);

void cg_rt_event_tensor_destroy(cg_rt_event_tensor_t *event_tensor);

size_t cg_rt_event_tensor_num_cells(const cg_rt_event_tensor_t *event_tensor);

/*
 * Atomic notify: decrement the counter at ``linear_idx`` by
 * ``decrement``.  When the counter reaches zero (or below) any blocked
 * waiters on that cell are released.
 */
cg_rt_status_t cg_rt_event_tensor_notify(cg_rt_event_tensor_t *event_tensor,
                                         size_t                linear_idx,
                                         int64_t               decrement);

cg_rt_status_t cg_rt_event_tensor_wait(cg_rt_event_tensor_t *event_tensor,
                                       size_t                linear_idx,
                                       uint64_t              timeout_ns);

cg_rt_status_t cg_rt_event_tensor_query(cg_rt_event_tensor_t *event_tensor,
                                        size_t                linear_idx,
                                        int64_t              *out_value);

cg_rt_status_t cg_rt_event_tensor_reset(cg_rt_event_tensor_t *event_tensor,
                                        int64_t               value);


/* ============================================================
 *  Phase 4 — CUDA megakernel runtime
 *
 *  Symbols below are exported only when the library is built with
 *  ``-DCG_RT_WITH_CUDA=ON`` AND a CUDA toolkit was available at
 *  configure time. On non-CUDA builds the linker rejects these
 *  symbols at load time, which the Python ctypes wrapper translates
 *  into a typed CudaUnavailableError.
 * ============================================================ */

/* CUDA event-tensor primitives (atomic ops + cooperative-spin wait).
 * Phase 4 ships device-side __device__ inlines + a host-callable
 * allocator. Phase 5's emitter inlines the device functions into
 * persistent-kernel bodies; the host shims below exist for testing
 * + bring-up.
 */
cg_rt_status_t cg_rt_cuda_etensor_alloc(long long **out_ptr,
                                        int         num_cells,
                                        long long   initial_wait_count);

void cg_rt_cuda_etensor_free(long long *ptr);

cg_rt_status_t cg_rt_cuda_etensor_load(long long *E,
                                       int        idx,
                                       long long *out_value);

/* On-GPU dynamic ready queue (Paper §3.2). One per megakernel
 * graph. Capacity must accommodate ceil(total_tasks * slack);
 * Phase 3's `compute_dynamic_schedule` picks the slack factor.
 */
cg_rt_status_t cg_rt_cuda_queue_alloc(int **out_ptr, int capacity);
void           cg_rt_cuda_queue_free (int  *ptr);

cg_rt_status_t cg_rt_cuda_queue_seed_initial(int       *q,
                                             const int *initial_task_ids,
                                             int        num_initial);

/* Persistent megakernel launcher.
 *
 * `kernel_handle` is the ``CUfunction`` Phase-5 NVRTC produced. The
 * launcher always sets ``CU_LAUNCH_ATTRIBUTE_COOPERATIVE``; cluster
 * launch is opt-in via nonzero `cluster_dim_*` fields.
 *
 * `kernel_args` is the standard CUDA varargs-style ``void**`` array
 * — the schedule-pass output knows the parameter order the persistent
 * kernel expects; Python's CudaMegakernelLauncher marshals tensor
 * device pointers + the dynamic queue pointer through this.
 */
typedef struct {
    void *kernel_handle;          /* CUfunction; opaque to the public ABI */
    int   grid_dim_x, grid_dim_y, grid_dim_z;
    int   block_dim_x, block_dim_y, block_dim_z;
    int   cluster_dim_x, cluster_dim_y, cluster_dim_z;  /* 0 ⇒ no cluster */
    int   shared_mem_bytes;
} cg_rt_cuda_megakernel_launch_t;

cg_rt_status_t cg_rt_cuda_megakernel_launch(
    cg_rt_device_t                       *device,
    const cg_rt_cuda_megakernel_launch_t *config,
    void                                **kernel_args);

/* Live device traits probe — Phase 6 native HAL backend.
 *
 * Filled by `cg_rt_cuda_probe_device`. Field order is stable; new
 * fields append at the end. Python's ctypes wrapper mirrors this
 * struct.
 */
typedef struct {
    char     device_name[128];
    int      compute_capability_major;
    int      compute_capability_minor;
    int      sm_count;
    int      num_visible_devices;
    int      max_threads_per_block;
    int      max_threads_per_multiprocessor;
    int      warp_size;
    int      max_grid_dim_x, max_grid_dim_y, max_grid_dim_z;
    long long max_device_memory_bytes;
    int      l2_cache_bytes;
    int      max_shared_memory_per_block_optin_bytes;
    int      max_blocks_per_cluster;
    int      cluster_launch;
    int      cooperative_launch;
    int      concurrent_kernels;
    int      concurrent_managed_access;
    int      supports_tma;
    int      supports_clusters;
    int      supports_fp8;
    int      supports_fp4;
    int      supports_ondevice_scheduler;
    int      driver_version;
    int      runtime_version;
} cg_rt_cuda_probe_t;

cg_rt_status_t cg_rt_cuda_probe_device(int                 device_index,
                                       cg_rt_cuda_probe_t *out);

/* -------------------------------------------------------------------------
 * Phase-4b: NCCL bridge — multi-GPU communication for the megakernel.
 *
 * Built only when CG_RT_WITH_NCCL is defined at compile time; on builds
 * without NCCL these symbols are not defined and ``cg_rt_cuda_comm_init``
 * will not be available. Callers should check via dlsym / has_attr in
 * the Python ctypes wrapper.
 *
 * The opaque ``cg_rt_cuda_comm_t`` wraps an array of ``ncclComm_t``
 * (one per local device) plus the per-device CUDA contexts, with peer
 * access enabled in both directions. Single-process multi-device
 * (``ncclCommInitAll``) is the only supported bring-up in v1; multi-
 * process bootstrap via ``ncclGetUniqueId`` + ``ncclCommInitRank``
 * lands in v2.
 * ------------------------------------------------------------------------- */

typedef struct cg_rt_cuda_comm cg_rt_cuda_comm_t;

cg_rt_status_t cg_rt_cuda_comm_init_local(int                 num_devices,
                                          const int          *device_indices,
                                          cg_rt_cuda_comm_t **out);

cg_rt_status_t cg_rt_cuda_comm_destroy(cg_rt_cuda_comm_t *comm);

/* Returns the number of devices the comm spans. */
int cg_rt_cuda_comm_size(cg_rt_cuda_comm_t *comm);

/* Trivial AllReduce over a per-rank float buffer. Used by the bring-up
 * smoke test to validate the NCCL link. ``input`` and ``output`` are
 * device pointers, one per rank in rank order. ``count`` is the number
 * of fp32 elements per rank.
 */
cg_rt_status_t cg_rt_cuda_comm_allreduce_fp32_sum(
    cg_rt_cuda_comm_t *comm,
    const void *const *inputs_per_rank,
    void *const       *outputs_per_rank,
    size_t             count);

#ifdef __cplusplus
}
#endif

#endif /* COMPGEN_RT_H_ */
