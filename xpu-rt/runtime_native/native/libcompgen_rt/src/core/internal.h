/*
 * Internal definitions for the cpu_sync driver. Not part of the public
 * ABI — opaque handles in compgen_rt.h are defined here.
 */

#ifndef COMPGEN_RT_INTERNAL_H_
#define COMPGEN_RT_INTERNAL_H_

#include "compgen_rt/compgen_rt.h"
#include "platform.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations of driver vtable (future-proofing for
 * multi-driver support). The cpu_sync driver uses a singleton
 * vtable instance initialised by cg_rt_instance_create. */

typedef struct cg_rt_driver_vtable {
    const char *name;
    /* device ops — drivers populate these inline. */
    cg_rt_status_t (*device_open)(cg_rt_instance_t *instance,
                                  uint32_t          device_index,
                                  cg_rt_device_t  **out_device);
    void (*device_close)(cg_rt_device_t *device);

    /* Optional: how many devices this driver exposes. NULL means
     * "ask via device_open" (legacy / single-device drivers). When
     * provided, ``cg_rt_instance_create`` calls this to populate
     * ``instance->device_count`` so the public API's bounds check
     * accepts every valid index. CUDA fills this in from
     * ``cuDeviceGetCount``; cpu_sync hardcodes 1. */
    uint32_t (*query_device_count)(void);

    /* Queue submit — per-driver because scheduling semantics differ
     * (cpu_sync is blocking, cpu_task is async, CUDA streams, etc.). */
    cg_rt_status_t (*queue_submit)(cg_rt_device_t               *device,
                                   uint32_t                      queue_index,
                                   const cg_rt_semaphore_point_t *wait,
                                   size_t                        n_wait,
                                   const cg_rt_semaphore_point_t *signal,
                                   size_t                        n_signal,
                                   cg_rt_command_buffer_t       *command_buffer);

    /* Optional: override buffer backing-memory allocation. When NULL
     * the shared host allocator (calloc/free) is used. CUDA provides
     * cuMemAllocManaged / cuMemFree so buffers live in GPU-visible
     * memory but remain host-addressable for cg_rt_buffer_map. */
    cg_rt_status_t (*data_alloc)(cg_rt_device_t       *device,
                                 size_t                size_bytes,
                                 cg_rt_memory_space_t  memory_space,
                                 void                **out_ptr);
    void (*data_free)(cg_rt_device_t *device, void *ptr);
} cg_rt_driver_vtable_t;

struct cg_rt_instance {
    const cg_rt_driver_vtable_t *vtable;
    uint32_t device_count;
};

struct cg_rt_device {
    const cg_rt_driver_vtable_t *vtable;
    cg_rt_device_traits_t        traits;
    uint32_t                     device_index;

    /* cpu_sync per-queue serialisation. All queue submits hold
     * ``queue_mutex[queue_index]`` for the duration of the submit so
     * submissions on the same queue are ordered but submissions on
     * different queues can run concurrently. cpu_task uses a
     * different per-driver struct stored in ``driver_state``. */
    cg_rt_platform_mutex_t *queue_mutexes;
    size_t           num_queues;

    /* Opaque per-driver state — set by the driver's ``device_open``
     * and freed by its ``device_close``. */
    void *driver_state;
};

/* -------------------------------------------------------------------- */
/* Buffer                                                                */
/* -------------------------------------------------------------------- */

struct cg_rt_buffer {
    void                 *data;
    size_t                size;
    cg_rt_memory_space_t  memory_space;
    uint32_t              usage_flags;
    bool                  mapped;
    cg_rt_device_t       *device;   /* owner, for data_free dispatch */
};

/* -------------------------------------------------------------------- */
/* Semaphore — timeline                                                  */
/* -------------------------------------------------------------------- */

struct cg_rt_semaphore {
    cg_rt_platform_mutex_t mutex;
    cg_rt_platform_cond_t  cond;
    _Atomic uint64_t value;       /* current payload */
    _Atomic int32_t  failure_code; /* 0 = healthy; <0 = failed */
};

/* -------------------------------------------------------------------- */
/* Executable                                                            */
/* -------------------------------------------------------------------- */

struct cg_rt_executable {
    /* CPU executables set ``entry_point`` and leave ``driver_impl``
     * NULL.  Driver-specific executables (e.g. CUDA) set
     * ``driver_impl`` to a pointer to their private struct and leave
     * ``entry_point`` NULL.  ``destroy_impl`` — when non-NULL — is
     * called from ``cg_rt_executable_destroy`` to release driver
     * resources before the generic free runs. */
    cg_rt_cpu_kernel_fn entry_point;
    void *              driver_impl;
    void (*destroy_impl)(cg_rt_executable_t *);
};

/* -------------------------------------------------------------------- */
/* Command buffer — recorded list of opcodes                             */
/* -------------------------------------------------------------------- */

typedef enum {
    CG_RT_CMD_OP_COPY     = 1,
    CG_RT_CMD_OP_FILL     = 2,
    CG_RT_CMD_OP_DISPATCH = 3,
    CG_RT_CMD_OP_BARRIER  = 4,
} cg_rt_cmd_op_t;

typedef struct {
    cg_rt_cmd_op_t op;
    union {
        struct {
            cg_rt_buffer_t *src;
            size_t          src_offset;
            cg_rt_buffer_t *dst;
            size_t          dst_offset;
            size_t          size_bytes;
        } copy;
        struct {
            cg_rt_buffer_t *dst;
            size_t          dst_offset;
            size_t          size_bytes;
            uint32_t        pattern;
        } fill;
        struct {
            cg_rt_executable_t *executable;
            void               *push_constants;  /* malloc'd copy */
            size_t              pc_size;
            cg_rt_buffer_t    **bindings;         /* malloc'd array */
            size_t              n_bindings;
        } dispatch;
    };
} cg_rt_command_t;

typedef enum {
    CG_RT_CB_STATE_NEW      = 0,
    CG_RT_CB_STATE_RECORDING,
    CG_RT_CB_STATE_EXECUTABLE,
} cg_rt_cb_state_t;

struct cg_rt_command_buffer {
    cg_rt_device_t   *device;
    cg_rt_command_t  *commands;
    size_t            num_commands;
    size_t            capacity;
    cg_rt_cb_state_t  state;
};

/* -------------------------------------------------------------------- */
/* Event tensor                                                          */
/* -------------------------------------------------------------------- */

struct cg_rt_event_tensor {
    size_t               rank;
    int64_t             *shape;      /* length == rank */
    size_t               num_cells;
    cg_rt_event_dtype_t  dtype;
    /* Backing storage: always int64 internally; i32 mode is enforced
     * at the notify/query boundary by clamping. */
    _Atomic int64_t          *cells;      /* length == num_cells */
    cg_rt_platform_mutex_t    mutex;
    cg_rt_platform_cond_t     cond;
};

/* -------------------------------------------------------------------- */
/* Shared CPU command-buffer executor                                    */
/* -------------------------------------------------------------------- */

/* Replay a finalised command buffer on the host. Used by both cpu_sync
 * (inline in queue_submit) and cpu_task (on worker threads). Does not
 * touch semaphores — the caller owns the wait/signal sequencing. */
cg_rt_status_t cg_rt_execute_cpu_command_buffer(cg_rt_command_buffer_t *cb);

/* -------------------------------------------------------------------- */
/* Driver vtable exports                                                 */
/* -------------------------------------------------------------------- */

extern const cg_rt_driver_vtable_t cg_rt_cpu_sync_vtable;

/* cpu_task depends on platform threads — omitted on the bare backend. */
#ifndef CG_RT_PLATFORM_BARE
extern const cg_rt_driver_vtable_t cg_rt_cpu_task_vtable;
#endif

#ifdef CG_RT_WITH_CUDA
extern const cg_rt_driver_vtable_t cg_rt_cuda_vtable;
#endif

#endif /* COMPGEN_RT_INTERNAL_H_ */
