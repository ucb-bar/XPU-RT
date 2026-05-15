/*
 * libxpu_rt XNNPACK bridge — stable C ABI shim.
 *
 * The XNNPACK provider in xpu_rt.kernels.providers.xnnpack emits per-
 * dispatch C kernel artifacts that call THIS header's symbols rather
 * than <xnnpack.h> directly. That keeps the artifact ABI stable across
 * XNNPACK version bumps and out of the kernel-ABI hash inputs.
 *
 * Lifecycle for any op:
 *   1. xpu_rt_xnn_global_initialize()           once per process
 *   2. xpu_rt_xnn_create(kind, ...)             at kernel-init time
 *   3. xpu_rt_xnn_reshape_setup(op, ...)        before every run
 *   4. xpu_rt_xnn_run(op)                       executes
 *   5. xpu_rt_xnn_destroy(op)                   at teardown
 *
 * When libxpu_rt is built without XPURT_WITH_XNNPACK, every entry
 * returns -XPU_RT_XNN_ENOTSUP and xpu_rt_xnn_create returns NULL so
 * the Python provider's probe() can detect the build flag honestly.
 */

#ifndef XPU_RT_XNNPACK_BRIDGE_H
#define XPU_RT_XNNPACK_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wire-stable enum. New entries APPEND-ONLY; never renumber. */
typedef enum {
    XPU_RT_XNN_OP_UNKNOWN                              = 0,

    /* f32 NHWC core */
    XPU_RT_XNN_OP_FULLY_CONNECTED_F32                  = 1,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_F32               = 2,
    XPU_RT_XNN_OP_DEPTHWISE_CONVOLUTION2D_NHWC_F32     = 3,
    XPU_RT_XNN_OP_BATCH_MATMUL_F32                     = 4,
    XPU_RT_XNN_OP_AVERAGE_POOLING2D_NHWC_F32           = 5,
    XPU_RT_XNN_OP_MAX_POOLING2D_NHWC_F32               = 6,
    XPU_RT_XNN_OP_GLOBAL_AVERAGE_POOLING_NHWC_F32      = 7,
    XPU_RT_XNN_OP_SOFTMAX_NC_F32                       = 8,
    XPU_RT_XNN_OP_UNARY_F32                            = 9,
    XPU_RT_XNN_OP_BINARY_F32                           = 10,
    XPU_RT_XNN_OP_REDUCE_F32                           = 11,
    XPU_RT_XNN_OP_TRANSPOSE                            = 12,
    XPU_RT_XNN_OP_RESIZE_BILINEAR2D_NHWC_F32           = 13,
    XPU_RT_XNN_OP_DECONVOLUTION2D_NHWC_F32             = 14,
    XPU_RT_XNN_OP_PRELU_NHWC_F32                       = 15,
    XPU_RT_XNN_OP_LEAKY_RELU_NHWC_F32                  = 16,
    XPU_RT_XNN_OP_STATIC_SLICE                         = 17,

    /* f16 — reserved at +100 */
    XPU_RT_XNN_OP_FULLY_CONNECTED_F16                  = 101,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_F16               = 102,
    XPU_RT_XNN_OP_BATCH_MATMUL_F16                     = 104,
    /* ...remaining f16 slots appended as wired up */

    /* Quantised — reserved at +200 (qs8), +300 (qu8), +400 (qd8→f32) */
    XPU_RT_XNN_OP_FULLY_CONNECTED_QS8                  = 201,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_QS8               = 202,
    XPU_RT_XNN_OP_DEPTHWISE_CONVOLUTION2D_NHWC_QS8     = 203,

    XPU_RT_XNN_OP_FULLY_CONNECTED_QU8                  = 301,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_QU8               = 302,

    XPU_RT_XNN_OP_FULLY_CONNECTED_QD8_F32              = 401,
    XPU_RT_XNN_OP_CONVOLUTION2D_NHWC_QD8_F32           = 402,
    XPU_RT_XNN_OP_BATCH_MATMUL_QD8_F32                 = 404,
} xpu_rt_xnn_op_kind;

/* Activation sub-kind for XPU_RT_XNN_OP_UNARY_F32. */
typedef enum {
    XPU_RT_XNN_UNARY_RELU      = 1,
    XPU_RT_XNN_UNARY_SIGMOID   = 2,
    XPU_RT_XNN_UNARY_TANH      = 3,
    XPU_RT_XNN_UNARY_HSWISH    = 4,
    XPU_RT_XNN_UNARY_GELU      = 5,
    XPU_RT_XNN_UNARY_ELU       = 6,
    XPU_RT_XNN_UNARY_ABS       = 7,
    XPU_RT_XNN_UNARY_NEG       = 8,
    XPU_RT_XNN_UNARY_FLOOR     = 9,
    XPU_RT_XNN_UNARY_CEIL      = 10,
    XPU_RT_XNN_UNARY_SQUARE    = 11,
    XPU_RT_XNN_UNARY_SQRT      = 12,
    XPU_RT_XNN_UNARY_LOG       = 13,
    XPU_RT_XNN_UNARY_EXP       = 14,
} xpu_rt_xnn_unary_kind;

/* Sub-kind for XPU_RT_XNN_OP_BINARY_F32. */
typedef enum {
    XPU_RT_XNN_BINARY_ADD      = 1,
    XPU_RT_XNN_BINARY_SUB      = 2,
    XPU_RT_XNN_BINARY_MUL      = 3,
    XPU_RT_XNN_BINARY_DIV      = 4,
    XPU_RT_XNN_BINARY_MAX      = 5,
    XPU_RT_XNN_BINARY_MIN      = 6,
    XPU_RT_XNN_BINARY_SQUARED_DIFFERENCE = 7,
} xpu_rt_xnn_binary_kind;

/* Sub-kind for XPU_RT_XNN_OP_REDUCE_F32. */
typedef enum {
    XPU_RT_XNN_REDUCE_SUM      = 1,
    XPU_RT_XNN_REDUCE_MEAN     = 2,
    XPU_RT_XNN_REDUCE_MAX      = 3,
    XPU_RT_XNN_REDUCE_MIN      = 4,
} xpu_rt_xnn_reduce_kind;

/* Error codes — mirror libxpu_rt's posix-style negative-errno convention. */
#define XPU_RT_XNN_OK         0
#define XPU_RT_XNN_EINVAL    -22
#define XPU_RT_XNN_ENOTSUP   -95
#define XPU_RT_XNN_ENOMEM    -12
#define XPU_RT_XNN_EFAULT    -14
#define XPU_RT_XNN_EBACKEND  -200   /* underlying xnn_* call returned non-success */

/* Opaque handle. */
typedef struct xpu_rt_xnn_op xpu_rt_xnn_op;

/*
 * Process-wide one-time XNNPACK init. Safe to call multiple times
 * (idempotent). Returns 0 on success, negative xpu-rt-xnn errno on
 * failure (e.g. -XPU_RT_XNN_ENOTSUP when libxpu_rt was built without
 * XPURT_WITH_XNNPACK).
 */
int xpu_rt_xnn_global_initialize(void);

/* Tear down XNNPACK process state. Safe to call without a prior init. */
void xpu_rt_xnn_global_deinitialize(void);

/*
 * Returns the XNNPACK version string ("vM.m.p") if the runtime was
 * built with XPURT_WITH_XNNPACK, otherwise NULL.
 */
const char* xpu_rt_xnn_version(void);

/*
 * Create an XNNPACK operator handle.
 *
 * `shape_dims`        — compile-time-known shape (rank-aware; e.g. for
 *                       convolution this is {N, IH, IW, IC, KH, KW, OC,
 *                       stride_h, stride_w, dilation_h, dilation_w,
 *                       pad_top, pad_right, pad_bottom, pad_left,
 *                       groups}). Layout is op-kind-specific; see
 *                       xpu-rt/python/xpu_rt/kernels/xnnpack_adapter.py
 *                       for the canonical packing.
 * `int_params`        — small integer parameters (e.g. unary sub-kind,
 *                       reduction axes encoded as a bitmask, transpose
 *                       perm).
 * `float_params`      — small float parameters (e.g. clamp min/max,
 *                       leaky-relu slope, requantisation scales).
 *
 * Returns the opaque handle, or NULL on any failure. Inspect
 * xpu_rt_xnn_last_error() for the typed reason.
 */
xpu_rt_xnn_op* xpu_rt_xnn_create(
    xpu_rt_xnn_op_kind kind,
    const int64_t* shape_dims,    size_t n_shape_dims,
    const int32_t* int_params,    size_t n_int_params,
    const float*   float_params,  size_t n_float_params,
    const void*    static_weights,  /* may be NULL */
    size_t         static_weights_bytes);

/*
 * Reshape + setup. Must be called before every run when input shapes
 * change. Cheap when they don't.
 *
 * `runtime_shape_dims` — runtime-known shape entries (e.g. batch size
 *                       if the op was created with dynamic-batch).
 * `inputs`             — array of pointers to input buffers; layout
 *                       NHWC for spatial ops, NC for FC/softmax.
 * `outputs`            — array of pointers to output buffers.
 *
 * Returns 0 on success, negative on failure.
 */
int xpu_rt_xnn_reshape_setup(
    xpu_rt_xnn_op* op,
    const int64_t* runtime_shape_dims, size_t n_runtime_shape_dims,
    const void* const* inputs,  size_t n_inputs,
    void* const* outputs,       size_t n_outputs);

/* Execute the operator. Returns 0 on success, negative on failure. */
int xpu_rt_xnn_run(xpu_rt_xnn_op* op);

/* Destroy the operator handle. Safe to call with NULL. */
void xpu_rt_xnn_destroy(xpu_rt_xnn_op* op);

/*
 * Thread-local last-error string (NUL-terminated). Cleared at the
 * start of every public call. NULL when there is no error.
 */
const char* xpu_rt_xnn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* XPU_RT_XNNPACK_BRIDGE_H */
