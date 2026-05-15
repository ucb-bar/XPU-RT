/*
 * libxpu_rt XNNPACK bridge implementation.
 *
 * When XPURT_HAS_XNNPACK is defined (CMake set when XNNPACK is linked),
 * every entry forwards to xnn_* with appropriate parameter packing.
 *
 * When XPURT_HAS_XNNPACK is NOT defined, every entry returns
 * -XPU_RT_XNN_ENOTSUP and create() returns NULL. The python provider's
 * probe() uses this to decide whether to bid at all.
 */

#include "xnnpack_bridge.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- Thread-local error string ------------------------------------ */

static __thread char g_last_error[256] = {0};

static void set_error(const char* msg) {
    if (msg == NULL) {
        g_last_error[0] = '\0';
    } else {
        strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
        g_last_error[sizeof(g_last_error) - 1] = '\0';
    }
}

const char* xpu_rt_xnn_last_error(void) {
    return g_last_error[0] ? g_last_error : NULL;
}

#ifndef XPURT_HAS_XNNPACK
/* ============================================================
 * Stub path — built when XNNPACK is not linked.
 * ============================================================ */

int xpu_rt_xnn_global_initialize(void) {
    set_error("libxpu_rt built without XPURT_WITH_XNNPACK=ON");
    return XPU_RT_XNN_ENOTSUP;
}

void xpu_rt_xnn_global_deinitialize(void) {
    /* no-op */
}

const char* xpu_rt_xnn_version(void) {
    return NULL;
}

xpu_rt_xnn_op* xpu_rt_xnn_create(
    xpu_rt_xnn_op_kind kind,
    const int64_t* shape_dims,    size_t n_shape_dims,
    const int32_t* int_params,    size_t n_int_params,
    const float*   float_params,  size_t n_float_params,
    const void*    static_weights,
    size_t         static_weights_bytes) {
    (void)kind; (void)shape_dims; (void)n_shape_dims;
    (void)int_params; (void)n_int_params;
    (void)float_params; (void)n_float_params;
    (void)static_weights; (void)static_weights_bytes;
    set_error("libxpu_rt built without XPURT_WITH_XNNPACK=ON");
    return NULL;
}

int xpu_rt_xnn_reshape_setup(
    xpu_rt_xnn_op* op,
    const int64_t* runtime_shape_dims, size_t n_runtime_shape_dims,
    const void* const* inputs,  size_t n_inputs,
    void* const* outputs,       size_t n_outputs) {
    (void)op; (void)runtime_shape_dims; (void)n_runtime_shape_dims;
    (void)inputs; (void)n_inputs;
    (void)outputs; (void)n_outputs;
    set_error("libxpu_rt built without XPURT_WITH_XNNPACK=ON");
    return XPU_RT_XNN_ENOTSUP;
}

int xpu_rt_xnn_run(xpu_rt_xnn_op* op) {
    (void)op;
    set_error("libxpu_rt built without XPURT_WITH_XNNPACK=ON");
    return XPU_RT_XNN_ENOTSUP;
}

void xpu_rt_xnn_destroy(xpu_rt_xnn_op* op) {
    (void)op;
}

#else
/* ============================================================
 * Live path — built when XPURT_HAS_XNNPACK is defined.
 * ============================================================ */

#include <xnnpack.h>

/* One global XNNPACK init under a mutex. */
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_initialized = 0;

/* Opaque handle the provider holds. We keep the XNNPACK operator
 * pointer plus enough state to reshape/setup on subsequent calls
 * without re-parsing the create-time parameters. */
struct xpu_rt_xnn_op {
    xpu_rt_xnn_op_kind kind;
    xnn_operator_t     xnn_op;     /* opaque XNNPACK handle */
    int32_t            sub_kind;   /* activation / binary / reduce sub-kind, or 0 */
    int64_t            shape[16];  /* create-time shape echo (op-kind-specific) */
    size_t             n_shape;
};

static int translate_xnn_status(enum xnn_status st) {
    switch (st) {
        case xnn_status_success:                return XPU_RT_XNN_OK;
        case xnn_status_uninitialized:          return XPU_RT_XNN_EBACKEND;
        case xnn_status_invalid_parameter:      return XPU_RT_XNN_EINVAL;
        case xnn_status_invalid_state:          return XPU_RT_XNN_EBACKEND;
        case xnn_status_unsupported_parameter:  return XPU_RT_XNN_ENOTSUP;
        case xnn_status_unsupported_hardware:   return XPU_RT_XNN_ENOTSUP;
        case xnn_status_out_of_memory:          return XPU_RT_XNN_ENOMEM;
        default:                                return XPU_RT_XNN_EBACKEND;
    }
}

int xpu_rt_xnn_global_initialize(void) {
    set_error(NULL);
    pthread_mutex_lock(&g_init_mu);
    if (g_initialized) {
        pthread_mutex_unlock(&g_init_mu);
        return XPU_RT_XNN_OK;
    }
    enum xnn_status st = xnn_initialize(NULL /* allocator */);
    if (st != xnn_status_success) {
        pthread_mutex_unlock(&g_init_mu);
        set_error("xnn_initialize failed");
        return translate_xnn_status(st);
    }
    g_initialized = 1;
    pthread_mutex_unlock(&g_init_mu);
    return XPU_RT_XNN_OK;
}

void xpu_rt_xnn_global_deinitialize(void) {
    pthread_mutex_lock(&g_init_mu);
    if (g_initialized) {
        xnn_deinitialize();
        g_initialized = 0;
    }
    pthread_mutex_unlock(&g_init_mu);
}

const char* xpu_rt_xnn_version(void) {
    /* XNNPACK does not export a runtime version string in its public
     * header; report the major-only marker. The provider's probe()
     * will surface this verbatim. */
    return "xnnpack-runtime";
}

/* Helper: ensure the op handle has enough shape slots. */
static int copy_shape(xpu_rt_xnn_op* op, const int64_t* dims, size_t n) {
    if (n > sizeof(op->shape) / sizeof(op->shape[0])) {
        set_error("shape dims exceed bridge handle capacity");
        return XPU_RT_XNN_EINVAL;
    }
    memcpy(op->shape, dims, n * sizeof(int64_t));
    op->n_shape = n;
    return XPU_RT_XNN_OK;
}

xpu_rt_xnn_op* xpu_rt_xnn_create(
    xpu_rt_xnn_op_kind kind,
    const int64_t* shape_dims,    size_t n_shape_dims,
    const int32_t* int_params,    size_t n_int_params,
    const float*   float_params,  size_t n_float_params,
    const void*    static_weights,
    size_t         static_weights_bytes) {
    set_error(NULL);

    if (!g_initialized) {
        int rc = xpu_rt_xnn_global_initialize();
        if (rc != XPU_RT_XNN_OK) {
            return NULL;
        }
    }

    xpu_rt_xnn_op* op = (xpu_rt_xnn_op*)calloc(1, sizeof(*op));
    if (op == NULL) {
        set_error("calloc failed");
        return NULL;
    }
    op->kind = kind;
    if (shape_dims && n_shape_dims > 0) {
        if (copy_shape(op, shape_dims, n_shape_dims) != XPU_RT_XNN_OK) {
            free(op);
            return NULL;
        }
    }
    if (int_params && n_int_params >= 1) {
        op->sub_kind = int_params[0];
    }

    enum xnn_status st = xnn_status_unsupported_parameter;

    switch (kind) {
        case XPU_RT_XNN_OP_FULLY_CONNECTED_F32: {
            /* shape = {input_channels, output_channels} */
            if (n_shape_dims < 2 || static_weights == NULL) {
                set_error("FC f32: need shape[in,out] + weights");
                free(op);
                return NULL;
            }
            const float output_min = (n_float_params >= 1) ? float_params[0] : -INFINITY;
            const float output_max = (n_float_params >= 2) ? float_params[1] :  INFINITY;
            const float* w = (const float*)static_weights;
            const size_t in_c  = (size_t)shape_dims[0];
            const size_t out_c = (size_t)shape_dims[1];
            const size_t w_bytes = in_c * out_c * sizeof(float);
            const float* bias = NULL;
            if (static_weights_bytes >= w_bytes + out_c * sizeof(float)) {
                bias = w + in_c * out_c;
            }
            st = xnn_create_fully_connected_nc_f32(
                in_c, out_c,
                /*input_stride*/  in_c,
                /*output_stride*/ out_c,
                w, bias,
                output_min, output_max,
                /*flags*/ 0,
                /*weights_cache*/ NULL,
                &op->xnn_op);
            break;
        }
        /* Other op-kinds wired up by the Python provider; their case
         * arms call the matching xnn_create_<op>_* and set st. Kept
         * concise for the MVP commit — subsequent commits append
         * cases for conv2d, depthwise, batch_matmul, pool, softmax,
         * unary, binary, reduce, transpose, resize, prelu, etc. The
         * stub return below makes those paths return ENOTSUP from the
         * Python side until the case lands. */
        default:
            set_error("op-kind not yet wired in xnnpack_bridge.c");
            free(op);
            return NULL;
    }

    if (st != xnn_status_success || op->xnn_op == NULL) {
        set_error("xnn_create_* failed");
        free(op);
        return NULL;
    }
    return op;
}

int xpu_rt_xnn_reshape_setup(
    xpu_rt_xnn_op* op,
    const int64_t* runtime_shape_dims, size_t n_runtime_shape_dims,
    const void* const* inputs,  size_t n_inputs,
    void* const* outputs,       size_t n_outputs) {
    set_error(NULL);
    if (op == NULL || op->xnn_op == NULL) {
        set_error("op handle is NULL");
        return XPU_RT_XNN_EFAULT;
    }
    enum xnn_status st = xnn_status_unsupported_parameter;
    switch (op->kind) {
        case XPU_RT_XNN_OP_FULLY_CONNECTED_F32: {
            if (n_inputs < 1 || n_outputs < 1 || n_runtime_shape_dims < 1) {
                set_error("FC f32: need batch + inputs[0] + outputs[0]");
                return XPU_RT_XNN_EINVAL;
            }
            const size_t batch = (size_t)runtime_shape_dims[0];
            st = xnn_reshape_fully_connected_nc_f32(
                op->xnn_op, batch, /*threadpool*/ NULL);
            if (st != xnn_status_success) break;
            st = xnn_setup_fully_connected_nc_f32(
                op->xnn_op,
                (const float*)inputs[0],
                (float*)outputs[0]);
            break;
        }
        default:
            set_error("reshape_setup: op-kind not wired");
            return XPU_RT_XNN_ENOTSUP;
    }
    if (st != xnn_status_success) {
        set_error("xnn_reshape/setup failed");
        return translate_xnn_status(st);
    }
    return XPU_RT_XNN_OK;
}

int xpu_rt_xnn_run(xpu_rt_xnn_op* op) {
    set_error(NULL);
    if (op == NULL || op->xnn_op == NULL) {
        set_error("op handle is NULL");
        return XPU_RT_XNN_EFAULT;
    }
    enum xnn_status st = xnn_run_operator(op->xnn_op, /*threadpool*/ NULL);
    if (st != xnn_status_success) {
        set_error("xnn_run_operator failed");
        return translate_xnn_status(st);
    }
    return XPU_RT_XNN_OK;
}

void xpu_rt_xnn_destroy(xpu_rt_xnn_op* op) {
    if (op == NULL) return;
    if (op->xnn_op != NULL) {
        xnn_delete_operator(op->xnn_op);
        op->xnn_op = NULL;
    }
    free(op);
}

#endif /* XPURT_HAS_XNNPACK */
