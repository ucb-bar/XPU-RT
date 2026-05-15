/*
 * CompGen Performance Counters — backend dispatcher.
 *
 * The backend is selected at compile time via CG_PERF_BACKEND.
 * Default is "none" (all operations are no-ops).
 *
 * Supported backends:
 *   - none          : Stub implementation (always returns 0).
 *   - linux_perf    : Linux perf_event_open.
 *   - zephyr_timing : Zephyr timing API (k_cycle_get / timing_*).
 *   - bare_metal_csr: RISC-V CSR reads.
 *   - cuda_cupti    : Stub (requires CUDA toolkit to build).
 */

#include "compgen/perf_counters.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Context structure                                                   */
/* ------------------------------------------------------------------ */

#define CG_PERF_MAX_COUNTERS 32

struct cg_perf_ctx {
    size_t   num_counters;
    char    *names[CG_PERF_MAX_COUNTERS];
    int64_t  values[CG_PERF_MAX_COUNTERS];
    int64_t  start_values[CG_PERF_MAX_COUNTERS];
    int      running;

#if defined(CG_PERF_BACKEND_LINUX_PERF)
    int      fds[CG_PERF_MAX_COUNTERS];
#endif
};

/* ------------------------------------------------------------------ */
/* Backend name                                                        */
/* ------------------------------------------------------------------ */

const char *cg_perf_backend_name(void) {
#if defined(CG_PERF_BACKEND_LINUX_PERF)
    return "linux_perf";
#elif defined(CG_PERF_BACKEND_ZEPHYR_TIMING)
    return "zephyr_timing";
#elif defined(CG_PERF_BACKEND_BARE_METAL_CSR)
    return "bare_metal_csr";
#elif defined(CG_PERF_BACKEND_CUDA_CUPTI)
    return "cuda_cupti";
#else
    return "none";
#endif
}

/* ------------------------------------------------------------------ */
/* Common helpers                                                      */
/* ------------------------------------------------------------------ */

static char *_strdup(const char *s) {
    size_t len = strlen(s);
    char *dup = (char *)malloc(len + 1);
    if (dup) memcpy(dup, s, len + 1);
    return dup;
}

/* ------------------------------------------------------------------ */
/* Backend: none (stub)                                                */
/* ------------------------------------------------------------------ */

#if !defined(CG_PERF_BACKEND_LINUX_PERF)    && \
    !defined(CG_PERF_BACKEND_ZEPHYR_TIMING) && \
    !defined(CG_PERF_BACKEND_BARE_METAL_CSR) && \
    !defined(CG_PERF_BACKEND_CUDA_CUPTI)

int cg_perf_ctx_create(const char *const *counter_names,
                        size_t num_counters,
                        cg_perf_ctx_t **out_ctx) {
    if (!out_ctx) return -1;
    if (num_counters > CG_PERF_MAX_COUNTERS) return -1;

    cg_perf_ctx_t *ctx = (cg_perf_ctx_t *)calloc(1, sizeof(cg_perf_ctx_t));
    if (!ctx) return -1;

    ctx->num_counters = num_counters;
    for (size_t i = 0; i < num_counters; i++) {
        ctx->names[i] = _strdup(counter_names[i]);
    }

    *out_ctx = ctx;
    return 0;
}

void cg_perf_ctx_destroy(cg_perf_ctx_t *ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->num_counters; i++) {
        free(ctx->names[i]);
    }
    free(ctx);
}

int cg_perf_start(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    ctx->running = 1;
    return 0;
}

int cg_perf_stop(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    ctx->running = 0;
    return 0;
}

int cg_perf_read(cg_perf_ctx_t *ctx, int64_t *out_values, size_t max_values) {
    if (!ctx || !out_values) return -1;
    size_t n = ctx->num_counters < max_values ? ctx->num_counters : max_values;
    for (size_t i = 0; i < n; i++) {
        out_values[i] = ctx->values[i];
    }
    return (int)n;
}

int cg_perf_reset(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    memset(ctx->values, 0, sizeof(ctx->values));
    memset(ctx->start_values, 0, sizeof(ctx->start_values));
    return 0;
}

#endif /* none backend */

/* ------------------------------------------------------------------ */
/* Backend: bare_metal_csr (RISC-V)                                    */
/* ------------------------------------------------------------------ */

#if defined(CG_PERF_BACKEND_BARE_METAL_CSR)

static int64_t _read_csr(const char *name) {
    int64_t val = 0;
    #if defined(__riscv)
    if (strcmp(name, "cycles") == 0 || strcmp(name, "mcycle") == 0) {
        __asm__ volatile("rdcycle %0" : "=r"(val));
    } else if (strcmp(name, "instructions") == 0 || strcmp(name, "minstret") == 0) {
        __asm__ volatile("rdinstret %0" : "=r"(val));
    }
    /* Additional mhpmcounter3..31 would use csrr instructions */
    #endif
    (void)name;
    return val;
}

int cg_perf_ctx_create(const char *const *counter_names,
                        size_t num_counters,
                        cg_perf_ctx_t **out_ctx) {
    if (!out_ctx) return -1;
    if (num_counters > CG_PERF_MAX_COUNTERS) return -1;

    cg_perf_ctx_t *ctx = (cg_perf_ctx_t *)calloc(1, sizeof(cg_perf_ctx_t));
    if (!ctx) return -1;

    ctx->num_counters = num_counters;
    for (size_t i = 0; i < num_counters; i++) {
        ctx->names[i] = _strdup(counter_names[i]);
    }

    *out_ctx = ctx;
    return 0;
}

void cg_perf_ctx_destroy(cg_perf_ctx_t *ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->num_counters; i++) {
        free(ctx->names[i]);
    }
    free(ctx);
}

int cg_perf_start(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    for (size_t i = 0; i < ctx->num_counters; i++) {
        ctx->start_values[i] = _read_csr(ctx->names[i]);
    }
    ctx->running = 1;
    return 0;
}

int cg_perf_stop(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    for (size_t i = 0; i < ctx->num_counters; i++) {
        int64_t end = _read_csr(ctx->names[i]);
        ctx->values[i] = end - ctx->start_values[i];
    }
    ctx->running = 0;
    return 0;
}

int cg_perf_read(cg_perf_ctx_t *ctx, int64_t *out_values, size_t max_values) {
    if (!ctx || !out_values) return -1;
    size_t n = ctx->num_counters < max_values ? ctx->num_counters : max_values;
    for (size_t i = 0; i < n; i++) {
        out_values[i] = ctx->values[i];
    }
    return (int)n;
}

int cg_perf_reset(cg_perf_ctx_t *ctx) {
    if (!ctx) return -1;
    memset(ctx->values, 0, sizeof(ctx->values));
    memset(ctx->start_values, 0, sizeof(ctx->start_values));
    return 0;
}

#endif /* bare_metal_csr backend */

/* ------------------------------------------------------------------ */
/* Common query functions                                              */
/* ------------------------------------------------------------------ */

size_t cg_perf_num_counters(const cg_perf_ctx_t *ctx) {
    return ctx ? ctx->num_counters : 0;
}

const char *cg_perf_counter_name(const cg_perf_ctx_t *ctx, size_t index) {
    if (!ctx || index >= ctx->num_counters) return NULL;
    return ctx->names[index];
}
