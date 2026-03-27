/*
 * CompGen Performance Counters — hardware PMU abstraction.
 *
 * Provides a target-agnostic API for reading hardware performance
 * counters.  The backend is selected at compile time via CG_PERF_BACKEND:
 *
 *   - linux_perf    : Linux perf_event_open (x86, ARM, RISC-V)
 *   - zephyr_timing : Zephyr k_cycle_get / timing API
 *   - bare_metal_csr: RISC-V CSR reads (mcycle, minstret, mhpmcounterN)
 *   - cuda_cupti    : NVIDIA CUPTI (stub — requires CUDA toolkit)
 *   - none          : All operations are no-ops
 *
 * Usage:
 *   cg_perf_ctx_t *ctx = NULL;
 *   const char *counters[] = {"cycles", "instructions"};
 *   cg_perf_ctx_create(counters, 2, &ctx);
 *   cg_perf_start(ctx);
 *   ... kernel ...
 *   cg_perf_stop(ctx);
 *   int64_t values[2];
 *   cg_perf_read(ctx, values, 2);
 *   cg_perf_ctx_destroy(ctx);
 */

#ifndef COMPGEN_PERF_COUNTERS_H_
#define COMPGEN_PERF_COUNTERS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Opaque context handle                                               */
/* ------------------------------------------------------------------ */

typedef struct cg_perf_ctx cg_perf_ctx_t;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/**
 * Create a performance counter context.
 *
 * @param counter_names  Array of counter name strings.  Names are
 *                       backend-specific (see docs for each backend).
 * @param num_counters   Number of counters to monitor.
 * @param out_ctx        On success, receives the new context pointer.
 * @return 0 on success, non-zero on failure.
 */
int cg_perf_ctx_create(const char *const *counter_names,
                        size_t num_counters,
                        cg_perf_ctx_t **out_ctx);

/**
 * Destroy a performance counter context and release resources.
 */
void cg_perf_ctx_destroy(cg_perf_ctx_t *ctx);

/* ------------------------------------------------------------------ */
/* Measurement                                                         */
/* ------------------------------------------------------------------ */

/**
 * Start counting.  Must be called before cg_perf_stop.
 *
 * @return 0 on success, non-zero on failure.
 */
int cg_perf_start(cg_perf_ctx_t *ctx);

/**
 * Stop counting.  After this call, values can be read.
 *
 * @return 0 on success, non-zero on failure.
 */
int cg_perf_stop(cg_perf_ctx_t *ctx);

/**
 * Read counter values accumulated between start and stop.
 *
 * @param ctx         The context.
 * @param out_values  Array of at least num_counters int64_t values.
 * @param max_values  Size of the out_values array.
 * @return Number of values written, or -1 on error.
 */
int cg_perf_read(cg_perf_ctx_t *ctx, int64_t *out_values, size_t max_values);

/**
 * Reset all counters to zero without destroying the context.
 *
 * @return 0 on success, non-zero on failure.
 */
int cg_perf_reset(cg_perf_ctx_t *ctx);

/* ------------------------------------------------------------------ */
/* Query                                                               */
/* ------------------------------------------------------------------ */

/**
 * Return the number of counters in this context.
 */
size_t cg_perf_num_counters(const cg_perf_ctx_t *ctx);

/**
 * Return the name of counter at the given index.
 *
 * @return Interned string (valid for the lifetime of ctx), or NULL.
 */
const char *cg_perf_counter_name(const cg_perf_ctx_t *ctx, size_t index);

/**
 * Return the name of the active backend.
 *
 * @return Static string: "linux_perf", "zephyr_timing",
 *         "bare_metal_csr", "cuda_cupti", or "none".
 */
const char *cg_perf_backend_name(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* COMPGEN_PERF_COUNTERS_H_ */
