/*
 * CompGen Trace API — compile-time gated instrumentation.
 *
 * When CG_TRACE_ENABLED is defined, trace macros expand to ring-buffer
 * recording calls.  Otherwise they compile to nothing (zero overhead).
 *
 * On Zephyr targets, CG_TRACE_ZEPHYR routes through sys_trace_*.
 * On Linux/bare-metal, the built-in ring buffer collects Chrome Trace
 * Event Format JSON.
 *
 * Usage:
 *   CG_TRACE_BEGIN("dispatch", "matmul_tile_0");
 *   ... kernel dispatch ...
 *   CG_TRACE_END();
 *
 *   CG_TRACE_COUNTER("dma_bytes", 4096);
 *   CG_TRACE_TILE("region_3", 7, "latency_us", 42);
 */

#ifndef COMPGEN_TRACE_H_
#define COMPGEN_TRACE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Trace event types                                                   */
/* ------------------------------------------------------------------ */

typedef enum cg_trace_event_type {
    CG_TRACE_EVENT_BEGIN   = 0,
    CG_TRACE_EVENT_END     = 1,
    CG_TRACE_EVENT_COUNTER = 2,
    CG_TRACE_EVENT_TILE    = 3,
    CG_TRACE_EVENT_INSTANT = 4,
} cg_trace_event_type_t;

/* ------------------------------------------------------------------ */
/* Trace event record (stored in ring buffer)                          */
/* ------------------------------------------------------------------ */

typedef struct cg_trace_event {
    cg_trace_event_type_t type;
    uint64_t              timestamp_ns;
    const char           *category;    /* interned string, not owned */
    const char           *name;        /* interned string, not owned */
    int64_t               value;       /* counter value or tile index */
    const char           *metric;      /* for TILE events */
} cg_trace_event_t;

/* ------------------------------------------------------------------ */
/* Trace session lifecycle                                             */
/* ------------------------------------------------------------------ */

/**
 * Initialize the trace subsystem.
 *
 * @param buffer_size  Ring buffer capacity in bytes.  0 = default (1 MB).
 * @return 0 on success, non-zero on failure.
 */
int cg_trace_init(size_t buffer_size);

/**
 * Shut down the trace subsystem and free the ring buffer.
 */
void cg_trace_shutdown(void);

/**
 * Flush all buffered events to the given file path as Chrome Trace
 * Event Format JSON.
 *
 * @param path  Output file path.  NULL = stdout.
 * @return Number of events written, or -1 on error.
 */
int cg_trace_flush(const char *path);

/**
 * Return the number of events currently in the ring buffer.
 */
size_t cg_trace_event_count(void);

/* ------------------------------------------------------------------ */
/* Recording functions (called by macros)                              */
/* ------------------------------------------------------------------ */

void cg_trace_record_begin(const char *category, const char *name);
void cg_trace_record_end(void);
void cg_trace_record_counter(const char *name, int64_t value);
void cg_trace_record_tile(const char *region_id, int tile_idx,
                           const char *metric, int64_t value);
void cg_trace_record_instant(const char *category, const char *name);

/* ------------------------------------------------------------------ */
/* Macros — compile to no-ops unless CG_TRACE_ENABLED                  */
/* ------------------------------------------------------------------ */

#ifdef CG_TRACE_ENABLED

  #ifdef CG_TRACE_ZEPHYR
    /* Route through Zephyr tracing subsystem */
    #include <zephyr/tracing/tracing.h>
    #define CG_TRACE_BEGIN(cat, name) \
        do { sys_trace_idle(); cg_trace_record_begin((cat), (name)); } while (0)
    #define CG_TRACE_END() \
        do { cg_trace_record_end(); } while (0)
  #else
    #define CG_TRACE_BEGIN(cat, name)  cg_trace_record_begin((cat), (name))
    #define CG_TRACE_END()             cg_trace_record_end()
  #endif

  #define CG_TRACE_COUNTER(name, val) \
      cg_trace_record_counter((name), (int64_t)(val))
  #define CG_TRACE_TILE(region, tile, metric, val) \
      cg_trace_record_tile((region), (tile), (metric), (int64_t)(val))
  #define CG_TRACE_INSTANT(cat, name) \
      cg_trace_record_instant((cat), (name))

#else /* CG_TRACE_ENABLED not defined */

  #define CG_TRACE_BEGIN(cat, name)         ((void)0)
  #define CG_TRACE_END()                    ((void)0)
  #define CG_TRACE_COUNTER(name, val)       ((void)0)
  #define CG_TRACE_TILE(region, tile, m, v) ((void)0)
  #define CG_TRACE_INSTANT(cat, name)       ((void)0)

#endif /* CG_TRACE_ENABLED */

/* ------------------------------------------------------------------ */
/* Scoped trace helper (C11 / GCC cleanup attribute)                   */
/* ------------------------------------------------------------------ */

#if defined(CG_TRACE_ENABLED) && (defined(__GNUC__) || defined(__clang__))
  static inline void _cg_trace_scope_cleanup(const char **unused) {
      (void)unused;
      cg_trace_record_end();
  }
  #define CG_TRACE_SCOPE(cat, name) \
      CG_TRACE_BEGIN(cat, name); \
      const char *_cg_scope_##__LINE__ \
          __attribute__((cleanup(_cg_trace_scope_cleanup), unused)) = (name)
#else
  #define CG_TRACE_SCOPE(cat, name) ((void)0)
#endif

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* COMPGEN_TRACE_H_ */
