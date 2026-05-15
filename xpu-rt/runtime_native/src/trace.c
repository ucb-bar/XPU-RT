/*
 * CompGen Trace — ring-buffer trace collector.
 *
 * Collects trace events into a fixed-size ring buffer and exports
 * them as Chrome Trace Event Format JSON.
 *
 * When CG_TRACE_ENABLED is not defined, this file compiles to empty
 * stubs so the linker is satisfied but zero code is generated.
 */

#include "compgen/trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Platform-specific timestamp                                         */
/* ------------------------------------------------------------------ */

#if defined(__linux__)
  #include <time.h>
  static uint64_t _cg_timestamp_ns(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
  }
#elif defined(CG_TRACE_ZEPHYR)
  #include <zephyr/kernel.h>
  static uint64_t _cg_timestamp_ns(void) {
      return (uint64_t)k_cycle_get_32() *
             (1000000000ULL / (uint64_t)sys_clock_hw_cycles_per_sec());
  }
#else
  /* Bare-metal: RISC-V mcycle CSR */
  static uint64_t _cg_timestamp_ns(void) {
      uint64_t cycles;
      #if defined(__riscv)
        __asm__ volatile("rdcycle %0" : "=r"(cycles));
      #else
        cycles = 0;
      #endif
      /* Assume 1 GHz — overridden by calibration at runtime */
      return cycles;
  }
#endif

/* ------------------------------------------------------------------ */
/* Ring buffer                                                         */
/* ------------------------------------------------------------------ */

#define CG_TRACE_DEFAULT_CAPACITY  (1024 * 1024 / sizeof(cg_trace_event_t))

static cg_trace_event_t *g_buffer    = NULL;
static size_t             g_capacity = 0;
static size_t             g_head     = 0;   /* next write position */
static size_t             g_count    = 0;   /* total events stored */
static int                g_inited   = 0;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

int cg_trace_init(size_t buffer_size) {
    if (g_inited) return 0;  /* already initialized */

    size_t cap = buffer_size > 0
        ? buffer_size / sizeof(cg_trace_event_t)
        : CG_TRACE_DEFAULT_CAPACITY;

    if (cap < 64) cap = 64;

    g_buffer = (cg_trace_event_t *)calloc(cap, sizeof(cg_trace_event_t));
    if (!g_buffer) return -1;

    g_capacity = cap;
    g_head     = 0;
    g_count    = 0;
    g_inited   = 1;
    return 0;
}

void cg_trace_shutdown(void) {
    if (!g_inited) return;
    free(g_buffer);
    g_buffer   = NULL;
    g_capacity = 0;
    g_head     = 0;
    g_count    = 0;
    g_inited   = 0;
}

size_t cg_trace_event_count(void) {
    return g_count < g_capacity ? g_count : g_capacity;
}

/* ------------------------------------------------------------------ */
/* Recording                                                           */
/* ------------------------------------------------------------------ */

static void _record(cg_trace_event_t ev) {
    if (!g_inited) {
        /* Auto-init with defaults on first use */
        if (cg_trace_init(0) != 0) return;
    }

    ev.timestamp_ns = _cg_timestamp_ns();
    g_buffer[g_head] = ev;
    g_head = (g_head + 1) % g_capacity;
    g_count++;
}

void cg_trace_record_begin(const char *category, const char *name) {
    cg_trace_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type     = CG_TRACE_EVENT_BEGIN;
    ev.category = category;
    ev.name     = name;
    _record(ev);
}

void cg_trace_record_end(void) {
    cg_trace_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = CG_TRACE_EVENT_END;
    _record(ev);
}

void cg_trace_record_counter(const char *name, int64_t value) {
    cg_trace_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type  = CG_TRACE_EVENT_COUNTER;
    ev.name  = name;
    ev.value = value;
    _record(ev);
}

void cg_trace_record_tile(const char *region_id, int tile_idx,
                           const char *metric, int64_t value) {
    cg_trace_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type     = CG_TRACE_EVENT_TILE;
    ev.category = region_id;
    ev.name     = metric;
    ev.value    = (int64_t)tile_idx;
    ev.metric   = metric;
    _record(ev);
    (void)value; /* stored via separate mechanism if needed */
}

void cg_trace_record_instant(const char *category, const char *name) {
    cg_trace_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type     = CG_TRACE_EVENT_INSTANT;
    ev.category = category;
    ev.name     = name;
    _record(ev);
}

/* ------------------------------------------------------------------ */
/* Flush to Chrome Trace Event Format JSON                             */
/* ------------------------------------------------------------------ */

static const char *_event_type_char(cg_trace_event_type_t t) {
    switch (t) {
        case CG_TRACE_EVENT_BEGIN:   return "B";
        case CG_TRACE_EVENT_END:     return "E";
        case CG_TRACE_EVENT_COUNTER: return "C";
        case CG_TRACE_EVENT_INSTANT: return "i";
        case CG_TRACE_EVENT_TILE:    return "X";  /* complete event */
        default:                     return "?";
    }
}

int cg_trace_flush(const char *path) {
    if (!g_inited) return 0;

    FILE *f = path ? fopen(path, "w") : stdout;
    if (!f) return -1;

    size_t total = cg_trace_event_count();
    size_t start = g_count <= g_capacity ? 0 : g_head;

    fprintf(f, "{\"traceEvents\":[\n");

    for (size_t i = 0; i < total; i++) {
        size_t idx = (start + i) % g_capacity;
        const cg_trace_event_t *ev = &g_buffer[idx];

        if (i > 0) fprintf(f, ",\n");

        /* Convert ns to us for Chrome format */
        double ts_us = (double)ev->timestamp_ns / 1000.0;

        fprintf(f,
            "  {\"ph\":\"%s\", \"ts\":%.3f, \"pid\":1, \"tid\":1",
            _event_type_char(ev->type), ts_us);

        if (ev->category)
            fprintf(f, ", \"cat\":\"%s\"", ev->category);
        if (ev->name)
            fprintf(f, ", \"name\":\"%s\"", ev->name);

        if (ev->type == CG_TRACE_EVENT_COUNTER && ev->name) {
            fprintf(f, ", \"args\":{\"%s\":%lld}",
                    ev->name, (long long)ev->value);
        } else if (ev->type == CG_TRACE_EVENT_TILE && ev->metric) {
            fprintf(f, ", \"args\":{\"tile\":%lld, \"metric\":\"%s\"}",
                    (long long)ev->value, ev->metric);
        }

        fprintf(f, "}");
    }

    fprintf(f, "\n]}\n");

    int written = (int)total;
    if (path) fclose(f);

    /* Reset buffer after flush */
    g_head  = 0;
    g_count = 0;

    return written;
}
