/*
 * Minimal stdio stubs for picolibc bare-metal.
 *
 * picolibc's libc.a contains a few translation units (e.g. assert
 * helpers, abort) that reference `stderr` / `stdout` / `__fprintf_chk`
 * unconditionally. Under the bare-metal HTIF flow we don't link
 * picolibc's stdio init, so these become undefined references.
 *
 * We provide weak fallback symbols here. They are never actually used
 * at runtime — every code path that would hit them is excluded by
 * `-DNDEBUG` or by HTIF tohost direct I/O — but the link must
 * resolve.
 */

#include <stddef.h>

/* picolibc FILE pointer — opaque, never dereferenced because we never
 * call into stdio. Defined as a null pointer in the .data section so
 * any accidental use crashes early rather than silently. */
void* stdout = NULL;
void* stderr = NULL;
void* stdin  = NULL;

int fprintf(void* stream, const char* fmt, ...) {
    (void)stream; (void)fmt;
    return 0;
}

int fputs(const char* s, void* stream) {
    (void)s; (void)stream;
    return 0;
}

int fputc(int c, void* stream) {
    (void)c; (void)stream;
    return 0;
}

int fflush(void* stream) {
    (void)stream;
    return 0;
}

void __assert_func(const char* file, int line, const char* fn, const char* expr) {
    (void)file; (void)line; (void)fn; (void)expr;
    while (1) { /* halt */ }
}

void __assert_no_args(void) {
    while (1) { /* halt */ }
}
