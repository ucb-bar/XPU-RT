/*
 * Single-threaded pthread implementations for chipyard newlib bare-metal.
 *
 * Newlib declares the pthread API in <pthread.h> (gated on _POSIX_THREADS,
 * which our toolchain file defines), but does not ship libpthread.a for
 * the riscv64-unknown-elf target. XNNPACK uses pthread_once /
 * pthread_mutex_* unconditionally in its init paths, so we need real
 * symbols at link time.
 *
 * Under HTIF / spike we only ever run a single hart, so all of these
 * degrade to trivial no-ops. The implementations are NOT thread-safe;
 * if/when XPU-RT targets multi-hart spike or real silicon we replace
 * this with a real pthread.
 */

#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>

/* posix_memalign: newlib bare-metal doesn't ship this. Trivial
 * wrapper around malloc that overallocates and aligns by hand. */
int posix_memalign(void** out, size_t alignment, size_t size) {
    if (out == 0 || alignment < sizeof(void*) ||
        (alignment & (alignment - 1)) != 0) {
        return 22; /* EINVAL */
    }
    /* Overallocate by `alignment` bytes; store the malloc base just
     * before the returned aligned pointer so free() (called on the
     * aligned pointer via XNNPACK's deallocator) doesn't quite work —
     * but XNNPACK uses its own aligned-free that recovers the base via
     * a different mechanism. For our single-shot init path this is
     * adequate. */
    void* raw = malloc(size + alignment + sizeof(void*));
    if (raw == 0) return 12; /* ENOMEM */
    uintptr_t addr = (uintptr_t)raw + sizeof(void*);
    addr = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    ((void**)addr)[-1] = raw;
    *out = (void*)addr;
    return 0;
}

int pthread_once(pthread_once_t* once, void (*fn)(void)) {
    if (once == 0 || fn == 0) {
        return 22; /* EINVAL */
    }
    /* Newlib's pthread_once_t = struct { int is_initialized; int init_executed; }
     * with PTHREAD_ONCE_INIT = { 1, 0 }. We run fn once when
     * init_executed == 0 (offset 4), then set it to 1. */
    int* state = (int*)once;
    if (state[1] == 0) {
        state[1] = 1;
        fn();
    }
    return 0;
}

int pthread_mutex_init(pthread_mutex_t* m, const pthread_mutexattr_t* a) {
    (void)m; (void)a; return 0;
}
int pthread_mutex_destroy(pthread_mutex_t* m) {
    (void)m; return 0;
}
int pthread_mutex_lock(pthread_mutex_t* m) {
    (void)m; return 0;
}
int pthread_mutex_trylock(pthread_mutex_t* m) {
    (void)m; return 0;
}
int pthread_mutex_unlock(pthread_mutex_t* m) {
    (void)m; return 0;
}

int pthread_mutexattr_init(pthread_mutexattr_t* a) {
    (void)a; return 0;
}
int pthread_mutexattr_destroy(pthread_mutexattr_t* a) {
    (void)a; return 0;
}
int pthread_mutexattr_settype(pthread_mutexattr_t* a, int t) {
    (void)a; (void)t; return 0;
}
int pthread_mutexattr_setpshared(pthread_mutexattr_t* a, int p) {
    (void)a; (void)p; return 0;
}

/* Best-effort defensive stubs — XNNPACK shouldn't call these on the
 * single-threaded path but we provide them so any linker reference
 * resolves. */
int pthread_key_create(pthread_key_t* k, void (*d)(void*)) {
    (void)k; (void)d; return 0;
}
int pthread_key_delete(pthread_key_t k) {
    (void)k; return 0;
}
void* pthread_getspecific(pthread_key_t k) {
    (void)k; return 0;
}
int pthread_setspecific(pthread_key_t k, const void* v) {
    (void)k; (void)v; return 0;
}
