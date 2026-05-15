/*
 * Bare-metal smoke test for libcompgen_rt.
 *
 * Freestanding RISC-V test that exercises the core primitives
 * without any OS:
 *   - open a cpu_sync device
 *   - allocate two buffers
 *   - record a command buffer with a fill + copy
 *   - submit through queue_submit with wait/signal semaphores
 *   - verify the dst buffer content
 *   - exercise an event tensor notify/query path
 *
 * Runs on Spike via the Saturn OPU Spike harness. Exit status is
 * communicated through the HTIF ``tohost`` symbol (the Spike ABI):
 * a non-zero tohost write terminates the simulator with the encoded
 * status. 0 means pass.
 *
 * No printf, no libc — every comparison is a branch-and-fail path.
 */

#include "compgen_rt/compgen_rt.h"

#include <stdint.h>
#include <stddef.h>

/* HTIF tohost/fromhost pair — Spike's Syscall / exit interface. */
volatile uint64_t tohost   __attribute__((section(".tohost"))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"))) = 0;

static void htif_exit(uint64_t code) {
    /* Spike treats a non-zero write to tohost as an exit request when
     * bit 0 is set. Encode: ``(code << 1) | 1``. */
    tohost = (code << 1) | 1;
    for (;;) {
        /* Halt the hart on unexpected return. */
#if defined(__riscv)
        __asm__ volatile ("wfi");
#endif
    }
}

/* Minimal memset / memcmp since we're freestanding. The toolchain's
 * libgcc provides __memset_*, but for clarity we spell them out. */
static void *_memset(void *dst, int v, size_t n) {
    uint8_t *d = dst;
    for (size_t i = 0; i < n; ++i) d[i] = (uint8_t)v;
    return dst;
}

static int _memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *pa = a, *pb = b;
    for (size_t i = 0; i < n; ++i) {
        if (pa[i] != pb[i]) return (int)pa[i] - (int)pb[i];
    }
    return 0;
}

/* Every check returns a distinct exit code so Spike's final status
 * points directly at the failing assertion. */
#define CHECK(cond, code) do { if (!(cond)) htif_exit(code); } while (0)
#define REQUIRE(expr) CHECK((expr) == CG_RT_OK, __LINE__)

int main(void) {
    cg_rt_instance_t *inst = NULL;
    REQUIRE(cg_rt_instance_create("cpu_sync", &inst));
    cg_rt_device_t *dev = NULL;
    REQUIRE(cg_rt_device_open(inst, 0, &dev));

    cg_rt_device_traits_t traits;
    REQUIRE(cg_rt_device_query_traits(dev, &traits));
    CHECK(traits.device_class == CG_RT_DEVICE_CLASS_CPU, 2);
    CHECK(traits.supports_event_tensors,                  3);

    /* --- Buffer fill + copy through the queue ------------------- */
    cg_rt_buffer_t *src = NULL, *dst = NULL;
    REQUIRE(cg_rt_buffer_alloc(dev, 32, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &src));
    REQUIRE(cg_rt_buffer_alloc(dev, 32, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &dst));

    /* Seed src via map. */
    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(src, 0, 32, &ptr));
    _memset(ptr, 0xA5, 32);
    REQUIRE(cg_rt_buffer_unmap(src));

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(dev, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    REQUIRE(cg_rt_command_buffer_copy(cb, src, 0, dst, 0, 32));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(dev, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(dev, 0, NULL, 0, &sig, 1, cb));

    /* cpu_sync + bare: queue_submit completes synchronously on this
     * hart so the semaphore must be signalled by the time we poll. */
    uint64_t sem_val = 0;
    REQUIRE(cg_rt_semaphore_query(done, &sem_val));
    CHECK(sem_val == 1, 4);

    /* Validate dst. */
    REQUIRE(cg_rt_buffer_map(dst, 0, 32, &ptr));
    uint8_t expected[32];
    _memset(expected, 0xA5, 32);
    CHECK(_memcmp(ptr, expected, 32) == 0, 5);
    REQUIRE(cg_rt_buffer_unmap(dst));

    /* --- Event tensor notify + query ---------------------------- */
    int64_t shape[] = {4};
    cg_rt_event_tensor_t *et = NULL;
    REQUIRE(cg_rt_event_tensor_create(dev, 1, shape, CG_RT_EVENT_DTYPE_I64,
                                      3, &et));
    REQUIRE(cg_rt_event_tensor_notify(et, 2, 1));
    REQUIRE(cg_rt_event_tensor_notify(et, 2, 1));
    REQUIRE(cg_rt_event_tensor_notify(et, 2, 1));
    int64_t v = -1;
    REQUIRE(cg_rt_event_tensor_query(et, 2, &v));
    CHECK(v == 0, 6);
    /* Poll-wait must now return OK. */
    REQUIRE(cg_rt_event_tensor_wait(et, 2, CG_RT_TIMEOUT_POLL));

    /* Cleanup. */
    cg_rt_event_tensor_destroy(et);
    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_buffer_destroy(src);
    cg_rt_buffer_destroy(dst);
    cg_rt_device_close(dev);
    cg_rt_instance_destroy(inst);

    /* All checks passed. */
    htif_exit(0);
    return 0; /* unreachable */
}
