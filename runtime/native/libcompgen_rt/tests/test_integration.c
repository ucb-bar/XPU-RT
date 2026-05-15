/*
 * End-to-end: two queues with a producer/consumer semaphore chain.
 *
 * Queue 0 runs a fill; Queue 1 waits on the fill semaphore, then runs
 * a copy. Exercises: command buffers, buffers, cross-queue semaphore
 * handoff, driver traits reporting.
 */

#include "compgen_rt/compgen_rt.h"
#include "test_harness.h"

#include <stdint.h>
#include <string.h>

static cg_rt_instance_t *g_instance = NULL;
static cg_rt_device_t   *g_device   = NULL;

static void setup_device(void) {
    if (g_instance != NULL) return;
    cg_rt_instance_create("cpu_sync", &g_instance);
    cg_rt_device_open(g_instance, 0, &g_device);
}

TEST_CASE(int_traits_report_cpu_class, "cpu_sync reports the expected trait vector") {
    setup_device();
    cg_rt_device_traits_t t;
    REQUIRE(cg_rt_device_query_traits(g_device, &t));
    EXPECT_EQ(t.device_class, CG_RT_DEVICE_CLASS_CPU);
    EXPECT_TRUE(t.has_native_timeline_semaphores);
    EXPECT_TRUE(t.has_global_atomics);
    EXPECT_TRUE(t.supports_event_tensors);
    EXPECT_TRUE(t.supports_command_buffers);
    EXPECT_TRUE(t.max_concurrent_queues >= 2);
    EXPECT_EQ(strcmp(t.vendor, "host"), 0);
}

TEST_CASE(int_two_queue_chain, "cross-queue producer/consumer via semaphore") {
    setup_device();

    cg_rt_buffer_t *mid = NULL, *out = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &mid));
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &out));

    /* Queue 0: fill mid with 0xCAFEBABE. Signal ``fill_done``=1. */
    cg_rt_command_buffer_t *cb0 = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb0));
    REQUIRE(cg_rt_command_buffer_begin(cb0));
    REQUIRE(cg_rt_command_buffer_fill(cb0, mid, 0, 16, 0xCAFEBABEu));
    REQUIRE(cg_rt_command_buffer_end(cb0));

    /* Queue 1: copy mid -> out. Waits on ``fill_done``=1, signals
     * ``copy_done``=1 on completion. */
    cg_rt_command_buffer_t *cb1 = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb1));
    REQUIRE(cg_rt_command_buffer_begin(cb1));
    REQUIRE(cg_rt_command_buffer_copy(cb1, mid, 0, out, 0, 16));
    REQUIRE(cg_rt_command_buffer_end(cb1));

    cg_rt_semaphore_t *fill_done = NULL;
    cg_rt_semaphore_t *copy_done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &fill_done));
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &copy_done));

    /* On cpu_sync both submits are synchronous; we submit queue 0
     * first so the wait on queue 1 is already satisfied. Even so the
     * wait+signal contract must be honoured. */
    cg_rt_semaphore_point_t fill_sig = { .semaphore = fill_done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &fill_sig, 1, cb0));

    cg_rt_semaphore_point_t copy_wait = { .semaphore = fill_done, .value = 1 };
    cg_rt_semaphore_point_t copy_sig  = { .semaphore = copy_done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 1, &copy_wait, 1, &copy_sig, 1, cb1));

    EXPECT_EQ(cg_rt_semaphore_wait(copy_done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    /* Validate. */
    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(out, 0, 16, &ptr));
    uint32_t *u32 = ptr;
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(u32[i], 0xCAFEBABEu);
    }
    cg_rt_buffer_unmap(out);

    cg_rt_semaphore_destroy(fill_done);
    cg_rt_semaphore_destroy(copy_done);
    cg_rt_command_buffer_destroy(cb0);
    cg_rt_command_buffer_destroy(cb1);
    cg_rt_buffer_destroy(mid);
    cg_rt_buffer_destroy(out);
}

int main(void) {
    setup_device();
    int rc = run_tests();
    cg_rt_device_close(g_device);
    cg_rt_instance_destroy(g_instance);
    return rc;
}
