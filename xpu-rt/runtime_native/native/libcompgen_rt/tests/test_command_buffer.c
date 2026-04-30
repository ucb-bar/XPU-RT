/*
 * Command buffer recording + replay via cpu_sync queue submit.
 */

#include "compgen_rt/compgen_rt.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static cg_rt_instance_t *g_instance = NULL;
static cg_rt_device_t   *g_device   = NULL;

static void setup_device(void) {
    if (g_instance != NULL) return;
    cg_rt_instance_create("cpu_sync", &g_instance);
    cg_rt_device_open(g_instance, 0, &g_device);
}

TEST_CASE(cb_begin_end_required, "record calls outside begin/end are rejected") {
    setup_device();
    cg_rt_buffer_t *a = NULL, *b = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &a));
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &b));
    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));

    /* Before begin — should fail precondition. */
    EXPECT_EQ(cg_rt_command_buffer_copy(cb, a, 0, b, 0, 4),
              CG_RT_ERR_FAILED_PRECOND);

    REQUIRE(cg_rt_command_buffer_begin(cb));
    REQUIRE(cg_rt_command_buffer_copy(cb, a, 0, b, 0, 4));
    REQUIRE(cg_rt_command_buffer_end(cb));

    /* After end — also rejected. */
    EXPECT_EQ(cg_rt_command_buffer_copy(cb, a, 0, b, 0, 4),
              CG_RT_ERR_FAILED_PRECOND);

    cg_rt_command_buffer_destroy(cb);
    cg_rt_buffer_destroy(a);
    cg_rt_buffer_destroy(b);
}

TEST_CASE(cb_copy_executes, "queue_submit runs a recorded copy") {
    setup_device();
    cg_rt_buffer_t *src = NULL, *dst = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &src));
    REQUIRE(cg_rt_buffer_alloc(g_device, 16, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &dst));

    /* Seed src with distinctive bytes. */
    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(src, 0, 16, &ptr));
    memset(ptr, 0xAB, 16);
    cg_rt_buffer_unmap(src);

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    REQUIRE(cg_rt_command_buffer_copy(cb, src, 0, dst, 0, 16));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &sig, 1, cb));

    EXPECT_EQ(cg_rt_semaphore_wait(done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    /* Validate dst. */
    REQUIRE(cg_rt_buffer_map(dst, 0, 16, &ptr));
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(((uint8_t *)ptr)[i], 0xAB);
    }
    cg_rt_buffer_unmap(dst);

    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_buffer_destroy(src);
    cg_rt_buffer_destroy(dst);
}

TEST_CASE(cb_fill_executes, "queue_submit runs a recorded fill") {
    setup_device();
    cg_rt_buffer_t *buf = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 32, CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_TRANSFER, &buf));

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    REQUIRE(cg_rt_command_buffer_fill(cb, buf, 0, 32, 0x11223344u));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &sig, 1, cb));
    EXPECT_EQ(cg_rt_semaphore_wait(done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(buf, 0, 32, &ptr));
    uint32_t *as_u32 = ptr;
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(as_u32[i], 0x11223344u);
    }
    cg_rt_buffer_unmap(buf);

    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_buffer_destroy(buf);
}

/* ---- dispatch test: a simple element-wise add executable ---- */

static int add_kernel(const void *pc,
                      size_t pc_size,
                      void  **bindings,
                      const size_t *binding_sizes,
                      size_t n_bindings) {
    if (n_bindings != 3 || pc_size != sizeof(uint32_t)) {
        return 1;
    }
    uint32_t n = *(const uint32_t *)pc;
    const float *a = bindings[0];
    const float *b = bindings[1];
    float       *c = bindings[2];
    /* Bounds check via binding_sizes. */
    if (binding_sizes[2] < n * sizeof(float)) return 2;
    for (uint32_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
    return 0;
}

TEST_CASE(cb_dispatch_executes, "queue_submit invokes a registered CPU kernel") {
    setup_device();
    enum { N = 8 };
    cg_rt_buffer_t *ba = NULL, *bb = NULL, *bc = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, N * sizeof(float), CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_DISPATCH, &ba));
    REQUIRE(cg_rt_buffer_alloc(g_device, N * sizeof(float), CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_DISPATCH, &bb));
    REQUIRE(cg_rt_buffer_alloc(g_device, N * sizeof(float), CG_RT_MEMORY_SPACE_HOST,
                               CG_RT_BUFFER_USAGE_DISPATCH, &bc));

    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(ba, 0, N * sizeof(float), &ptr));
    float *fa = ptr;
    for (int i = 0; i < N; ++i) fa[i] = (float)i;
    cg_rt_buffer_unmap(ba);

    REQUIRE(cg_rt_buffer_map(bb, 0, N * sizeof(float), &ptr));
    float *fb = ptr;
    for (int i = 0; i < N; ++i) fb[i] = 100.0f;
    cg_rt_buffer_unmap(bb);

    cg_rt_executable_t *exe = NULL;
    REQUIRE(cg_rt_executable_create_cpu(g_device, add_kernel, &exe));

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    uint32_t n = N;
    cg_rt_buffer_t *bindings[] = { ba, bb, bc };
    REQUIRE(cg_rt_command_buffer_dispatch(cb, exe, &n, sizeof(n),
                                          bindings, 3));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &sig, 1, cb));
    EXPECT_EQ(cg_rt_semaphore_wait(done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    REQUIRE(cg_rt_buffer_map(bc, 0, N * sizeof(float), &ptr));
    float *fc = ptr;
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(fc[i] == (float)i + 100.0f);
    }
    cg_rt_buffer_unmap(bc);

    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_executable_destroy(exe);
    cg_rt_buffer_destroy(ba);
    cg_rt_buffer_destroy(bb);
    cg_rt_buffer_destroy(bc);
}

int main(void) {
    setup_device();
    int rc = run_tests();
    cg_rt_device_close(g_device);
    cg_rt_instance_destroy(g_instance);
    return rc;
}
