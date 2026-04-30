/*
 * CUDA driver — end-to-end sanity on real hardware.
 *
 * Skipped at runtime if no CUDA device is available.  Exercises:
 *   - cg_rt_instance_create("cuda")
 *   - device_open + traits
 *   - managed-memory buffer alloc / map
 *   - command buffer with a fill (translates to cuMemsetD32Async)
 *   - queue submit synchronous path
 *   - NVRTC compile + dispatch of a trivial CUDA kernel
 */

#include "compgen_rt/compgen_rt.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static cg_rt_instance_t *g_instance = NULL;
static cg_rt_device_t   *g_device   = NULL;
static int g_cuda_available = -1; /* -1 unknown, 0 no, 1 yes */

static int cuda_available(void) {
    if (g_cuda_available != -1) return g_cuda_available;
    cg_rt_instance_t *tmp_inst = NULL;
    if (cg_rt_instance_create("cuda", &tmp_inst) != CG_RT_OK) {
        g_cuda_available = 0;
        return 0;
    }
    cg_rt_device_t *tmp_dev = NULL;
    cg_rt_status_t rc = cg_rt_device_open(tmp_inst, 0, &tmp_dev);
    if (rc != CG_RT_OK) {
        cg_rt_instance_destroy(tmp_inst);
        g_cuda_available = 0;
        return 0;
    }
    g_instance = tmp_inst;
    g_device = tmp_dev;
    g_cuda_available = 1;
    return 1;
}

#define SKIP_IF_NO_CUDA()                                                   \
    do {                                                                    \
        if (!cuda_available()) {                                            \
            fprintf(stderr, "  (no CUDA device available, skipping)\n");    \
            return;                                                         \
        }                                                                   \
    } while (0)

TEST_CASE(cuda_traits_report_gpu, "CUDA driver reports GPU class + NVIDIA vendor") {
    SKIP_IF_NO_CUDA();
    cg_rt_device_traits_t t;
    REQUIRE(cg_rt_device_query_traits(g_device, &t));
    EXPECT_EQ(t.device_class, CG_RT_DEVICE_CLASS_GPU);
    EXPECT_EQ(strcmp(t.vendor, "nvidia"), 0);
    EXPECT_TRUE(t.max_device_memory_bytes > 0);
    EXPECT_TRUE(t.supports_command_buffers);
    EXPECT_TRUE(t.supports_graph_capture);
    EXPECT_TRUE(t.max_concurrent_queues >= 2);
}

TEST_CASE(cuda_buffer_is_host_visible, "managed-memory buffer is host-addressable") {
    SKIP_IF_NO_CUDA();
    cg_rt_buffer_t *buf = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 256, CG_RT_MEMORY_SPACE_DEVICE,
                               CG_RT_BUFFER_USAGE_TRANSFER, &buf));
    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(buf, 0, 256, &ptr));
    /* Zero-init contract. */
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(((uint8_t *)ptr)[i], 0);
    }
    /* Host writes should be visible to subsequent reads. */
    memset(ptr, 0x7F, 256);
    cg_rt_buffer_unmap(buf);

    REQUIRE(cg_rt_buffer_map(buf, 0, 256, &ptr));
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(((uint8_t *)ptr)[i], 0x7F);
    }
    cg_rt_buffer_unmap(buf);
    cg_rt_buffer_destroy(buf);
}

TEST_CASE(cuda_fill_runs_on_stream, "fill op executes via cuMemsetD32Async") {
    SKIP_IF_NO_CUDA();
    cg_rt_buffer_t *buf = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 64, CG_RT_MEMORY_SPACE_DEVICE,
                               CG_RT_BUFFER_USAGE_TRANSFER, &buf));

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    REQUIRE(cg_rt_command_buffer_fill(cb, buf, 0, 64, 0x12345678u));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &sig, 1, cb));
    EXPECT_EQ(cg_rt_semaphore_wait(done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(buf, 0, 64, &ptr));
    uint32_t *u32 = ptr;
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(u32[i], 0x12345678u);
    }
    cg_rt_buffer_unmap(buf);

    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_buffer_destroy(buf);
}

/* A real NVRTC-compiled kernel: scalar add-one over 32 int32 elements.
 * The grid/block descriptor is encoded in the push_constants block
 * per the public contract. */
static const char *kAddOneSource =
    "extern \"C\" __global__ void add_one(int *data) {\n"
    "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i < 32) data[i] += 1;\n"
    "}\n";

TEST_CASE(cuda_nvrtc_dispatch, "NVRTC-compiled kernel runs via cuLaunchKernel") {
    SKIP_IF_NO_CUDA();

    cg_rt_executable_t *exe = NULL;
    cg_rt_status_t rc = cg_rt_executable_create_cuda_ptx(
        g_device, kAddOneSource, "add_one", &exe);
    REQUIRE(rc);

    cg_rt_buffer_t *buf = NULL;
    REQUIRE(cg_rt_buffer_alloc(g_device, 32 * sizeof(int32_t),
                               CG_RT_MEMORY_SPACE_DEVICE,
                               CG_RT_BUFFER_USAGE_DISPATCH, &buf));

    void *ptr = NULL;
    REQUIRE(cg_rt_buffer_map(buf, 0, 32 * sizeof(int32_t), &ptr));
    for (int i = 0; i < 32; ++i) ((int32_t *)ptr)[i] = 10 + i;
    cg_rt_buffer_unmap(buf);

    /* Launch descriptor: grid=(1,1,1), block=(32,1,1). */
    uint32_t cfg[6] = { 1, 1, 1, 32, 1, 1 };

    cg_rt_command_buffer_t *cb = NULL;
    REQUIRE(cg_rt_command_buffer_create(g_device, &cb));
    REQUIRE(cg_rt_command_buffer_begin(cb));
    cg_rt_buffer_t *bindings[] = { buf };
    REQUIRE(cg_rt_command_buffer_dispatch(cb, exe, cfg, sizeof(cfg),
                                          bindings, 1));
    REQUIRE(cg_rt_command_buffer_end(cb));

    cg_rt_semaphore_t *done = NULL;
    REQUIRE(cg_rt_semaphore_create(g_device, 0, &done));
    cg_rt_semaphore_point_t sig = { .semaphore = done, .value = 1 };
    REQUIRE(cg_rt_queue_submit(g_device, 0, NULL, 0, &sig, 1, cb));
    EXPECT_EQ(cg_rt_semaphore_wait(done, 1, CG_RT_TIMEOUT_INFINITE), CG_RT_OK);

    REQUIRE(cg_rt_buffer_map(buf, 0, 32 * sizeof(int32_t), &ptr));
    int32_t *as_i32 = ptr;
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(as_i32[i], 10 + i + 1);
    }
    cg_rt_buffer_unmap(buf);

    cg_rt_semaphore_destroy(done);
    cg_rt_command_buffer_destroy(cb);
    cg_rt_executable_destroy(exe);
    cg_rt_buffer_destroy(buf);
}

int main(void) {
    int rc = run_tests();
    if (g_device != NULL) cg_rt_device_close(g_device);
    if (g_instance != NULL) cg_rt_instance_destroy(g_instance);
    return rc;
}
