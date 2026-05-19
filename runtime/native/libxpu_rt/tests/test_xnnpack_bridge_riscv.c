/*
 * End-to-end smoke test for the XNNPACK bridge running on
 * Chipyard's Spike (riscv64 + RVV-128), bare-metal HTIF.
 *
 * Built by clang 18 (Merlin's IREE bundle) against Chipyard's newlib
 * sysroot. Linked by Chipyard's riscv64-unknown-elf-gcc with
 * `-specs=htif_nano.specs` so newlib's HTIF stubs (`_write`, `_exit`,
 * crt0) resolve correctly. printf and exit therefore go through
 * Spike's HTIF tohost/fromhost mechanism with no manual plumbing.
 *
 * Run:
 *
 *   spike --isa=rv64gcv test_xnnpack_bridge_riscv.elf
 *
 * The program drives one fully-connected f32 op through the bridge,
 * prints each output float and a 64-bit FNV-1a checksum, and exits 0
 * on success or a distinct non-zero code per failed assertion.
 */

#include "drivers/xnnpack/xnnpack_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN_CHANNELS  4
#define OUT_CHANNELS 3
#define BATCH        1

static uint64_t fnv1a_u64(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main(void) {
    puts("riscv-spike xnnpack bridge smoke: begin");

    int rc = xpu_rt_xnn_global_initialize();
    if (rc != XPU_RT_XNN_OK) {
        const char* err = xpu_rt_xnn_last_error();
        printf("FAIL: global_initialize rc=%d msg=%s\n",
               rc, err ? err : "(none)");
        return 1;
    }
    puts("xnn init: OK");

    static float weights_and_bias[OUT_CHANNELS * IN_CHANNELS + OUT_CHANNELS] = {
        /* row 0 */  0.1f,  0.2f,  0.3f,  0.4f,
        /* row 1 */ -0.1f,  0.5f,  0.0f,  0.1f,
        /* row 2 */  1.0f, -1.0f,  0.5f, -0.5f,
        /* bias */   0.01f, 0.02f, 0.03f,
    };

    int64_t shape[2]   = { IN_CHANNELS, OUT_CHANNELS };
    int32_t ints[1]    = { 0 };
    float   floats[2]  = { -1.0e30f, 1.0e30f };

    xpu_rt_xnn_op* op = xpu_rt_xnn_create(
        XPU_RT_XNN_OP_FULLY_CONNECTED_F32,
        shape, 2,
        ints,  1,
        floats, 2,
        weights_and_bias,
        sizeof(weights_and_bias));
    if (op == NULL) {
        const char* err = xpu_rt_xnn_last_error();
        printf("FAIL: create (%s)\n", err ? err : "(no error msg)");
        return 2;
    }
    puts("xnn create: OK");

    static float input[IN_CHANNELS]   = { 1.0f, 2.0f, 3.0f, 4.0f };
    static float output[OUT_CHANNELS] = { 0.0f, 0.0f, 0.0f };

    int64_t rshape[1] = { BATCH };
    const void* inputs[1]  = { input };
    void*       outputs[1] = { output };

    rc = xpu_rt_xnn_reshape_setup(op, rshape, 1, inputs, 1, outputs, 1);
    if (rc != XPU_RT_XNN_OK) {
        const char* err = xpu_rt_xnn_last_error();
        printf("FAIL: reshape_setup (%s)\n", err ? err : "(no error msg)");
        return 3;
    }
    rc = xpu_rt_xnn_run(op);
    if (rc != XPU_RT_XNN_OK) {
        const char* err = xpu_rt_xnn_last_error();
        printf("FAIL: run (%s)\n", err ? err : "(no error msg)");
        return 4;
    }
    puts("xnn run: OK");

    /* Print outputs as hex-bits so the host wrapper can decode them
     * losslessly without depending on the device's printf %.9g. */
    uint32_t o0, o1, o2;
    memcpy(&o0, &output[0], sizeof(o0));
    memcpy(&o1, &output[1], sizeof(o1));
    memcpy(&o2, &output[2], sizeof(o2));
    printf("output: 0x%08x 0x%08x 0x%08x\n",
           (unsigned)o0, (unsigned)o1, (unsigned)o2);
    /* newlib's nano printf doesn't handle %llx — split the 64-bit
     * checksum into two 32-bit halves for portable output. */
    uint64_t cs = fnv1a_u64(output, sizeof(output));
    printf("checksum: 0x%08lx%08lx\n",
           (unsigned long)(cs >> 32),
           (unsigned long)(cs & 0xffffffffUL));
    puts("PASS");

    xpu_rt_xnn_destroy(op);
    xpu_rt_xnn_global_deinitialize();
    return 0;
}
