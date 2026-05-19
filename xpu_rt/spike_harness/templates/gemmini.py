"""Gemmini-specific ``init.c`` + ``driver.c`` templates.

Routed by :mod:`xpu_rt.spike_harness.templates` based on
``target_id``. The protocol stays uniform across targets — the
driver always prints ``mismatches=N/M`` + ``cycles=N`` — so the
same evaluator parser works regardless of which template fired.

Cycle source: ``MAIN_LD_ST_EX_CYCLES`` from
``include/gemmini_counter.h``.
"""

from __future__ import annotations

from pathlib import Path


_INIT_C = """\
// Starter scalar kernel for vanilla-KB-on-Gemmini. KB's RL loop rewrites
// the body of launch_gpu_implementation to call Gemmini intrinsics.
//
// Contract: int8 inputs, int32 accumulator. Output is full-precision i32.
//
// Vanilla KB's prompts expect this exact signature:
//   void launch_gpu_implementation(void *output,
//                                  void *input_A,
//                                  void *input_B,
//                                  int64_t M, int64_t K, int64_t N)
// — the agent's rewrites must preserve it (it's what the driver calls).

#include <stdint.h>
#include <stddef.h>
#include "include/gemmini.h"

void launch_gpu_implementation(void *output,
                               void *input_A,
                               void *input_B,
                               int64_t M, int64_t K, int64_t N) {
    const int8_t  *A = (const int8_t  *)input_A;
    const int8_t  *B = (const int8_t  *)input_B;
    int32_t       *C = (int32_t       *)output;

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}
"""


def render_init_c() -> str:
    return _INIT_C


_DRIVER_C = """\
// Harness for vanilla-KB-on-Gemmini. Compiles together with init.c
// (which provides launch_gpu_implementation). Reports correctness +
// Gemmini cycle counter on stdout in a format KB's parser can
// recognise.
//
// Outputs (in order, one per line):
//   M=%lld K=%lld N=%lld ops=%lld
//   mismatches=%d/%d
//   first_diff_at=%d ref=%d got=%d        (only when mismatches>0)
//   cycles=%lld
//   speedup_baseline_us=%lld              (placeholder — vanilla KB
//                                          expects a per-iteration time;
//                                          we report Gemmini cycles.)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "include/gemmini.h"
#include "include/gemmini_counter.h"

// Per-contract shape constants. Named TEST_M/K/N so they don't clash
// with launch_gpu_implementation's int64_t M, K, N parameters.
#define TEST_M @@M@@LL
#define TEST_K @@K@@LL
#define TEST_N @@N@@LL

extern void launch_gpu_implementation(void *output,
                                      void *input_A,
                                      void *input_B,
                                      int64_t M, int64_t K, int64_t N);

static uint32_t _lcg(uint32_t *s) {
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}

static void fill_i8(int8_t *p, int64_t n, uint32_t seed) {
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) p[i] = (int8_t)((_lcg(&s) >> 16) & 0xFF);
}

static void scalar_ref(const int8_t *A, const int8_t *B, int32_t *C) {
    for (int64_t m = 0; m < TEST_M; ++m)
        for (int64_t n = 0; n < TEST_N; ++n) {
            int32_t acc = 0;
            for (int64_t k = 0; k < TEST_K; ++k)
                acc += (int32_t)A[m*TEST_K + k] * (int32_t)B[k*TEST_N + n];
            C[m*TEST_N + n] = acc;
        }
}

int main(void) {
    static int8_t  A[TEST_M * TEST_K];
    static int8_t  B[TEST_K * TEST_N];
    static int32_t Cref[TEST_M * TEST_N];
    static int32_t Ctest[TEST_M * TEST_N];

    fill_i8(A, TEST_M * TEST_K, 0xC0FFEE);
    fill_i8(B, TEST_K * TEST_N, 0xBEEF0042);

    scalar_ref(A, B, Cref);

    gemmini_flush(0);
    counter_configure(0, MAIN_LD_ST_EX_CYCLES);
    counter_snapshot_reset();
    int64_t cycles_before = counter_read(0);

    launch_gpu_implementation((void *)Ctest, (void *)A, (void *)B, TEST_M, TEST_K, TEST_N);

    gemmini_fence();
    counter_snapshot_take();
    int64_t cycles_after = counter_read(0);
    int64_t cycles = cycles_after - cycles_before;

    int mismatches = 0;
    int total = TEST_M * TEST_N;
    int first_idx = -1;
    int32_t first_ref = 0, first_got = 0;
    for (int64_t i = 0; i < (int64_t)total; ++i) {
        if (Cref[i] != Ctest[i]) {
            if (first_idx < 0) {
                first_idx = (int)i;
                first_ref = Cref[i];
                first_got = Ctest[i];
            }
            ++mismatches;
        }
    }

    printf("M=%lld K=%lld N=%lld ops=%lld\\n",
           (long long)TEST_M, (long long)TEST_K, (long long)TEST_N,
           (long long)(TEST_M * TEST_N * TEST_K * 2));
    printf("mismatches=%d/%d\\n", mismatches, total);
    if (first_idx >= 0)
        printf("first_diff_at=%d ref=%d got=%d\\n", first_idx, first_ref, first_got);
    printf("cycles=%lld\\n", (long long)cycles);
    // Vanilla KB looks for a per-iteration time. We don't measure
    // wall-clock on Spike (functional sim); report cycles as a stand-in.
    printf("speedup_baseline_us=%lld\\n", (long long)cycles);
    return mismatches == 0 ? 0 : 1;
}
"""


def render_driver_c(*, M: int, K: int, N: int) -> str:
    return (
        _DRIVER_C
        .replace("@@M@@", str(M))
        .replace("@@K@@", str(K))
        .replace("@@N@@", str(N))
    )


def stage_contract_dir(out_dir: Path, *, M: int, K: int, N: int) -> Path:
    """Create a per-contract dir with ``init.cu`` + ``driver.cpp``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "init.cu").write_text(render_init_c())
    (out_dir / "driver.cpp").write_text(render_driver_c(M=M, K=K, N=N))
    return out_dir


__all__ = ["render_driver_c", "render_init_c", "stage_contract_dir"]
