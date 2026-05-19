"""Saturn / OPU vector-unit ``init.c`` + ``driver.c`` templates.

Routed by :mod:`xpu_rt.spike_harness.templates` for ``target_id``
starting with ``"saturn"`` / ``"opu"``.

  * starter ``init.c`` is a scalar i8×i8→i32 matmul that KB rewrites
    to use RVV 1.0 intrinsics (``vsetvli`` + ``vle8`` + ``vwmacc`` +
    ``vse32``). OPU outer-product asm-macros (``OPMVINBCAST`` /
    ``VOPACC``) are emitted by the agent only when the operator's
    Spike binary supports them (the Saturn-OPU Spike fork at
    https://github.com/CobbledSteel/riscv-isa-sim/tree/saturn-opu-extension).
  * ``driver.c`` reads cycles from the standard ``mcycle`` CSR
    (the same pattern Saturn benchmarks use at
    ``chipyard/generators/saturn/benchmarks/common/util.h``).
  * Reports ``mismatches=N/M`` + ``cycles=N`` — same protocol as the
    Gemmini template so the evaluator parser is uniform.
"""

from __future__ import annotations

from pathlib import Path


_INIT_C = """\
// Starter scalar kernel for Saturn-OPU (RVV 1.0 + zvl128b + zicntr).
// KB's RL loop rewrites the body of launch_gpu_implementation to use
// RVV intrinsics (vsetvli + vle8.v + vwmacc.vx + vse32.v) and, when
// the operator's Spike binary supports it, the bme.h OPU asm macros
// (OPMVINBCAST / VOPACC etc.).
//
// Contract: int8 inputs, int32 accumulator. Output is full-precision i32.
//
// Signature is the vanilla KB one (KB's parser depends on it):
//   void launch_gpu_implementation(void *output,
//                                  void *input_A,
//                                  void *input_B,
//                                  int64_t M, int64_t K, int64_t N)

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

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
// Harness for Saturn-OPU. Compiles together with init.c (which
// provides launch_gpu_implementation). Reports correctness + cycle
// count via the standard mcycle CSR (chipyard/generators/saturn/
// benchmarks/common/util.h uses the same read_csr(mcycle) pattern).

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// rdcycle / read_csr pattern copied from Saturn benchmarks. Stock
// RVV 1.0 toolchain assembles it as long as -march includes zicntr.
#define READ_CSR(reg) ({ unsigned long __tmp; \\
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); __tmp; })

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

    unsigned long cycles_before = READ_CSR(mcycle);
    launch_gpu_implementation((void *)Ctest, (void *)A, (void *)B,
                              TEST_M, TEST_K, TEST_N);
    unsigned long cycles_after = READ_CSR(mcycle);
    long long cycles = (long long)(cycles_after - cycles_before);

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
    printf("cycles=%lld\\n", cycles);
    printf("speedup_baseline_us=%lld\\n", cycles);
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
