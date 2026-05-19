"""x86 CPU cost model — placeholder TFLOPS table for the roofline.

Wave 1.15 stub. Real numbers will calibrate from microbenchmarks
in a future iteration. Today we use representative published
peaks so the universal cost predictor can produce non-zero
numbers when CPU is the chosen target.
"""

from __future__ import annotations

# Per-arch SIMD-throughput peaks (GFLOPS/s/core, fp32). Conservative
# published figures — the real numbers depend on clock speed,
# turbo, AVX-512 frequency throttling, etc.
_X86_FP32_GFLOPS_PER_CORE = {
    "x86_avx512": 200.0,  # 512-bit SIMD × 2 FMA ports × ~3 GHz
    "x86_avx2": 80.0,  # 256-bit SIMD
    "x86_scalar": 8.0,  # No SIMD
}


# Eager BLAS launch overhead on CPU is much smaller than GPU
# (~1 µs vs ~10 µs for cuBLAS) — no driver involved.
_CPU_EAGER_LAUNCH_OVERHEAD_US = 1.0


# CPU has effectively zero per-task scheduling overhead — intra-
# CPU sync between two function calls is just a return + call.
# The megakernel collapses to a serial chain.
_CPU_SCHEDULING_OVERHEAD_US = 0.05


class X86CostModel:
    """Vendor-supplied perf coefficients for the universal
    roofline + ETC-vs-eager predictor."""

    def __init__(self, arch: str = "x86_avx512", num_cores: int = 8) -> None:
        self._arch = arch
        self._num_cores = num_cores

    def peak_tflops_per_sm(self, *, dtype: str, tensor_core: bool) -> float:
        """CPU has no tensor cores. Returns scalar SIMD peak in
        TFLOPS/s/core — naming retained for Protocol parity."""
        del dtype, tensor_core  # stub uses fp32 SIMT-equivalent only
        gflops = _X86_FP32_GFLOPS_PER_CORE.get(self._arch, 8.0)
        return gflops / 1000.0  # GFLOPS → TFLOPS

    def sm_count(self) -> int:
        """Per Protocol parity — returns core count for CPU."""
        return self._num_cores

    def scheduling_overhead_us(self) -> float:
        return _CPU_SCHEDULING_OVERHEAD_US

    def eager_launch_overhead_us(self) -> float:
        return _CPU_EAGER_LAUNCH_OVERHEAD_US
