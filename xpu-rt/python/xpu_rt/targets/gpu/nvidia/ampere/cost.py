"""Per-arch perf table — Ampere (sm_80 / sm_86).

A100 datacenter (sm_80) + RTX 30/40 consumer (sm_86). Pre-Hopper
mma.sync atoms — much smaller bf16 throughput than Blackwell.
"""

from __future__ import annotations

PEAK_FP32_TFLOPS_PER_SM = 3.0  # A100 baseline; sm_86 closer to 2.0
PEAK_BF16_TC_TFLOPS_PER_SM = 16.0  # A100 with Tensor Cores
SM_COUNT_DEFAULT = {"sm_80": 108, "sm_86": 84}  # A100 / RTX 4090 typical
EAGER_LAUNCH_OVERHEAD_US = 10.0

# Empirical eager throughputs. Pre-Hopper mma.sync — bf16 path
# meaningfully slower than Hopper/Blackwell. No fp8 hardware
# pre-Hopper, so the fp8 rate is best-effort cuBLAS lookup.
PEAK_EAGER_FP32_TFLOPS_PER_SM = 0.20
PEAK_EAGER_BF16_TFLOPS_PER_SM = 1.20
PEAK_EAGER_FP8_TFLOPS_PER_SM = 1.20  # no native fp8; same as bf16

COOPERATIVE_SYNC_US_PER_WAVE = 50.0
