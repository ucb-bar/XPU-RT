"""Per-arch perf table — Hopper (sm_90).

H100 / H200 numbers. Hopper has tensor cores but the wgmma.async
path differs from Blackwell's tcgen05.mma. Wave 1.6 enabled
clusters on Blackwell only; Hopper-cluster enablement is a
follow-up.
"""

from __future__ import annotations

PEAK_FP32_TFLOPS_PER_SM = 4.0
PEAK_BF16_TC_TFLOPS_PER_SM = 40.0
SM_COUNT_DEFAULT = {"sm_90": 132}
EAGER_LAUNCH_OVERHEAD_US = 10.0

# Empirical eager throughputs (see blackwell/cost.py for rationale).
# Hopper has wgmma but no tcgen05 → fp8 path is via wgmma.fp8.
PEAK_EAGER_FP32_TFLOPS_PER_SM = 0.40
PEAK_EAGER_BF16_TFLOPS_PER_SM = 2.30
PEAK_EAGER_FP8_TFLOPS_PER_SM = 4.50

COOPERATIVE_SYNC_US_PER_WAVE = 60.0
