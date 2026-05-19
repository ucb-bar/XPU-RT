"""Per-arch perf table — Blackwell (sm_100 / sm_120).

The numbers are conservative published TFLOPS — production code
calibrates from a microbenchmark on first probe (Wave 1.6b
candidate). For now these are good-enough for the universal
ETC-vs-eager predictor's roofline.

Owned by ``blackwell/`` because the bf16+fp32-acc tensor-core
peak is arch-specific. Hopper has its own number, Ampere has a
much smaller one.
"""

from __future__ import annotations

# Peak fp32 SIMT (TFLOPS/s/SM). Limited by 5th-gen CUDA cores.
PEAK_FP32_TFLOPS_PER_SM = 4.5


# Peak bf16+fp32-acc throughput (TFLOPS/s/SM). The tcgen05.mma
# path engages on sm_100 with cuBLASDx Size<64,64,16> +
# Precision<bf16,bf16,fp32> per bridge #095.
PEAK_BF16_TC_TFLOPS_PER_SM = 50.0


# Total SMs on the chip. sm_100 (B100/B200 datacenter): 132.
# sm_120 (RTX PRO 6000 workstation Blackwell): 188. Probe-selected
# via ``CudaDeviceProbe.sm_count``; this is the static fallback.
SM_COUNT_DEFAULT = {
    "sm_100": 132,
    "sm_120": 188,
}


# Eager cuBLAS launch overhead (microseconds). Empirically ~5-10 µs
# on Blackwell; we use the conservative high end so the predictor
# doesn't over-promise ETC speedups.
EAGER_LAUNCH_OVERHEAD_US = 10.0


# Empirical *achieved* cuBLAS eager throughputs per SM (TFLOPS/s).
# These are NOT the silicon peaks — eager pays driver dispatch +
# tile-shape inefficiency + dtype-routing cost. Calibrated from
# bridge #124 MLP-1 measurement on RTX PRO 6000 Blackwell:
# - fp32: 62 TFLOPS device-wide / 188 SMs = 0.33 TFLOPS/SM
#   (analytic FLOPs / measured 26.6 ms = 6.20e+13 = 62 TFLOPS).
# - bf16: 351 TFLOPS device-wide / 188 SMs = 1.87 TFLOPS/SM
#   (from the #102 hardware probe).
# Earlier calibration was 0.47 / 2.65 — too optimistic by ~1.5×.
# The cost predictor uses these (not ``PEAK_BF16_TC_TFLOPS_PER_SM``)
# for the eager-vs-ETC speedup gate so the prediction tracks
# observed wall-clock, not theoretical peak.
PEAK_EAGER_FP32_TFLOPS_PER_SM = 0.33
PEAK_EAGER_BF16_TFLOPS_PER_SM = 1.87
PEAK_EAGER_FP8_TFLOPS_PER_SM = 5.30


# Cooperative-launch grid-sync cost (microseconds per wave) for
# the static-schedule megakernel. Per bridge #118 #102 follow-up:
# at MLP-1 we measured 230 ms ETC vs 26 ms eager — the ~204 ms
# gap is dominated by cooperative_grid sync between waves, not
# per-task scheduling. ~70 µs/wave × num_waves becomes the
# leading term when tasks-per-SM > 1.
COOPERATIVE_SYNC_US_PER_WAVE = 70.0
