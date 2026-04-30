"""Kernel cost models.

Two modes:

- **Measured** (``compgen.kernels.measure``): run the kernel on real
  hardware and time it.
- **Analytical** (``compgen.kernels.cost.roofline``): predict latency
  from arithmetic intensity + peak FLOPS/bandwidth.

Callers use measurement whenever possible and fall back to the
analytical path only when a runnable kernel is not yet available.
Neither path ever returns a placeholder ``0.0`` — if neither applies,
callers get :class:`~compgen.kernels.errors.RooflineUnavailableError`.
"""

from __future__ import annotations

from compgen.kernels.cost.etc_predict import (
    EtcCostPrediction,
    WontWinError,
    predict_etc_dispatch,
)
from compgen.kernels.cost.roofline import (
    RooflinePrediction,
    predict,
    predict_fusion_speedup,
    roofline_latency_us,
)

__all__ = [
    "EtcCostPrediction",
    "RooflinePrediction",
    "WontWinError",
    "predict",
    "predict_etc_dispatch",
    "predict_fusion_speedup",
    "roofline_latency_us",
]
