"""Template CostModel — fill in for your target's perf table."""

from __future__ import annotations


class TemplateCostModel:
    """Replace with ``YourArchCostModel``. Used by the universal
    ETC-vs-eager predictor (``compgen.kernels.cost.predict_etc_dispatch``).
    Wrong numbers don't break correctness — they just produce
    wrong gate predictions."""

    def peak_tflops_per_sm(self, *, dtype: str, tensor_core: bool) -> float:
        """Per-SM peak throughput for the given dtype + tensor-core
        path. Microbenchmark on first probe in production; the
        stub returns a placeholder."""
        return 1.0

    def sm_count(self) -> int:
        """SMs on the device (or analogous compute units —
        cores for CPU, slice tiles for TPU)."""
        return 1

    def scheduling_overhead_us(self) -> float:
        """Per-task megakernel scheduling cost. Vendors with
        cooperative-launch see ~1 µs; CPU sees near-zero."""
        return 1.0

    def eager_launch_overhead_us(self) -> float:
        """One-shot kernel-launch overhead for the vendor's eager
        BLAS library. Used as the eager side of ETC-vs-eager."""
        return 10.0
