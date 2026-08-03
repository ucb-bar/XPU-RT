"""Diagnostics for xpu-rt schedules: band invariants, etc."""
from .band_invariant import check_band_invariant, BandReport, BandViolation
from .plot_band_gantt import render_band_gantt

__all__ = [
    "check_band_invariant", "BandReport", "BandViolation",
    "render_band_gantt",
]
