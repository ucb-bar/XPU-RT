"""Typed errors for the kernel cost / measurement / provider layer.

Phase 2 of the production-hardening introduces these. Previously, sites
like ``exo_adapter.py:110`` swallowed the "we couldn't measure this
kernel yet" case behind ``latency_us = 0.0``. That meant downstream
selectors compared a real latency (say, 17 µs) against a placeholder
(0 µs) and picked the unmeasured candidate — a correctness bug
dressed as performance. These errors force callers to distinguish
"measured" from "no measurement was possible".

Hierarchy:

``KernelCostError``                  (RuntimeError)
├── ``UnmeasurableKernelError``      — no runnable kernel / no
│                                       golden inputs / no target
│                                       device; fall back to roofline
│                                       or reject.
├── ``RooflineUnavailableError``     — neither measurement nor
│                                       analytical model can produce
│                                       a number.
└── ``MissingInterconnectCostError`` — planner asked for
                                       (dev_a, dev_b) transfer cost
                                       but the TargetProfile didn't
                                       declare it.
"""

from __future__ import annotations


class KernelCostError(RuntimeError):
    """Base class for kernel-cost / kernel-measurement errors."""


class UnmeasurableKernelError(KernelCostError):
    """The caller asked us to measure a kernel but we can't.

    Typical reasons: the kernel source is not a runnable callable, the
    required hardware isn't present, the contract has no golden inputs.
    Callers should either (a) supply the missing piece, (b) fall back
    to the analytical :func:`compgen.kernels.cost.roofline.predict`,
    or (c) reject the kernel — they must NOT pretend ``latency_us=0``.
    """


class RooflineUnavailableError(KernelCostError):
    """Both measurement and analytical modelling are unavailable.

    Raised when :func:`compgen.kernels.cost.roofline.predict` can't
    compute a number because the ``DeviceTraits`` / ``TargetProfile``
    lacks required fields (peak FLOPS, peak bandwidth). Tells the
    caller the cost number they requested has no grounded source and
    a placeholder would be dishonest.
    """


class MissingInterconnectCostError(KernelCostError):
    """Planner needed a transfer cost between two devices that isn't
    declared in the ``TargetProfile``. Fix by declaring the
    interconnect in the profile YAML or pass an override."""

    def __init__(self, source_device: int | str, destination_device: int | str) -> None:
        self.source_device = source_device
        self.destination_device = destination_device
        super().__init__(
            f"no interconnect cost declared between device {source_device!r} and {destination_device!r}; "
            "add an entry to TargetProfile.interconnects or pass overrides"
        )


__all__ = [
    "KernelCostError",
    "MissingInterconnectCostError",
    "RooflineUnavailableError",
    "UnmeasurableKernelError",
]
