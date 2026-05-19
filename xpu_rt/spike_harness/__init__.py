"""Spike-based compile + run harness microservices.

Drop-in replacements for vanilla KernelBlaster's
``servers/compile.py`` + ``servers/gpu.py`` that route to per-target
configs. The KB-vanilla pipeline driver, KB-v2's
:class:`CRiscvEvaluator`, and the cross-target comparison runner all
consume this harness — the package name reflects that (these aren't
KB-specific anymore).

Per-target configs live in :mod:`xpu_rt.spike_harness.targets`.
Per-target template snippets (init.c starters + driver.c harnesses)
live in :mod:`xpu_rt.spike_harness.templates.<target>`.

Supported targets today:
  * ``gemmini_mx`` — Gemmini systolic RoCC (custom-3),
    ``spike --extension=gemmini pk``, cycle source
    ``MAIN_LD_ST_EX_CYCLES``.
  * ``saturn_opu_v128`` — Saturn OPU vector unit (RVV 1.0 + zvl128b),
    ``spike --isa=rv64gcv_zvl128b_zicntr pk``, cycle source
    ``mcycle`` CSR. OPU outer-product instructions are picked up
    automatically when ``XPU_RT_SPIKE_BIN`` points at the
    Saturn-OPU Spike fork.

Adding a target = add one :class:`SpikeTargetSpec` row in
``targets.py`` + one ``templates/<id>.py`` snippet module.
"""

from __future__ import annotations

from xpu_rt.spike_harness.targets import (
    SPIKE_TARGETS,
    SpikeTargetSpec,
    resolve_target,
)

__all__ = ["SPIKE_TARGETS", "SpikeTargetSpec", "resolve_target"]
