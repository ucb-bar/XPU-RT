"""Per-target Spike-harness configuration.

One :class:`SpikeTargetSpec` row per target captures every place the
compile + run path diverges across targets. Adding Hexagon-via-Spike
or any future RISC-V-Spike target = one extra row here + one
``templates/<id>.py`` file.

Resolution is by ``target_id`` prefix match (e.g. ``"saturn_dsp_v128"``
also resolves to the Saturn row), matching the same convention
:func:`xpu_rt.kernels.autocomp_adapter.resolve_autocomp_target` uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SpikeTargetSpec:
    """Everything the compile + run servers need to dispatch one target.

    Attributes:
        target_id: Canonical target id (``"gemmini_mx"`` / ``"saturn_opu_v128"``).
        march_flags: ``-march`` + ``-Wa,-march`` flags for the cross-
            compiler. e.g.
            ``("-march=rv64gc", "-Wa,-march=rv64gc")`` for Gemmini.
        spike_flag: Tuple of flags prepended to the ``spike`` command,
            e.g. ``("--extension=gemmini",)`` or
            ``("--isa=rv64gcv_zvl128b_zicntr",)``.
        include_args_factory: Callable returning the extra ``-I``
            include flags (a callable rather than a tuple so paths can
            be sourced from env vars at invocation time).
        extra_compile_flags: Any extra flags beyond the standard set.
        templates_module: Dotted module path under
            ``xpu_rt.spike_harness.templates`` with the
            ``render_init_c`` + ``render_driver_c`` functions.
        cycle_source_label: Stamped into the canonical row's
            ``cycle_source`` field. See
            :mod:`xpu_rt.benchmarks.canonical_metrics` for the
            recognised values.
    """

    target_id: str
    march_flags: tuple[str, ...]
    spike_flag: tuple[str, ...]
    include_args_factory: object  # callable () -> tuple[str, ...]
    extra_compile_flags: tuple[str, ...]
    templates_module: str
    cycle_source_label: str

    @property
    def include_args(self) -> tuple[str, ...]:
        return tuple(self.include_args_factory()) if callable(self.include_args_factory) else tuple(self.include_args_factory)


# ---------------------------------------------------------------------------
# Gemmini-specific helpers
# ---------------------------------------------------------------------------


def _gemmini_root() -> Path:
    return Path(
        os.environ.get(
            "XPU_RT_CHIPYARD_GEMMINI_ROOT",
            "/scratch2/agustin/chipyard/generators/gemmini",
        )
    )


def _gemmini_include_args() -> tuple[str, ...]:
    """Header search paths so the harness picks up gemmini.h +
    gemmini_counter.h. Sourced from the chipyard Gemmini submodule."""
    root = _gemmini_root() / "software" / "gemmini-rocc-tests"
    return (
        f"-I{root}",
        f"-I{root}/include",
        f"-I{root}/riscv-tests",
        f"-I{root}/riscv-tests/env",
    )


# ---------------------------------------------------------------------------
# Saturn / OPU helpers
# ---------------------------------------------------------------------------


def _saturn_include_args() -> tuple[str, ...]:
    """Saturn templates use stock ``<riscv_vector.h>`` from the
    toolchain; no extra include paths needed. Empty tuple by design."""
    return ()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_GEMMINI_SPEC = SpikeTargetSpec(
    target_id="gemmini",
    march_flags=("-march=rv64gc", "-Wa,-march=rv64gc"),
    spike_flag=("--extension=gemmini",),
    include_args_factory=_gemmini_include_args,
    extra_compile_flags=(),
    templates_module="xpu_rt.spike_harness.templates.gemmini",
    cycle_source_label="MAIN_LD_ST_EX_CYCLES",
)


SPIKE_TARGETS: dict[str, SpikeTargetSpec] = {
    # Canonical id for stock INT8 16×16 Gemmini.
    "gemmini": _GEMMINI_SPEC,
    # Historical alias — same hardware, retained so cached samples
    # and existing config YAML keep resolving cleanly.
    "gemmini_mx": SpikeTargetSpec(
        target_id="gemmini_mx",
        march_flags=("-march=rv64gc", "-Wa,-march=rv64gc"),
        spike_flag=("--extension=gemmini",),
        include_args_factory=_gemmini_include_args,
        extra_compile_flags=(),
        templates_module="xpu_rt.spike_harness.templates.gemmini",
        cycle_source_label="MAIN_LD_ST_EX_CYCLES",
    ),
    "saturn_opu_v128": SpikeTargetSpec(
        target_id="saturn_opu_v128",
        # zicntr exposes the mcycle CSR — required for the harness's
        # csrr mcycle to assemble. OPU instructions are picked up by
        # the Spike fork at https://github.com/CobbledSteel/riscv-isa-sim/
        # tree/saturn-opu-extension when XPU_RT_SPIKE_BIN points at it.
        march_flags=(
            "-march=rv64gcv_zvl128b_zicntr",
            "-Wa,-march=rv64gcv_zvl128b_zicntr",
        ),
        spike_flag=("--isa=rv64gcv_zvl128b_zicntr",),
        include_args_factory=_saturn_include_args,
        extra_compile_flags=(),
        templates_module="xpu_rt.spike_harness.templates.saturn",
        cycle_source_label="mcycle",
    ),
}


def resolve_target(target_id: str) -> SpikeTargetSpec:
    """Resolve a target_id (prefix-matched) to its SpikeTargetSpec.

    Raises:
        KeyError: when no registered target matches.
    """
    t = (target_id or "").lower()
    # Canonical "gemmini" wins over historical "_mx" alias; both
    # resolve to the same systolic stock-default behaviour.
    if t == "gemmini" or t == "gemmini_mx":
        return SPIKE_TARGETS[t]
    if t.startswith("gemmini"):
        return SPIKE_TARGETS["gemmini"]
    if t.startswith("saturn") or t.startswith("opu"):
        return SPIKE_TARGETS["saturn_opu_v128"]
    if target_id in SPIKE_TARGETS:
        return SPIKE_TARGETS[target_id]
    raise KeyError(
        f"no SpikeTargetSpec registered for target_id={target_id!r}; "
        f"known: {sorted(SPIKE_TARGETS)}"
    )


__all__ = ["SPIKE_TARGETS", "SpikeTargetSpec", "resolve_target"]
