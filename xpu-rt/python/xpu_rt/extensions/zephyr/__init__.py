"""Zephyr overlay extension: generate a drop-in sample for zephyr-chipyard-sw.

Given a compiled CompGen bundle (produced by ``compgen.runtime.embedded``),
:func:`emit_overlay` writes a ``samples/<name>/`` directory inside a
local clone of ``ucb-bar/zephyr-chipyard-sw`` containing everything
``west build`` needs: ``CMakeLists.txt``, ``prj.conf``,
``custom-sections.ld``, ``src/main.c``, and the CompGen artifacts
(``libcompgen_model.a``, ``model_blob.c``, ``compgen_model.h``).

The overlay is a thin wrapper: it links the prebuilt CompGen static
library into Zephyr's ``app`` target and calls
``compgen_init``/``compgen_invoke`` against static arenas placed in
``input_data_sec`` (the same linker-section pattern the ExecuTorch
sample uses).

This module is pure Python / string templates — no Zephyr build
dependency — so it's unit-testable without the RISC-V toolchain.
"""

from __future__ import annotations

from compgen.extensions.zephyr.overlay import (
    OverlayPaths,
    OverlayResult,
    ZephyrOverlayOptions,
    emit_overlay,
)

__all__ = [
    "OverlayPaths",
    "OverlayResult",
    "ZephyrOverlayOptions",
    "emit_overlay",
]
