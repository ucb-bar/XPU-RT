"""Embedded C ABI runtime emitter.

Produces a self-contained source tree that builds into
``libcompgen_model.a`` + ``model_blob.c`` + ``compgen_model.h`` — the
three artifacts the Zephyr overlay generator expects to link into an
embedded ``app``. The ABI is intentionally minimal (three entry points:
``compgen_init``, ``compgen_invoke``, ``compgen_shutdown``) so the
runtime is portable across bare-metal, Zephyr, and Linux userspace
without #ifdefs in the caller.

Sibling of :mod:`compgen.runtime.baremetal`: the bare-metal emitter
generates a whole standalone ELF (main + linker script + weights),
whereas this emitter produces a *library* meant to be linked into an
OS-hosted app (Zephyr today).
"""

from __future__ import annotations

from compgen.runtime.embedded.cnn_lowering import LoweredModel, lower_cnn_to_c
from compgen.runtime.embedded.emitter import (
    EmbeddedArtifacts,
    EmbeddedEmitter,
    EmbeddedOptions,
    emit_embedded,
)

__all__ = [
    "EmbeddedArtifacts",
    "EmbeddedEmitter",
    "EmbeddedOptions",
    "LoweredModel",
    "emit_embedded",
    "lower_cnn_to_c",
]
