"""Local (user-authored) LLM extensions for CompGen.

A user who installs CompGen via pip can drop custom ``Tool`` /
``InventSlot`` definitions in ``~/.compgen/extensions/*.py``. On the
first call to :func:`compgen.llm.registry.get_registry`, every file is
discovered, imported in isolation, and given a chance to register its
declarations against the global registry.

This is the *LLM extension* surface. The sibling directory
``compgen.extensions`` under the installed package is reserved for
MLIR dialect extensions (linked C++ state, dialect registration, etc.)
and is deliberately not reused here to avoid a namespace clash.
"""

from __future__ import annotations

from compgen.agent.extensions.local_loader import (
    LocalExtension,
    LocalExtensionLoadResult,
    load_local_extensions,
)

__all__ = [
    "LocalExtension",
    "LocalExtensionLoadResult",
    "load_local_extensions",
]
