"""Generate a self-contained CompGen extension-pack skeleton.

A scaffolded pack is a pip-installable Python package that ships a
``manifest.yaml`` and a ``compgen.packs`` entry point. Users extend
CompGen without cloning the repo.
"""

from __future__ import annotations

from compgen.packs.scaffolding.generator import (
    SUPPORTED_KINDS,
    ScaffoldResult,
    scaffold_pack,
)

__all__ = ["SUPPORTED_KINDS", "ScaffoldResult", "scaffold_pack"]
