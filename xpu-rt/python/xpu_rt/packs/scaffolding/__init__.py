"""Generate a self-contained XPU-RT extension-pack skeleton.

A scaffolded pack is a pip-installable Python package that ships a
``manifest.yaml`` and a ``xpu_rt.packs`` entry point. Users extend
XPU-RT without cloning the repo.
"""

from __future__ import annotations

from xpu_rt.packs.scaffolding.generator import (
    SUPPORTED_KINDS,
    ScaffoldResult,
    scaffold_pack,
)

__all__ = ["SUPPORTED_KINDS", "ScaffoldResult", "scaffold_pack"]
