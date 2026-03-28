"""Export-side decomposition synthesis hooks.

The current implementation keeps this conservative and returns no synthesized
torch.export decomposition. Payload-level synthesized translations live in
``synthesize_translation.py``.
"""

from __future__ import annotations

from typing import Any


def synthesize_export_decomposition(*_: Any, **__: Any) -> None:
    """Return ``None`` until export-side synthesis is implemented."""

    return None


__all__ = ["synthesize_export_decomposition"]
