"""Fake/meta-kernel synthesis hooks.

For the initial slice we only detect whether a Meta kernel exists; missing
registrations are surfaced in verification messages and recovery dossiers.
"""

from __future__ import annotations

from typing import Any


def synthesize_fake_kernel(*_: Any, **__: Any) -> None:
    """Return ``None`` until fake-kernel generation is implemented."""

    return None


__all__ = ["synthesize_fake_kernel"]
