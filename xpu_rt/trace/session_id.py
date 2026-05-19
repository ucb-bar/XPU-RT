"""Session-id construction — human-readable, sortable, unique.

Format: ``YYYYMMDD-HHMMSS_<model>_<target>_<short-uuid>``

- ``YYYYMMDD-HHMMSS`` (UTC) sorts lexicographically by wall clock so a
  directory listing is a timeline.
- ``<model>`` / ``<target>`` are slugified class names. Either segment
  is omitted if unknown — the id shrinks rather than carrying "unknown"
  placeholders.
- ``<short-uuid>`` (8 hex chars) disambiguates parallel runs with
  identical model+target at the same second.

Examples::

    20260422-164246_TinyMLP_test-gpu-simt_c494bf32
    20260422-164820_Gemma2B_cuda-a100_18ab09d4
    20260422-164930_drv_test-gpu-simt_a12f0011   # agent driver (no model)
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slugify(value: str, *, max_len: int = 32) -> str:
    """Return a filesystem-safe slug for ``value``.

    Collapses any run of disallowed characters to ``-``, trims leading
    / trailing dashes, and caps the length — short enough to keep the
    whole session id readable, long enough to stay unique-ish.
    """
    slug = _SLUG_RE.sub("-", str(value)).strip("-._")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-._")
    return slug


def _model_slug(model: Any) -> str:
    """Best-effort short name for a model object."""
    if model is None:
        return ""
    name = getattr(model, "model_name", None) or getattr(model, "name", None) or type(model).__name__
    if not name or name in {"NoneType", "object"}:
        return ""
    return _slugify(name)


def _target_slug(target_device: Any) -> str:
    """Best-effort short name for a target device."""
    if target_device is None:
        return ""
    profile = getattr(target_device, "profile", None)
    name = getattr(profile, "name", None) or getattr(target_device, "name", "")
    return _slugify(name)


def build_session_id(
    *,
    model: Any = None,
    target_device: Any = None,
    prefix: str = "",
    now: float | None = None,
) -> str:
    """Build a descriptive, sortable session id.

    Args:
        model: Optional model object; its class name (or ``model_name``
            attr) becomes the second segment.
        target_device: Optional device; its ``profile.name`` becomes
            the third segment.
        prefix: Optional leading tag (e.g. ``"drv"`` for agent-driver
            sessions where no model is bound yet).
        now: Override timestamp (seconds since epoch) for testing.

    Returns:
        A session id like
        ``20260422-164246_TinyMLP_test-gpu-simt_c494bf32``. Missing
        segments are omitted rather than replaced with ``unknown``.
    """
    t = time.gmtime(now if now is not None else time.time())
    ts = time.strftime("%Y%m%d-%H%M%S", t)
    short = uuid.uuid4().hex[:8]
    parts: list[str] = []
    if prefix:
        parts.append(_slugify(prefix, max_len=16))
    parts.append(ts)
    m = _model_slug(model)
    if m:
        parts.append(m)
    tgt = _target_slug(target_device)
    if tgt:
        parts.append(tgt)
    parts.append(short)
    return "_".join(parts)


__all__ = ["build_session_id"]
