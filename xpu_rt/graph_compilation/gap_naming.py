"""Canonical naming for gaps and their workspaces.

A single source of truth for:

- ``slug_for_target`` — sanitize an FX target into a filesystem-friendly token
- ``extension_id`` — ``<gap_kind>__<slug>__<target_id>__<sha8>`` (the canonical
  identifier used by Gap Discovery, Extension Closure, and the registry)
- ``suggested_extension_path`` — where Extension Closure will materialize the
  workspace if/when this gap is acted on

The sha8 is content-addressed over ``(gap_kind, fx_target, shape_signature,
dtype_signature, target_id)``: same gap → same id; different shapes or
different target → different id.

Both ``gaps.py`` (Gap Discovery) and ``extension_materialize.py`` (Extension
Closure) MUST use these helpers so the gap_action_queue's ``extension_id``
matches the workspace ID materialize will produce.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")
_SLUG_MAX_LEN = 64


_RUN_OF_UNDERSCORES = re.compile(r"_+")


def slug_for_target(fx_target: str) -> str:
    """Sanitize an FX target into a filesystem-safe slug.

    ``crgtoy.affine_gelu`` → ``crgtoy_affine_gelu``
    ``aten.relu.default`` → ``aten_relu_default``
    ``<built-in function linear>`` → ``built_in_function_linear``
    ``aten._native_batch_norm_legit_no_training.default`` →
        ``aten_native_batch_norm_legit_no_training_default``

    Runs of ``_`` are collapsed to a single ``_`` so the extension_id
    separator (``__``) stays unambiguous regardless of source-name
    quirks like leading-underscore aten op names.
    """
    s = _SANITIZE_RE.sub("_", fx_target).strip("_")
    s = _RUN_OF_UNDERSCORES.sub("_", s)
    return s[:_SLUG_MAX_LEN] or "unnamed"


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def content_sha8(
    *,
    gap_kind: str,
    fx_target: str,
    shape_signature: dict[str, Any] | None,
    dtype_signature: dict[str, Any] | None,
    target_id: str,
) -> str:
    """Compute the canonical sha8 over the identity-bearing fields of a gap."""
    canonical = json.dumps(
        {
            "gap_kind": gap_kind,
            "fx_target": fx_target,
            "shape_signature": shape_signature or {},
            "dtype_signature": dtype_signature or {},
            "target_id": target_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _short_hash(canonical)


def extension_id(
    *,
    gap_kind: str,
    fx_target: str,
    target_id: str,
    shape_signature: dict[str, Any] | None,
    dtype_signature: dict[str, Any] | None,
) -> str:
    """Canonical extension identifier.

    Format: ``<gap_kind>__<slug>__<target_id>__<sha8>``.

    Examples
    --------
    >>> extension_id(
    ...     gap_kind="unsupported_op",
    ...     fx_target="crgtoy.affine_gelu",
    ...     target_id="host_cpu",
    ...     shape_signature={"inputs": [[2, 16]], "outputs": [[2, 8]]},
    ...     dtype_signature={"inputs": ["torch.float32"], "outputs": ["torch.float32"]},
    ... )
    'unsupported_op__crgtoy_affine_gelu__host_cpu__...'
    """
    slug = slug_for_target(fx_target)
    target_slug = _SANITIZE_RE.sub("_", target_id).strip("_")
    sha = content_sha8(
        gap_kind=gap_kind,
        fx_target=fx_target,
        shape_signature=shape_signature,
        dtype_signature=dtype_signature,
        target_id=target_id,
    )
    return f"{gap_kind}__{slug}__{target_slug}__{sha}"


def workspace_dir_name(
    *,
    gap_kind: str,
    fx_target: str,
    target_id: str,
    shape_signature: dict[str, Any] | None,
    dtype_signature: dict[str, Any] | None,
) -> str:
    """Directory name under ``<extensions_root>/<gap_kind>/`` for a gap.

    Format: ``<slug>__<target_id>__<sha8>`` (no gap_kind prefix because the
    parent directory already carries it). The full ``extension_id`` keeps
    the gap_kind for stable global identity.
    """
    slug = slug_for_target(fx_target)
    target_slug = _SANITIZE_RE.sub("_", target_id).strip("_")
    sha = content_sha8(
        gap_kind=gap_kind,
        fx_target=fx_target,
        shape_signature=shape_signature,
        dtype_signature=dtype_signature,
        target_id=target_id,
    )
    return f"{slug}__{target_slug}__{sha}"


def suggested_extension_path(
    *,
    gap_kind: str,
    fx_target: str,
    target_id: str,
    shape_signature: dict[str, Any] | None,
    dtype_signature: dict[str, Any] | None,
    extensions_root: Path | str = ".crg-artifacts/extensions",
) -> str:
    """Repo-relative path Extension Closure will materialize into."""
    name = workspace_dir_name(
        gap_kind=gap_kind,
        fx_target=fx_target,
        target_id=target_id,
        shape_signature=shape_signature,
        dtype_signature=dtype_signature,
    )
    return f"{Path(extensions_root).as_posix()}/{gap_kind}/{name}"
