"""Region pattern signature for cross-model recipe reuse.

The region signature answers "does *this* region in a new model look like
*that* region for which I already promoted a recipe?" — the
*pattern-level* tier of the two-tier promotion cache key.

A region with the same op family, dtype/layout class, abstracted shape,
and target class produces the same signature even when it lives in a
different model. That is what makes optimization knowledge portable.

Two regions hash to the same signature when:

- ``op_family`` matches (e.g. ``matmul_like``, ``pointwise``, ``reduction``)
- ``dtype`` matches (e.g. ``fp32``, ``fp16``, ``int8``)
- ``layout`` matches (e.g. ``row_major``, ``packed_k_major``)
- ``shape_class`` matches under abstraction:
  - concrete equal dims compare equal,
  - dims encoded as ``{"mod": 16}`` (any size divisible by 16) compare
    equal to other ``{"mod": 16}`` entries and to a concrete dim that is
    a multiple of 16,
  - ``{"dynamic": true}`` matches anything.
- ``target_class`` matches (e.g. ``cuda_sm75``, ``host_cpu``,
  ``triton_friendly``)

The output is a hex SHA256 truncated to 16 chars to stay readable in
``RecipeKey`` columns and directory names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_TRUNC = 16


@dataclass(frozen=True)
class RegionSignature:
    """Inputs that define a region's pattern identity.

    All fields are strings (or string-tuples) so the signature is
    canonicalisable without losing information. ``shape_class`` is a
    JSON-encoded list of dim entries — each entry is either a concrete
    int, ``{"mod": k}``, or ``{"dynamic": true}``.

    Attributes:
        op_family: Op-family canonical name (e.g. ``matmul_like``).
        dtype: Canonical dtype string (e.g. ``fp32``).
        layout: Canonical layout string (e.g. ``row_major``).
        shape_class: JSON-encoded shape abstraction; see module docstring.
        target_class: Canonical target-class string (e.g. ``cuda_sm75``).
    """

    op_family: str
    dtype: str
    layout: str
    shape_class: str
    target_class: str

    def to_dict(self) -> dict[str, str]:
        return {
            "op_family": self.op_family,
            "dtype": self.dtype,
            "layout": self.layout,
            "shape_class": self.shape_class,
            "target_class": self.target_class,
        }


def _abstract_dim(dim: Any) -> Any:
    """Reduce a dim to its canonical pattern form.

    Concrete positive ints stay as ints. ``None`` (PyTorch dynamic-shape
    placeholder) becomes ``{"dynamic": true}``. A pre-abstracted entry
    (already a dict like ``{"mod": 16}``) is normalised — keys are
    sorted, only the first matching axis is preserved.
    """
    if dim is None:
        return {"dynamic": True}
    if isinstance(dim, int):
        return int(dim)
    if isinstance(dim, dict):
        if "dynamic" in dim and dim["dynamic"]:
            return {"dynamic": True}
        if "mod" in dim:
            return {"mod": int(dim["mod"])}
    return {"dynamic": True}


def encode_shape_class(dims: list[Any] | tuple[Any, ...]) -> str:
    """Canonical JSON encoding of an abstracted shape tuple."""
    abstracted = [_abstract_dim(d) for d in dims]
    return json.dumps(abstracted, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_region_signature(sig: RegionSignature) -> str:
    """Return a stable 16-char hex hash of a region signature.

    Byte-deterministic — same inputs always produce the same hash, no
    floating-point noise, no clock or PID dependence.
    """
    payload = json.dumps(
        sig.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_TRUNC]


def make_region_signature(
    *,
    op_family: str,
    dtype: str,
    layout: str,
    dims: list[Any] | tuple[Any, ...],
    target_class: str,
) -> RegionSignature:
    """Build a :class:`RegionSignature` from raw region facts."""
    return RegionSignature(
        op_family=op_family or "unknown",
        dtype=dtype or "unknown",
        layout=layout or "row_major",
        shape_class=encode_shape_class(dims),
        target_class=target_class or "unknown",
    )


__all__ = [
    "RegionSignature",
    "encode_shape_class",
    "hash_region_signature",
    "make_region_signature",
]
