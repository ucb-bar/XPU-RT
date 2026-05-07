"""Stable content-addressed hashes for :class:`KernelContractV3`.

Two hashes — one keyed on concrete shapes, one keyed on shape *class*:

- :func:`instance_contract_hash` — the historical
  :func:`hash_contract`. Hashes the full ``kernel_facing()`` projection
  including concrete IO dims. Two contracts hash identically iff every
  field the kernel codegen is allowed to read is byte-identical.
  Used for: per-binding plan keys, the standard
  ``04_kernel_codegen/certificates/<hash>.json`` filename, the
  M-43 commit path's hash invariant.
- :func:`canonical_contract_hash` — Phase D / M-58. Same projection,
  but IO ``dims`` are passed through
  :func:`compgen.promotion.region_signature.encode_shape_class` first
  so dynamic dims (``None``) become ``{"dynamic": true}`` and
  divisibility-class dims become ``{"mod": k}``. Two contracts with
  concrete dim values that match under shape-class abstraction hash
  identically. Used for: cross-model recipe library lookup, the
  M-63 coverage-first kernel-per-archetype scheduler.

Compiler-only fields (fusion policy, observability hooks, output-buffer
lifetimes, dispatch concurrency caps, selection hints, cost estimates)
are excluded *by construction* from both hashes — they don't change
the kernel; they change how the planner uses it.

Output is a hex SHA256 truncated to 16 chars (matching
``RecipeKey._compute_hash`` so directory names stay readable).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any

from compgen.kernels.contract_v3 import KernelContractV3
from compgen.promotion.region_signature import _abstract_dim

_TRUNC = 16


def _normalize(obj: Any) -> Any:
    """Convert a kernel-facing-view subtree to a JSON-stable form."""
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return {f.name: _normalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (tuple, list)):
        return [_normalize(x) for x in obj]
    return obj


def _abstract_shape_dims_in_payload(payload: Any) -> Any:
    """Walk a normalized kernel-facing payload and rewrite every
    ``shape.dims`` entry through ``_abstract_dim`` so the resulting
    JSON encodes shape-class form rather than concrete dims.

    Concrete int dims pass through unchanged; ``None`` becomes
    ``{"dynamic": true}``; pre-abstracted ``{"mod": k}`` /
    ``{"dynamic": true}`` are normalised. The walk only rewrites the
    ``dims`` list inside any nested ``shape`` dict — every other
    field in the projection is preserved verbatim.
    """
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if k == "shape" and isinstance(v, dict) and "dims" in v:
                inner = dict(v)
                inner["dims"] = [_abstract_dim(d) for d in (v.get("dims") or [])]
                out[k] = inner
            else:
                out[k] = _abstract_shape_dims_in_payload(v)
        return out
    if isinstance(payload, list):
        return [_abstract_shape_dims_in_payload(x) for x in payload]
    return payload


def instance_contract_hash(contract: KernelContractV3) -> str:
    """Hash a kernel contract over its concrete kernel-facing projection.

    Two contracts hash identically iff every kernel-facing field is
    byte-equal — including concrete shape dims. This is the per-binding
    cache key used by the M-43 commit path and the M-45 certificate
    filename.
    """
    view = contract.kernel_facing()
    payload = _normalize(view)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:_TRUNC]


def canonical_contract_hash(contract: KernelContractV3) -> str:
    """Hash a kernel contract over its shape-class kernel-facing projection.

    Same projection as :func:`instance_contract_hash`, but every IO
    ``shape.dims`` entry is run through
    :func:`compgen.promotion.region_signature.encode_shape_class`
    abstraction before hashing. This makes contracts that differ only
    in concrete shape values within the same class collide on a single
    canonical hash, enabling cross-model recipe-library lookup.
    """
    view = contract.kernel_facing()
    payload = _abstract_shape_dims_in_payload(_normalize(view))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:_TRUNC]


def hash_contract(contract: KernelContractV3) -> str:
    """Backward-compatible alias for :func:`instance_contract_hash`.

    Kept so existing Phase B/C callers (M-41 / M-43 / M-44 / M-45 / M-46)
    continue to work without churn. New code should use
    :func:`instance_contract_hash` (concrete-shape) or
    :func:`canonical_contract_hash` (shape-class) explicitly.
    """
    return instance_contract_hash(contract)


__all__ = [
    "canonical_contract_hash",
    "hash_contract",
    "instance_contract_hash",
]
