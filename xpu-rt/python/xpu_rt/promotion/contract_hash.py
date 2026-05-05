"""Stable content-addressed hash for :class:`KernelContractV3`.

The hash answers "is this the same kernel I have already compiled?" — used
as the *exact-kernel* tier of the two-tier promotion cache key. Two
contracts hash to the same value iff their
:meth:`KernelContractV3.kernel_facing` projections are equal — i.e.
every field the kernel codegen is allowed to read matches by value
(archetype, granularity, IO shapes/dtypes/layouts/alignment, numerics,
execution envelope, memory residency, event declarations, dispatch
model).

Compiler-only fields (fusion policy, observability hooks, output-buffer
lifetimes, dispatch concurrency caps, selection hints, cost estimates)
are excluded *by construction* — they don't change the kernel; they
change how the planner uses it. Using the existing
``kernel_facing()`` projection guarantees we honour exactly the same
boundary the contract author intended.

The output is a hex SHA256 truncated to 16 chars (matching
``RecipeKey._compute_hash`` so directory names stay readable).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any

from compgen.kernels.contract_v3 import KernelContractV3

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


def hash_contract(contract: KernelContractV3) -> str:
    """Return a stable 16-char hex hash of a kernel contract.

    Hashes the kernel-facing projection only; compiler-only fields
    (fusion policy, observability, planner annotations, search hints,
    cost estimates) are excluded by construction.
    """
    view = contract.kernel_facing()
    payload = _normalize(view)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:_TRUNC]


__all__ = ["hash_contract"]
