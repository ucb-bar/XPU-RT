"""M-64 — Forward-compatible contract refinement + version migration.

The dream's Section 7 Dream 5 calls for refinement contracts: a v3
contract upgraded to v3.1 (e.g. by adding ``prefetch_distance``) must
NOT invalidate cached v3 kernels. The mechanism:

1. New optional fields land in
   :attr:`KernelContractV3.optional_v3_1_fields` keyed by name. Field
   names are declared in
   :data:`compgen.kernels.contract_v3._OPTIONAL_V3_1_FIELD_NAMES`.
2. Both the canonical and the instance hash projections (kernel_facing
   + compiler_only) exclude this slot, so adding a new field name +
   default never changes a cached cert's hash.
3. v3 cert bodies (no ``optional_v3_1_fields`` key) load via
   :func:`migrate_contract_v3_to_v3_1` which fills in defaults from
   :data:`compgen.kernels.contract_v3._OPTIONAL_V3_1_FIELD_DEFAULTS`.
4. v3.1 reader code can read any optional field via
   :func:`get_optional_v3_1_field` without breaking when the field is
   absent (returns the recognized default).
5. The audit gate ``contract_version_consistency`` re-hashes every
   on-disk certificate after migrating its body and verifies the
   canonical_contract_hash still matches. Any drift surfaces as a
   typed gate failure (the migration would have broken cache).

This is the "refinement contracts" half of Section 7 Dream 5.
Parametric contracts (the other half — contracts parameterised by
target capability bits) remain a future extension.
"""

from __future__ import annotations

from typing import Any

from compgen.kernels.contract_v3 import (
    _OPTIONAL_V3_1_FIELD_DEFAULTS,
    _OPTIONAL_V3_1_FIELD_NAMES,
    CONTRACT_REFINEMENT_VERSION,
    CONTRACT_VERSION,
)


class ContractRefinementError(ValueError):
    """A migration step would have changed kernel-facing semantics."""


def migrate_contract_body_v3_to_v3_1(body: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a serialized v3 contract body to a v3.1-shaped body.

    The migration is a no-op for canonical hashing: it ONLY fills in
    ``optional_v3_1_fields`` defaults. Other body fields stay
    byte-identical. Calling this on a body that's already v3.1 is
    idempotent.

    Args:
        body: A dict produced by ``contract_to_dict`` or any
            equivalent v3 / v3.1 serialization.

    Returns:
        A NEW dict (defensive copy) with ``optional_v3_1_fields``
        populated for every recognized name that was missing.
    """
    out = dict(body)
    existing = dict(out.get("optional_v3_1_fields") or {})
    for name in _OPTIONAL_V3_1_FIELD_NAMES:
        if name not in existing:
            existing[name] = _OPTIONAL_V3_1_FIELD_DEFAULTS[name]
    out["optional_v3_1_fields"] = existing
    return out


def get_optional_v3_1_field(contract: Any, name: str) -> Any:
    """Read a v3.1 optional field by name with the recognized default.

    Use this in any code that touches the v3.1 fields so the access
    is robust to v3 contract bodies (which lack the slot).
    """
    if name not in _OPTIONAL_V3_1_FIELD_NAMES:
        raise ContractRefinementError(
            f"unknown v3.1 optional field {name!r}; recognized names: "
            f"{sorted(_OPTIONAL_V3_1_FIELD_NAMES)!r}"
        )
    fields = getattr(contract, "optional_v3_1_fields", None) or {}
    if name in fields:
        return fields[name]
    return _OPTIONAL_V3_1_FIELD_DEFAULTS[name]


def contract_version_tuple() -> tuple[int, int]:
    """Return the (major, minor) refinement version this build supports."""
    return CONTRACT_REFINEMENT_VERSION


def is_compatible_with(
    *,
    body_version: int,
    body_refinement: tuple[int, int] | None = None,
) -> bool:
    """Check whether this build can read a contract body with the given
    declared version.

    Compatibility rule: same major version, this build's minor >= body
    minor. v3 → v3.1 reads OK; v3.1 → v3 read also OK (we discard the
    fields we don't recognise but keep the canonical hash invariant).
    """
    if int(body_version) != CONTRACT_VERSION:
        return False
    return True


__all__ = [
    "ContractRefinementError",
    "contract_version_tuple",
    "get_optional_v3_1_field",
    "is_compatible_with",
    "migrate_contract_body_v3_to_v3_1",
]
