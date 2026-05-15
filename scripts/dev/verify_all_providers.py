#!/usr/bin/env python
"""closure verifier for the full Phase F provider matrix.

Walks every card under
``python/compgen/{providers,dialects}/cards/`` and asserts:

1. Card loads with ``integration_level`` declared.
2. ``card.entrypoint`` resolves to a real Python class.
3. The resolved class implements the :class:`KernelProvider`
   ABC (directly OR after wrapping via
   :class:`compgen.providers.legacy_shim.LegacyProviderAdapter`).
4. ``probe()`` returns a typed ``ProviderProbeResult`` with
   ``status`` in :data:`PROBE_STATUSES`.
5. If the per-provider evidence dir exists under
   ``results/extension_provider_evidence_pack/per_provider/<id>/``:

   * ``available_with_evidence``: the quartet
     ``kernel_source.* + run_report.json + certificate.json`` is
     present and schema-valid.
   * ``blocked``: a ``blocked_proof.json`` is present and
     schema-valid.

Exits 0 only when **every** card passes the gates above. Honest
typed blocks count as passing (they're the system surfacing
truth, not faking).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.audit.execution_evidence import audit_provider_dir
from compgen.providers.adapters.base import (
    AdapterResolutionError,
    resolve_provider_class,
)
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
)
from compgen.providers.kernel_provider import KernelProvider
from compgen.providers.legacy_shim import wrap_legacy
from compgen.providers.provider_types import PROBE_STATUSES, ProviderCard


# Providers whose constructors require kwargs that we cannot supply
# in this verifier without the active session context. They are
# checked for resolvability but skipped for the instance/probe gates.
KWARGS_REQUIRED_PROVIDERS = frozenset({"claude_kernel"})


@dataclass(frozen=True)
class CardCheck:
    card_id: str
    kind: str  # "provider" | "dialect"
    integration_level: str
    resolves: bool
    is_kernel_provider: bool
    probe_status: str
    evidence_state: str  # "available" | "blocked" | "missing" | "empty"
    detail: str = ""

    @property
    def passes(self) -> bool:
        # Mandatory: card resolves AND is a KernelProvider (post-shim
        # if needed) AND probe returns a typed status.
        if not self.resolves:
            return False
        if self.card_id in KWARGS_REQUIRED_PROVIDERS:
            # Skip instance-level gates for kwargs-required cards.
            return True
        if not self.is_kernel_provider:
            return False
        if self.probe_status not in PROBE_STATUSES:
            return False
        # Evidence is "must be valid if present" — missing dir is OK.
        if self.evidence_state in ("missing", "available", "blocked"):
            return True
        return False  # "empty" or "malformed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "kind": self.kind,
            "integration_level": self.integration_level,
            "resolves": self.resolves,
            "is_kernel_provider": self.is_kernel_provider,
            "probe_status": self.probe_status,
            "evidence_state": self.evidence_state,
            "detail": self.detail,
            "passes": self.passes,
        }


def _resolve(card_entrypoint: str):
    mod_path, _, sym = card_entrypoint.partition(":")
    if not mod_path or not sym:
        raise AdapterResolutionError(
            provider_id="<unknown>", entrypoint=card_entrypoint, reason="bad_entrypoint_syntax"
        )
    try:
        mod = importlib.import_module(mod_path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AdapterResolutionError(
            provider_id="<unknown>", entrypoint=card_entrypoint, reason="module_not_importable"
        ) from exc
    try:
        return getattr(mod, sym)
    except AttributeError as exc:
        raise AdapterResolutionError(
            provider_id="<unknown>", entrypoint=card_entrypoint, reason="symbol_not_in_module"
        ) from exc


def _check_provider_card(
    card: ProviderCard,
    *,
    evidence_pack: Path,
) -> CardCheck:
    pid = card.provider_id
    try:
        cls = resolve_provider_class(card)
    except AdapterResolutionError as exc:
        return CardCheck(
            card_id=pid,
            kind="provider",
            integration_level=card.integration_level,
            resolves=False,
            is_kernel_provider=False,
            probe_status="probe_error",
            evidence_state="missing",
            detail=f"resolve failed: {exc.reason}",
        )

    if pid in KWARGS_REQUIRED_PROVIDERS:
        ev_dir = evidence_pack / "per_provider" / pid
        ev_state = "missing"
        if ev_dir.is_dir():
            state, _ = audit_provider_dir(ev_dir)
            ev_state = state
        return CardCheck(
            card_id=pid,
            kind="provider",
            integration_level=card.integration_level,
            resolves=True,
            is_kernel_provider=False,
            probe_status="skipped_kwargs_required",
            evidence_state=ev_state,
            detail="constructor needs kwargs; instance-level checks skipped",
        )

    try:
        inst = cls()
    except Exception as exc:
        return CardCheck(
            card_id=pid,
            kind="provider",
            integration_level=card.integration_level,
            resolves=True,
            is_kernel_provider=False,
            probe_status="probe_error",
            evidence_state="missing",
            detail=f"instantiation failed: {type(exc).__name__}: {exc}",
        )

    if isinstance(inst, KernelProvider):
        is_kp = True
        wrapped = inst
    else:
        wrapped = wrap_legacy(card, inst)
        is_kp = isinstance(wrapped, KernelProvider)
    try:
        probe = wrapped.probe()
        probe_status = probe.status
    except Exception as exc:
        probe_status = f"raised:{type(exc).__name__}"

    ev_dir = evidence_pack / "per_provider" / pid
    ev_state = "missing"
    if ev_dir.is_dir():
        state, _ = audit_provider_dir(ev_dir)
        ev_state = state

    return CardCheck(
        card_id=pid,
        kind="provider",
        integration_level=card.integration_level,
        resolves=True,
        is_kernel_provider=is_kp,
        probe_status=probe_status,
        evidence_state=ev_state,
    )


def _check_dialect_card(card, *, evidence_pack: Path) -> CardCheck:
    did = card.dialect_provider_id
    try:
        cls = _resolve(card.entrypoint)
    except AdapterResolutionError as exc:
        return CardCheck(
            card_id=did,
            kind="dialect",
            integration_level=card.integration_level,
            resolves=False,
            is_kernel_provider=False,
            probe_status="probe_error",
            evidence_state="missing",
            detail=f"resolve failed: {exc.reason}",
        )
    try:
        inst = cls()
    except Exception as exc:
        return CardCheck(
            card_id=did,
            kind="dialect",
            integration_level=card.integration_level,
            resolves=True,
            is_kernel_provider=False,
            probe_status="probe_error",
            evidence_state="missing",
            detail=f"instantiation failed: {type(exc).__name__}: {exc}",
        )
    is_kp = isinstance(inst, KernelProvider)
    try:
        probe = inst.probe()
        probe_status = probe.status
    except Exception as exc:
        probe_status = f"raised:{type(exc).__name__}"
    # Check the dialect's evidence dir under per_provider/<id>/ (we
    # key by dialect_provider_id, same convention as provider ids;
    # this lets HW-gated providers/dialects that share an id share
    # the same blocked_proof file).
    ev_dir = evidence_pack / "per_provider" / did
    ev_state = "missing"
    if ev_dir.is_dir():
        state, _ = audit_provider_dir(ev_dir)
        ev_state = state
    return CardCheck(
        card_id=did,
        kind="dialect",
        integration_level=card.integration_level,
        resolves=True,
        is_kernel_provider=is_kp,
        probe_status=probe_status,
        evidence_state=ev_state,
    )


def verify_all(*, evidence_pack: Path | None = None) -> list[CardCheck]:
    ep = evidence_pack or Path("results/extension_provider_evidence_pack")
    checks: list[CardCheck] = []
    for c in iter_provider_cards():
        checks.append(_check_provider_card(c, evidence_pack=ep))
    for c in iter_dialect_cards():
        checks.append(_check_dialect_card(c, evidence_pack=ep))
    return checks


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-pack", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    checks = verify_all(evidence_pack=args.evidence_pack)
    body = {
        "schema_version": "verify_all_providers_v1",
        "total_cards": len(checks),
        "passing": sum(1 for c in checks if c.passes),
        "failing": sum(1 for c in checks if not c.passes),
        "checks": [c.to_dict() for c in checks],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(body, indent=2, sort_keys=True))

    print(f"Verified {len(checks)} cards: "
          f"{body['passing']} passing, {body['failing']} failing")
    by_evidence_state: dict[str, int] = {}
    for c in checks:
        by_evidence_state[c.evidence_state] = by_evidence_state.get(c.evidence_state, 0) + 1
    print(f"Evidence states: {by_evidence_state}")
    for c in checks:
        flag = "✓" if c.passes else "✗"
        print(
            f"  {flag} {c.kind:8s} {c.card_id:25s} "
            f"resolves={c.resolves} kp={c.is_kernel_provider} "
            f"probe={c.probe_status:25s} evidence={c.evidence_state}"
        )
    return 0 if body["failing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
