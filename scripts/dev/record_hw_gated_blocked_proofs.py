#!/usr/bin/env python
"""record typed blocked_proof.json for each HW-gated provider
under ``results/extension_provider_evidence_pack/per_provider/<id>/``.

Calls each remote shell's ``probe()``, captures the typed status +
blocked_reason + detail, and writes a
:class:`compgen.audit.execution_evidence.BlockedProof` to disk.

When the user later populates ``configs/remote_targets/<file>.yaml``
with a real SSH host, the same provider's probe flips to
``available`` and a follow-up run of
``exercise_core4_providers.py``-style scripts can record the full
quartet via ``execute_on_remote_and_record``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from compgen.audit.execution_evidence import (
    BLOCKED_PROOF_REASONS,
    EVIDENCE_SCHEMA_VERSION,
    BlockedProof,
    record_block,
)


HW_GATED_PROVIDERS = {
    "pallas": "compgen.providers.adapters.pallas:PallasProvider",
    "nki": "compgen.providers.adapters.nki:NkiProvider",
    "hexagon_mlir": "compgen.providers.adapters.hexagon_mlir:HexagonMLIRProvider",
    "gemmini_c": "compgen.providers.adapters.gemmini_c:GemminiCProvider",
    "radiance_muon": "compgen.providers.adapters.radiance_muon:RadianceMuonProvider",
}


# Probe blocked_reason → BLOCKED_PROOF_REASONS mapping. The remote-shell
# probe already uses values from the BLOCKED_REASONS enum; this is a
# defensive translation when something unexpected lands.
def _coerce_reason(reason: str | None) -> str:
    if reason and reason in BLOCKED_PROOF_REASONS:
        return reason
    return "probe_exception"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--evidence-pack",
        type=Path,
        default=Path("results/extension_provider_evidence_pack"),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.evidence_pack.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for pid, entry in HW_GATED_PROVIDERS.items():
        mp, _, sym = entry.partition(":")
        cls = getattr(importlib.import_module(mp), sym)
        inst = cls()
        probe = inst.probe()
        proof = BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id=pid,
            status=probe.status if probe.status in (
                "blocked",
                "unsupported",
                "probe_error",
                "not_installed",
            ) else "blocked",
            blocked_reason=_coerce_reason(probe.blocked_reason),
            detail=(probe.detail or f"probe={probe.status}")[:1024],
            missing=(probe.detail or "")[:256] if probe.detail else "",
            verified_utc=_now(),
        )
        record_block(
            evidence_pack=args.evidence_pack,
            provider_id=pid,
            proof=proof,
        )
        outcomes.append(
            {
                "provider_id": pid,
                "probe_status": probe.status,
                "blocked_reason": proof.blocked_reason,
                "detail": proof.detail[:80],
            }
        )

    summary_path = args.evidence_pack / "hw_gated_blocked_proof_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "m91b_blocked_proof_summary_v1",
                "generated_at_utc": _now(),
                "outcomes": outcomes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Recorded blocked_proof for {len(outcomes)} HW-gated providers")
    for o in outcomes:
        print(
            f"  {o['provider_id']:15s} {o['probe_status']:10s} "
            f"{o['blocked_reason']:25s} {o['detail']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
