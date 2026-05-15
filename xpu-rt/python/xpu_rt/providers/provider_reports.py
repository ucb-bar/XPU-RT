"""Probe-report and matrix writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
    iter_target_cards,
)
from compgen.providers.provider_probe import (
    probe_dialect_provider,
    probe_provider,
)
from compgen.providers.provider_types import (
    ProviderCard,
    ProviderProbeResult,
)
from compgen.targets.target_types import TargetCard

PROBE_REPORT_SCHEMA_VERSION = "solver_backend_status_v1_compat"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True))


def _provider_status_body(
    cards: Iterable[ProviderCard],
    results: Iterable[ProviderProbeResult],
) -> dict:
    by_id = {r.provider_id: r for r in results}
    body = {
        "schema_version": "provider_status_v1",
        "generated_at_utc": _now(),
        "providers": [],
    }
    for card in cards:
        r = by_id[card.provider_id]
        body["providers"].append(
            {
                "provider_id": card.provider_id,
                "integration_level": card.integration_level,
                "status": r.status,
                "blocked_reason": r.blocked_reason,
                "detail": r.detail,
                "version": r.version,
                "supports": list(r.supports),
                "required_env": list(card.required_env),
                "required_commands": list(card.required_commands),
                "required_python_imports": list(card.required_python_imports),
                "target_families": list(card.target_families),
                "contract_kinds": list(card.contract_kinds),
                "emits": list(card.emits),
                "paper_claimable": card.paper_claimable,
            }
        )
    return body


def _target_status_body(cards: Iterable[TargetCard]) -> dict:
    return {
        "schema_version": "target_status_v1",
        "generated_at_utc": _now(),
        "targets": [
            {
                "target_id": c.target_id,
                "family": c.family,
                "vendor": c.vendor,
                "dispatch_modes": list(c.dispatch_modes),
                "memory_tiers": [
                    {"name": t.name, "kind": t.kind, "capacity_bytes": t.capacity_bytes}
                    for t in c.memory_tiers
                ],
            }
            for c in cards
        ],
    }


def _dialect_status_body(
    cards: Iterable[DialectProviderCard],
    results: Iterable[ProviderProbeResult],
) -> dict:
    by_id = {r.provider_id: r for r in results}
    return {
        "schema_version": "dialect_status_v1",
        "generated_at_utc": _now(),
        "dialect_providers": [
            {
                "dialect_provider_id": c.dialect_provider_id,
                "dialect_name": c.dialect_name,
                "integration_level": c.integration_level,
                "status": by_id[c.dialect_provider_id].status,
                "blocked_reason": by_id[c.dialect_provider_id].blocked_reason,
                "detail": by_id[c.dialect_provider_id].detail,
                "consumes": list(c.consumes),
                "emits": list(c.emits),
                "required_env": list(c.required_env),
                "paper_claimable": c.paper_claimable,
            }
            for c in cards
        ],
    }


def _pass_tool_status_stub() -> dict:
    """Placeholder pass-tool status — populated ."""

    return {
        "schema_version": "pass_tool_status_v1",
        "generated_at_utc": _now(),
        "pass_tools": [],
        "notes": "pass-tool registry lands in M-85; this file is a typed empty stub.",
    }


def _write_provider_target_matrix(
    path: Path,
    provider_cards: Iterable[ProviderCard],
    target_cards: Iterable[TargetCard],
    provider_results: Iterable[ProviderProbeResult],
) -> None:
    by_id = {r.provider_id: r for r in provider_results}
    targets = tuple(target_cards)
    fieldnames = ["provider_id", "integration_level", "status"] + [
        t.target_id for t in targets
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for card in provider_cards:
            r = by_id[card.provider_id]
            row = {
                "provider_id": card.provider_id,
                "integration_level": card.integration_level,
                "status": r.status,
            }
            for t in targets:
                if t.family in card.target_families:
                    row[t.target_id] = r.status
                else:
                    row[t.target_id] = ""
            w.writerow(row)


def _write_provider_contract_matrix(
    path: Path,
    provider_cards: Iterable[ProviderCard],
    provider_results: Iterable[ProviderProbeResult],
) -> None:
    by_id = {r.provider_id: r for r in provider_results}
    all_kinds: set[str] = set()
    for c in provider_cards:
        all_kinds.update(c.contract_kinds)
    kinds = sorted(all_kinds)
    fieldnames = ["provider_id", "integration_level", "status"] + kinds
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for card in provider_cards:
            r = by_id[card.provider_id]
            row = {
                "provider_id": card.provider_id,
                "integration_level": card.integration_level,
                "status": r.status,
            }
            for k in kinds:
                row[k] = "yes" if k in card.contract_kinds else ""
            w.writerow(row)


def _baseline_ok(provider_results: Iterable[ProviderProbeResult]) -> bool:
    """Baseline = cffi_c available."""

    by_id = {r.provider_id: r for r in provider_results}
    return by_id.get("cffi_c", None) is not None and by_id["cffi_c"].status == "available"


def _summary_md(
    provider_cards: Iterable[ProviderCard],
    provider_results: Iterable[ProviderProbeResult],
    dialect_cards: Iterable[DialectProviderCard],
    dialect_results: Iterable[ProviderProbeResult],
    baseline_ok: bool,
) -> str:
    p_results = {r.provider_id: r for r in provider_results}
    d_results = {r.provider_id: r for r in dialect_results}
    lines = [
        "# Extension provider probe summary",
        "",
        f"Generated: {_now()}",
        "",
        f"Baseline available (cffi_c): **{'yes' if baseline_ok else 'NO'}**",
        "",
        "## Providers",
        "",
        "| provider | integration_level | status | blocked_reason | detail |",
        "|---|---|---|---|---|",
    ]
    for c in provider_cards:
        r = p_results[c.provider_id]
        lines.append(
            f"| {c.provider_id} | {c.integration_level} | "
            f"{r.status} | {r.blocked_reason or ''} | {r.detail or ''} |"
        )
    lines += [
        "",
        "## Dialect providers",
        "",
        "| dialect | integration_level | status | blocked_reason | detail |",
        "|---|---|---|---|---|",
    ]
    for c in dialect_cards:
        r = d_results[c.dialect_provider_id]
        lines.append(
            f"| {c.dialect_provider_id} | {c.integration_level} | {r.status} | "
            f"{r.blocked_reason or ''} | {r.detail or ''} |"
        )
    return "\n".join(lines) + "\n"


def write_probe_reports(out_dir: Path) -> dict:
    """Write the full probe report set into ``out_dir``.

    Returns a dict of relative paths under ``out_dir`` for caller
    convenience (e.g. trust-report wiring).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_cards = tuple(iter_provider_cards())
    target_cards = tuple(iter_target_cards())
    dialect_cards = tuple(iter_dialect_cards())

    provider_results = tuple(probe_provider(c) for c in provider_cards)
    dialect_results = tuple(probe_dialect_provider(c) for c in dialect_cards)

    paths = {
        "provider_status": out_dir / "provider_status.json",
        "target_status": out_dir / "target_status.json",
        "dialect_status": out_dir / "dialect_status.json",
        "pass_tool_status": out_dir / "pass_tool_status.json",
        "provider_target_matrix": out_dir / "provider_target_matrix.csv",
        "provider_contract_matrix": out_dir / "provider_contract_matrix.csv",
        "probe_summary": out_dir / "probe_summary.md",
    }

    _write_json(
        paths["provider_status"],
        _provider_status_body(provider_cards, provider_results),
    )
    _write_json(paths["target_status"], _target_status_body(target_cards))
    _write_json(
        paths["dialect_status"],
        _dialect_status_body(dialect_cards, dialect_results),
    )
    _write_json(paths["pass_tool_status"], _pass_tool_status_stub())

    _write_provider_target_matrix(
        paths["provider_target_matrix"],
        provider_cards,
        target_cards,
        provider_results,
    )
    _write_provider_contract_matrix(
        paths["provider_contract_matrix"],
        provider_cards,
        provider_results,
    )

    baseline_ok = _baseline_ok(provider_results)
    paths["probe_summary"].write_text(
        _summary_md(
            provider_cards,
            provider_results,
            dialect_cards,
            dialect_results,
            baseline_ok,
        )
    )

    return {k: v.relative_to(out_dir).as_posix() for k, v in paths.items()}
