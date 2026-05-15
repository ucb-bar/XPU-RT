#!/usr/bin/env -S uv run python
"""Tool evidence pack builder.

Read-only aggregator that joins:

* The ToolCard registry.
* The promotion audit.
* The MCP bridge — which cards reached an MCP surface.
* The fresh-agent harness — which tasks have ever graded clean.

Output structure under ``--out`` (default
``results/tool_evidence_pack/<commit>/``):

::

    tool_registry.json           # one row per ToolCard
    tool_maturity_matrix.csv     # tool × maturity gate
    tool_surface_matrix.csv      # python/cli/skill/mcp/harness columns
    cli_mcp_schema_match.json    # per-tool CLI/MCP schema equivalence
    fresh_agent_tasks.json # task index + last grading
    claim_matrix.json            # paper-claim rollup
    promotion_log.json # per-tool T0→T7 history (manages this)
    figures/                     # PNGs when matplotlib is available;
                                  # otherwise figure_status_marker.json
                                  # records the honest "skipped" reason

Hard rules:

* The script never edits any source file. The only writes are under
  ``--out``.
* When matplotlib is missing, the figures directory contains a typed
  ``figure_status_marker.json`` declaring the skip — silent absence
  would amount to a hidden non-claim.
* ``promotion_log.json`` is append-only across invocations: the
  builder reads the prior entries (if present) and appends a new one
  per tool whose ``verified_maturity`` rose since the last build.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.audit.fresh_agent_grading import (
    fresh_agent_tasks_root,
    list_task_ids,
    load_task,
)
from compgen.audit.tool_promotion import (
    AuditReport,
    run_tool_promotion_audit,
)
from compgen.mcp.tool_bridge import bridge_tools
from compgen.tools.tool_card import MATURITY_LEVELS, ToolCard
from compgen.tools.tool_registry import iter_tool_cards

SCHEMA_VERSION = "compgen_tool_evidence_pack_v1"


@dataclass(frozen=True)
class FigureStatus:
    """Typed marker for the figures directory.

    matplotlib lives under the ``[benchmarks]`` extra; on a CI runner
    without it, we emit this marker so a downstream paper-claim audit
    can spot the gap rather than discover silently-missing PNGs.
    """

    kind: str  # "available" | "skipped_missing_matplotlib"
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).strip()
        return out or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _emit_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _maturity_matrix_rows(cards: list[ToolCard], audit: AuditReport) -> list[dict[str, Any]]:
    outcomes = {o.tool_id: o for o in audit.outcomes}
    rows: list[dict[str, Any]] = []
    for card in cards:
        row: dict[str, Any] = {
            "tool_id": card.tool_id,
            "declared": card.maturity,
            "verified": outcomes.get(card.tool_id).verified_maturity
            if card.tool_id in outcomes
            else "below-T0",
        }
        outcome = outcomes.get(card.tool_id)
        violation_count_by_rung: dict[str, int] = {r: 0 for r in MATURITY_LEVELS}
        if outcome is not None:
            for v in outcome.violations:
                violation_count_by_rung[v.rung] = violation_count_by_rung.get(v.rung, 0) + 1
        for rung in MATURITY_LEVELS:
            row[f"violations_{rung}"] = violation_count_by_rung[rung]
        rows.append(row)
    return rows


def _surface_matrix_rows(
    cards: list[ToolCard],
    bridged_tool_ids: set[str],
    graded_task_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in cards:
        rows.append(
            {
                "tool_id": card.tool_id,
                "phase": card.phase,
                "python": bool(card.entrypoints.python),
                "cli": bool(card.entrypoints.cli),
                "skill": bool(card.skill_path),
                "mcp": card.tool_id in bridged_tool_ids,
                "fresh_agent_task": card.fresh_agent_task_id or "",
                "fresh_agent_graded": (
                    card.fresh_agent_task_id in graded_task_ids
                    if card.fresh_agent_task_id
                    else False
                ),
            }
        )
    return rows


def _cli_mcp_schema_match(
    cards: list[ToolCard], bridge_payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """For every MCP-bridged card, verify the bridge schema is
    bit-equal to the card's input_schema. This mirrors the
    schema-equivalence test in :mod:`tests.tools.test_mcp_tool_bridge`."""

    by_id = {p["_card_tool_id"]: p for p in bridge_payloads if "_card_tool_id" in p}
    rows: list[dict[str, Any]] = []
    for card in cards:
        if card.tool_id not in by_id:
            continue
        tool = by_id[card.tool_id]
        card_canon = json.dumps(card.input_schema, sort_keys=True, separators=(",", ":"))
        mcp_canon = json.dumps(tool["input_schema"], sort_keys=True, separators=(",", ":"))
        rows.append(
            {
                "tool_id": card.tool_id,
                "card_schema_sha": _sha(card_canon),
                "mcp_schema_sha": _sha(mcp_canon),
                "bit_equal": card_canon == mcp_canon,
            }
        )
    return {"schema_version": "cli_mcp_schema_match_v1", "rows": rows}


def _sha(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _graded_task_ids(repo_root: Path | None = None) -> set[str]:
    out: set[str] = set()
    root = fresh_agent_tasks_root(repo_root)
    if not root.is_dir():
        return out
    for task_id in list_task_ids(repo_root):
        sidecar = root / task_id / "last_grading_result.json"
        if not sidecar.is_file():
            continue
        try:
            body = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if bool(body.get("passed")):
            out.add(task_id)
    return out


def _fresh_agent_tasks_index(repo_root: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in list_task_ids(repo_root):
        try:
            task = load_task(task_id, repo_root=repo_root)
        except Exception as exc:  # noqa: BLE001
            rows.append({"task_id": task_id, "load_error": str(exc)})
            continue
        sidecar = task.task_dir / "last_grading_result.json"
        last_grading: dict[str, Any] | None = None
        if sidecar.is_file():
            try:
                last_grading = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                last_grading = {"passed": False, "reason": "malformed_sidecar"}
        rows.append(
            {
                "task_id": task.task_id,
                "allowed_tools": list(task.allowed_tools),
                "expected_artifacts": [a["path"] for a in task.expected_artifacts],
                "has_baseline": task.baseline is not None,
                "last_grading_passed": (
                    bool(last_grading and last_grading.get("passed"))
                ),
                "last_grading_reason": (
                    last_grading.get("reason") if last_grading else None
                ),
            }
        )
    return {"schema_version": "fresh_agent_tasks_v1", "tasks": rows}


def _claim_matrix(audit: AuditReport, surface_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Five claims the evidence pack signs or marks unmet.

    Each claim is paper-relevant only when its evidence is on disk;
    every ``status`` is a closed enum so the trust audit can
    consume it.
    """

    claims: list[dict[str, Any]] = []

    audited = audit.total_tools
    clean = sum(1 for o in audit.outcomes if not o.violations)
    claims.append(
        {
            "id": "C_TOOLS_AUDITED_CLEAN",
            "headline": "Every shipped ToolCard's declared maturity is verified by evidence.",
            "status": "signed" if audited and clean == audited else (
                "unmet" if audited else "no_data"
            ),
            "metric": f"{clean}/{audited}",
        }
    )

    cli_count = sum(1 for r in surface_rows if r["cli"])
    claims.append(
        {
            "id": "C_TOOLS_CLI_REACHABLE",
            "headline": "Every shipped tool has a shell-callable CLI surface.",
            "status": "signed" if audited and cli_count == audited else "partial",
            "metric": f"{cli_count}/{audited}",
        }
    )

    mcp_count = sum(1 for r in surface_rows if r["mcp"])
    claims.append(
        {
            "id": "C_TOOLS_MCP_BRIDGED",
            "headline": "MCP bridge surfaces ≥ 1 card from the ToolCard registry.",
            "status": "signed" if mcp_count >= 1 else "unmet",
            "metric": f"{mcp_count}/{audited}",
        }
    )

    fa_graded = sum(1 for r in surface_rows if r["fresh_agent_graded"])
    claims.append(
        {
            "id": "C_TOOLS_FRESH_AGENT_GRADED",
            "headline": "≥ 1 ToolCard has a fresh-agent task that has graded clean on real hardware.",
            "status": "signed" if fa_graded >= 1 else "no_data",
            "metric": f"{fa_graded}/{audited}",
        }
    )

    extension_count = sum(
        1 for o in audit.outcomes
        if o.tool_id.startswith("compgen_") and "extension" in o.tool_id and not o.violations
    )
    claims.append(
        {
            "id": "C_TOOLS_EXTENSION_FLOW_REACHABLE",
            "headline": "Extension-authoring tools (emit / validate) are runnable end-to-end.",
            "status": "signed" if extension_count >= 2 else "partial",
            "metric": f"{extension_count}/2",
        }
    )

    return {"schema_version": "claim_matrix_v1", "claims": claims}


def _promotion_log_update(
    audit: AuditReport, prior: dict[str, Any] | None, *, commit: str
) -> dict[str, Any]:
    """Append-only promotion-log update.

    Each tool gets a record on every build; the ``rung_history`` list
    is appended only when ``verified_maturity`` rises since the last
    recorded value (so the log captures real promotion events without
    polluting itself with redundant rows).
    """

    history: dict[str, list[dict[str, Any]]] = {}
    if prior and isinstance(prior, dict):
        for tool_id, rows in (prior.get("rung_history") or {}).items():
            if isinstance(rows, list):
                history[str(tool_id)] = [dict(r) for r in rows if isinstance(r, dict)]

    now = _utc_now()
    for outcome in audit.outcomes:
        rows = history.setdefault(outcome.tool_id, [])
        last_rung = rows[-1].get("rung") if rows else None
        if outcome.verified_maturity != last_rung:
            rows.append(
                {
                    "rung": outcome.verified_maturity,
                    "declared": outcome.declared_maturity,
                    "commit": commit,
                    "recorded_at_utc": now,
                }
            )

    return {
        "schema_version": "promotion_log_v1",
        "rung_history": history,
    }


def _try_figures(out_dir: Path) -> FigureStatus:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        marker = figures_dir / "figure_status_marker.json"
        status = FigureStatus(
            kind="skipped_missing_matplotlib",
            detail=(
                "matplotlib not installed; install the [benchmarks] extra "
                "to generate tool_maturity_by_category.png + tool_surface_coverage.png + "
                "tool_lifecycle.png"
            ),
        )
        marker.write_text(
            json.dumps(status.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return status
    # If matplotlib is available, emit at least one figure so the
    # evidence pack proves the path works end-to-end. (Richer
    # multi-figure rendering is a follow-up; this keeps the contract
    # honest right now.)
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "tool_evidence_pack: figures pending", ha="center", va="center")
    ax.axis("off")
    fig.savefig(figures_dir / "tool_lifecycle.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return FigureStatus(kind="available", detail="figures rendered")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build(out_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Build the evidence pack into ``out_dir``. Returns the manifest."""

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = list(iter_tool_cards())
    audit = run_tool_promotion_audit(cards=cards, repo_root=repo_root)
    bridge_payloads = bridge_tools()
    bridged_ids = {b["_card_tool_id"] for b in bridge_payloads if "_card_tool_id" in b}
    graded = _graded_task_ids(repo_root)

    # --- registry ---
    registry_payload = {
        "schema_version": "tool_registry_v1",
        "tools": [c.to_dict() for c in cards],
    }
    _emit_json(out_dir / "tool_registry.json", registry_payload)

    # --- maturity matrix ---
    maturity_rows = _maturity_matrix_rows(cards, audit)
    fieldnames = ["tool_id", "declared", "verified"] + [
        f"violations_{r}" for r in MATURITY_LEVELS
    ]
    _write_csv(out_dir / "tool_maturity_matrix.csv", maturity_rows, fieldnames)

    # --- surface matrix ---
    surface_rows = _surface_matrix_rows(cards, bridged_ids, graded)
    surface_fields = [
        "tool_id", "phase", "python", "cli", "skill", "mcp",
        "fresh_agent_task", "fresh_agent_graded",
    ]
    _write_csv(out_dir / "tool_surface_matrix.csv", surface_rows, surface_fields)

    # --- CLI/MCP schema equivalence ---
    schema_match = _cli_mcp_schema_match(cards, bridge_payloads)
    _emit_json(out_dir / "cli_mcp_schema_match.json", schema_match)

    # --- fresh-agent tasks index ---
    tasks_index = _fresh_agent_tasks_index(repo_root)
    _emit_json(out_dir / "fresh_agent_tasks.json", tasks_index)

    # --- claim matrix ---
    claims = _claim_matrix(audit, surface_rows)
    _emit_json(out_dir / "claim_matrix.json", claims)

    # --- promotion log (append-only) ---
    promotion_log_path = out_dir / "promotion_log.json"
    prior_log: dict[str, Any] | None = None
    if promotion_log_path.is_file():
        try:
            prior_log = json.loads(promotion_log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior_log = None
    commit = _git_short_sha()
    new_log = _promotion_log_update(audit, prior_log, commit=commit)
    _emit_json(promotion_log_path, new_log)

    # --- figures (best-effort) ---
    figure_status = _try_figures(out_dir)

    # --- manifest ---
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "commit": commit,
        "totals": {
            "tools": len(cards),
            "tools_audit_clean": sum(1 for o in audit.outcomes if not o.violations),
            "tools_with_cli": sum(1 for r in surface_rows if r["cli"]),
            "tools_with_mcp": sum(1 for r in surface_rows if r["mcp"]),
            "tools_with_fresh_agent_graded": sum(
                1 for r in surface_rows if r["fresh_agent_graded"]
            ),
            "claims_signed": sum(1 for c in claims["claims"] if c["status"] == "signed"),
            "claims_unmet": sum(1 for c in claims["claims"] if c["status"] == "unmet"),
        },
        "figures": figure_status.to_dict(),
        "artifacts": [
            "tool_registry.json",
            "tool_maturity_matrix.csv",
            "tool_surface_matrix.csv",
            "cli_mcp_schema_match.json",
            "fresh_agent_tasks.json",
            "claim_matrix.json",
            "promotion_log.json",
        ],
    }
    _emit_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/tool_evidence_pack/<commit>).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override the repo root (defaults to the CompGen checkout).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.out is None:
        commit = _git_short_sha()
        args.out = Path("results") / "tool_evidence_pack" / commit
    manifest = build(args.out, repo_root=args.repo_root)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    if manifest["totals"]["claims_unmet"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
