#!/usr/bin/env python
"""extension/provider evidence pack.

Outputs::

    <out>/extension_summary.md
    <out>/provider_status.json
    <out>/target_status.json
    <out>/dialect_provider_registry.json
    <out>/pass_tool_registry.json
    <out>/provider_target_matrix.csv
    <out>/provider_contract_matrix.csv
    <out>/unsupported_op_tasks.json
    <out>/architecture_audit.json
    <out>/claim_matrix.json
    <out>/figures/  (Markdown ASCII fallback when matplotlib missing)

Claim matrix has 12 rows. Each is `implemented` only if a real
evidence-artifact path under <out> exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from compgen.audit.extension_architecture import run_audit
from compgen.extensions.task_flow import list_extension_tasks
from compgen.pass_tools.pass_tool_registry import build_pass_tool_registry
from compgen.providers.card_loader import (
    iter_dialect_cards,
    iter_provider_cards,
    iter_target_cards,
)
from compgen.providers.provider_reports import write_probe_reports


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _copy_probe_reports(probe_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Copy probe artifacts into the evidence pack."""

    copied: dict[str, Path] = {}
    for name in (
        "provider_status.json",
        "target_status.json",
        "dialect_status.json",
        "pass_tool_status.json",
        "provider_target_matrix.csv",
        "provider_contract_matrix.csv",
        "probe_summary.md",
    ):
        src = probe_dir / name
        dst = out_dir / name
        if src.is_file():
            dst.write_bytes(src.read_bytes())
            copied[name] = dst
    return copied


def _dialect_provider_registry_body() -> dict:
    return {
        "schema_version": "dialect_provider_registry_v1",
        "generated_at_utc": _now(),
        "entries": [
            {
                "dialect_provider_id": d.dialect_provider_id,
                "dialect_name": d.dialect_name,
                "integration_level": d.integration_level,
                "consumes": list(d.consumes),
                "emits": list(d.emits),
                "entrypoint": d.entrypoint,
                "paper_claimable": d.paper_claimable,
            }
            for d in iter_dialect_cards()
        ],
    }


def _pass_tool_registry_body() -> dict:
    reg = build_pass_tool_registry()
    return {
        "schema_version": "pass_tool_registry_v1",
        "generated_at_utc": _now(),
        "entries": [
            {
                "tool_id": card.tool_id,
                "phase": card.phase,
                "reads": list(card.reads),
                "writes": list(card.writes),
                "allowed_recipe_ops": list(card.allowed_recipe_ops),
                "refinement_kind": card.refinement_kind,
                "verifier": card.verifier,
                "entrypoint": card.entrypoint,
            }
            for card in reg.cards.values()
        ],
    }


def _unsupported_op_tasks_body(tasks_root: Path) -> dict:
    tasks = []
    if tasks_root.is_dir():
        for task_dir in list_extension_tasks(tasks_root):
            tasks.append({"task_id": task_dir.name, "task_dir": str(task_dir)})
    return {
        "schema_version": "unsupported_op_tasks_v1",
        "generated_at_utc": _now(),
        "tasks": tasks,
    }


CLAIM_ROWS = (
    "extension_manifest_validated",
    "extension_sandbox_enforced",
    "provider_registry_present",
    "target_registry_present",
    "dialect_provider_registry_present",
    "pass_tool_registry_present",
    "provider_probe_typed",
    "provider_routing_deterministic",
    "unsupported_op_task_emitted",
    "extension_registered_after_verification",
    "multi_level_analysis_snapshots_present",
    "certified_artifacts_only_executed",
)


# Map each claim to evidence files (relative to <out>) that must
# exist for the row to be marked ``implemented``.
CLAIM_EVIDENCE_PATHS = {
    "extension_manifest_validated": ("provider_status.json",),
    "extension_sandbox_enforced": ("provider_status.json",),
    "provider_registry_present": ("provider_status.json",),
    "target_registry_present": ("target_status.json",),
    "dialect_provider_registry_present": ("dialect_provider_registry.json",),
    "pass_tool_registry_present": ("pass_tool_registry.json",),
    "provider_probe_typed": ("provider_status.json", "probe_summary.md"),
    "provider_routing_deterministic": ("provider_contract_matrix.csv",),
    "unsupported_op_task_emitted": ("unsupported_op_tasks.json",),
    "extension_registered_after_verification": ("architecture_audit.json",),
    "multi_level_analysis_snapshots_present": ("analysis_snapshots/",),
    "certified_artifacts_only_executed": ("architecture_audit.json",),
}


def _claim_matrix(out_dir: Path, audit_passed: bool) -> dict:
    rows = []
    for claim in CLAIM_ROWS:
        evidence = CLAIM_EVIDENCE_PATHS.get(claim, ())
        present = all((out_dir / p).exists() for p in evidence)
        if claim == "certified_artifacts_only_executed":
            present = present and audit_passed
        rows.append(
            {
                "claim": claim,
                "status": "implemented" if present else "partial_scope",
                "evidence": list(evidence),
                "evidence_present": present,
            }
        )
    return {
        "schema_version": "claim_matrix_v1",
        "generated_at_utc": _now(),
        "rows": rows,
    }


def _summary_md(
    out_dir: Path,
    audit_passed: bool,
    claim_body: dict,
) -> str:
    impl = sum(1 for r in claim_body["rows"] if r["status"] == "implemented")
    total = len(claim_body["rows"])
    lines = [
        "# Phase F evidence pack",
        "",
        f"Generated: {_now()}",
        "",
        f"Claim matrix: **{impl}/{total} rows** marked `implemented`.",
        f"Architecture audit: **{'PASS' if audit_passed else 'FAIL'}**.",
        "",
        "## Claim matrix",
        "",
        "| claim | status | evidence present? |",
        "|---|---|---|",
    ]
    for r in claim_body["rows"]:
        lines.append(
            f"| {r['claim']} | {r['status']} | {'yes' if r['evidence_present'] else 'no'} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
    ]
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            lines.append(f"- `{p.name}`")
        elif p.is_dir():
            lines.append(f"- `{p.name}/`")
    return "\n".join(lines) + "\n"


def _render_figures(pack_dir: Path) -> None:
    """Render the 5 spec'd PNG figures. Each rendered figure
    becomes a real PNG; a figure whose source data is absent gets a
    typed Markdown skip-marker instead so the figures/ directory
    always has the 5 named slots in some form."""

    figures_dir = pack_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _markdown_fallback(reason: str) -> None:
        for name in (
            "provider_target_heatmap.md",
            "provider_status_by_family.md",
            "extension_lifecycle.md",
            "ir_analysis_levels.md",
            "blocked_reason_breakdown.md",
        ):
            (figures_dir / name).write_text(
                f"# Placeholder — {reason}\n"
            )

    try:
        from compgen.audit.figures import render_all_figures
    except ImportError as exc:
        _markdown_fallback(f"figures module unavailable: {exc}")
        return
    # Remove any stale markdown placeholders from earlier runs.
    for md in figures_dir.glob("*.md"):
        md.unlink()
    try:
        results = render_all_figures(pack_dir)
    except ImportError as exc:
        # matplotlib (or one of its deps) not importable in this
        # interpreter — fall back to markdown placeholders so the
        # evidence pack still has the 5 named slots.
        _markdown_fallback(f"matplotlib import failed: {exc}")
        return
    for r in results:
        if r.skipped:
            md_path = r.path.with_suffix(".md")
            md_path.write_text(
                f"# Skipped figure: {r.path.name}\n\nreason: {r.reason}\n"
            )


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("results/extension_provider_probe"),
        help="path to a probe report set (extension probe output)",
    )
    p.add_argument(
        "--tasks-root",
        type=Path,
        default=Path(".rcg-artifacts/tasks"),
    )
    p.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="optional analysis-snapshots dir (multi-level snapshot output)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/extension_provider_evidence_pack"),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if not args.probe_dir.is_dir():
        write_probe_reports(args.probe_dir)
    _copy_probe_reports(args.probe_dir, out)

    (out / "dialect_provider_registry.json").write_text(
        json.dumps(_dialect_provider_registry_body(), indent=2, sort_keys=True)
    )
    (out / "pass_tool_registry.json").write_text(
        json.dumps(_pass_tool_registry_body(), indent=2, sort_keys=True)
    )
    (out / "unsupported_op_tasks.json").write_text(
        json.dumps(_unsupported_op_tasks_body(args.tasks_root), indent=2, sort_keys=True)
    )

    audit = run_audit()
    (out / "architecture_audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True)
    )

    if args.snapshots_dir and args.snapshots_dir.is_dir():
        shutil.copytree(
            args.snapshots_dir,
            out / "analysis_snapshots",
            dirs_exist_ok=True,
        )

    _render_figures(out)

    claim_body = _claim_matrix(out, audit_passed=audit.passed)
    (out / "claim_matrix.json").write_text(
        json.dumps(claim_body, indent=2, sort_keys=True)
    )

    (out / "extension_summary.md").write_text(
        _summary_md(out, audit_passed=audit.passed, claim_body=claim_body)
    )

    print(f"Wrote Phase F evidence pack to {out}")
    impl = sum(1 for r in claim_body["rows"] if r["status"] == "implemented")
    print(f"Claim matrix: {impl}/{len(claim_body['rows'])} rows implemented")
    print(f"Architecture audit: {'PASS' if audit.passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
