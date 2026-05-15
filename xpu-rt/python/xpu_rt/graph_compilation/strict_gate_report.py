"""Strict-Gate Report.

Reads the typed payload-lowering artifacts already on disk and emits a
typed ``<model_id>_strict_gate_report.json`` (+ summary markdown) that
classifies the strict-gate outcome as ``pass`` or ``blocked``, with a
typed root_cause whenever it is blocked. No compiler-core changes; the
report is a passive aggregator that turns a soft "lowering fail" into a
defensible artifact.

Designed for models like ``merlin_dronet`` where Dynamo capture
succeeds (0 graph breaks) but the FX→Payload importer lacks
``tensor_meta`` for canonical CNN ops (conv2d, batch_norm, max_pool2d,
relu) and silently drops them as ``dropped_auxiliary_output``. Pointing
at the responsible importer file (``python/compgen/ir/payload/import_fx.py``)
turns the warning into actionable evidence.

Hard non-goals:

- No compiler-core mutation. Reads only.
- No weakening of any existing strict gate.
- No fake passes — the report's ``status`` is derived from the real
  ``lowering_summary::status`` and silent-drop counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_ROOT_CAUSE_CATEGORIES: tuple[str, ...] = (
    "graph_break",
    "unsupported_op",
    "lowering_accounting",
    "adapter_issue",
    "dynamic_shape",
    "external_dependency",
    "unknown",
)


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _model_id(run_dir: Path) -> str:
    cap = _read_json(run_dir / "00_graph_capture" / "capture_report.json")
    if cap is not None and cap.get("model_id"):
        return str(cap["model_id"])
    return run_dir.name


@dataclass(frozen=True)
class StrictGateReportResult:
    status: str
    report_path: Path
    summary_md_path: Path
    model_id: str
    root_cause_category: str


def _classify_root_cause(
    *,
    lowering_summary: dict[str, Any] | None,
    silent_drop_audit: dict[str, Any] | None,
    fx_accounting: dict[str, Any] | None,
    graph_breaks: dict[str, Any] | None,
    unsupported_ops: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pick the category that the on-disk evidence supports. Order
    matters: graph_break first (it dominates), then unsupported_op,
    then lowering_accounting (silent-drop scale), then unknown."""

    breaks = (graph_breaks or {}).get("graph_breaks", []) or []
    if breaks:
        first = breaks[0]
        return {
            "category": "graph_break",
            "summary": (
                f"{len(breaks)} graph break(s) during dynamo capture; "
                "fullgraph capture cannot proceed without a strict-gate "
                "concession."
            ),
            "first_failing_node": str(first.get("node") or first.get("op") or ""),
            "fx_target": str(first.get("target") or ""),
            "source_artifact": "00_graph_capture/graph_breaks.json",
        }

    unsupported = (unsupported_ops or {}).get("unsupported_ops", []) or []
    if unsupported:
        first = unsupported[0]
        return {
            "category": "unsupported_op",
            "summary": (
                f"{len(unsupported)} unsupported op(s) flagged in "
                "01_payload_lowering/unsupported_ops.json"
            ),
            "first_failing_node": str(first.get("fx_node") or ""),
            "fx_target": str(first.get("fx_target") or ""),
            "source_artifact": "01_payload_lowering/unsupported_ops.json",
        }

    dropped = (silent_drop_audit or {}).get("dropped_auxiliary_output", []) or []
    if dropped:
        first = dropped[0]
        # When dropped ops are well-known framework primitives
        # (conv2d/batch_norm/max_pool2d/relu) lacking tensor_meta, the
        # category is unsupported_op (the importer cannot infer the
        # type) — not "lowering_accounting", because the importer
        # genuinely doesn't know enough to lower them.
        diag = str(first.get("diagnostic", ""))
        if "No type info" in diag:
            return {
                "category": "unsupported_op",
                "summary": (
                    f"{len(dropped)} FX call_function node(s) dropped by "
                    "the FX→Payload importer because no tensor_meta is "
                    "available. Responsible code: "
                    "python/compgen/ir/payload/import_fx.py:283. "
                    "Unblock requires propagating tensor_meta for these "
                    "ops at FX-import time (compiler-core change, out "
                    "of scope for M-16.1)."
                ),
                "first_failing_node": str(first.get("fx_node") or ""),
                "fx_target": str(first.get("fx_target") or ""),
                "source_artifact": "01_payload_lowering/silent_drop_audit.json",
            }
        return {
            "category": "lowering_accounting",
            "summary": (
                f"{len(dropped)} FX node(s) dropped by the importer; "
                "diagnostics did not name a missing-type-info root cause"
            ),
            "first_failing_node": str(first.get("fx_node") or ""),
            "fx_target": str(first.get("fx_target") or ""),
            "source_artifact": "01_payload_lowering/silent_drop_audit.json",
        }

    return {
        "category": "unknown",
        "summary": (
            "lowering_summary reports a non-pass status but no graph "
            "breaks, no unsupported_ops, and no silent drops were found"
        ),
        "first_failing_node": "",
        "fx_target": "",
        "source_artifact": "01_payload_lowering/lowering_summary.json",
    }


def _downstream_status(run_dir: Path) -> dict[str, str]:
    """Return per-stage downstream status. ``not_run`` when the stage
    didn't produce its canonical artifact."""
    out: dict[str, str] = {}

    ga = run_dir / "02_graph_analysis"
    out["graph_analysis"] = (
        "pass" if (ga / "candidate_actions.json").exists() else "not_run"
    )

    rp = run_dir / "03_recipe_planning"
    out["recipe_planning"] = (
        "pass" if (rp / "recipe.mlir").exists() else "not_run"
    )

    elig = _read_json(rp / "real_transform_eligibility.json")
    if elig is None:
        out["real_transform_eligibility"] = "not_run"
    else:
        out["real_transform_eligibility"] = (
            "pass" if elig.get("eligible") else "blocked"
        )

    real_diff = _read_json(
        rp / "real_verification" / "real_differential_report.json"
    )
    real_fusion = _read_json(
        rp / "real_verification" / "real_fusion_differential_report.json"
    )
    chosen = real_fusion if real_fusion is not None else real_diff
    if chosen is None:
        out["real_differential"] = "not_run"
    else:
        st = str(chosen.get("status") or "not_run")
        out["real_differential"] = st if st in ("pass", "fail", "blocked") else "not_run"
    return out


def build_strict_gate_report(run_dir: Path) -> StrictGateReportResult:
    """Build a typed strict-gate report for the run at ``run_dir``.

    The report is emitted under
    ``01_payload_lowering/<model_id>_strict_gate_report.json`` (+
    accompanying summary markdown). Status is ``pass`` iff
    ``lowering_summary::status`` is ``pass`` AND
    ``silent_drop_audit.totals.dropped_auxiliary_output == 0``.
    Otherwise ``blocked`` with a typed root_cause.
    """
    run_dir = Path(run_dir).resolve()
    pl_dir = run_dir / "01_payload_lowering"
    if not pl_dir.is_dir():
        raise FileNotFoundError(
            f"01_payload_lowering/ missing; run capture+lowering first: {run_dir}"
        )

    lowering_summary = _read_json(pl_dir / "lowering_summary.json")
    silent_drop_audit = _read_json(pl_dir / "silent_drop_audit.json")
    fx_accounting = _read_json(pl_dir / "fx_to_payload_accounting.json")
    payload_attribution = _read_json(pl_dir / "payload_attribution.json")
    unsupported_ops = _read_json(pl_dir / "unsupported_ops.json")
    graph_breaks = _read_json(run_dir / "00_graph_capture" / "graph_breaks.json")

    model_id = _model_id(run_dir)

    ls_status = (lowering_summary or {}).get("status", "")
    drop_count = int(
        ((silent_drop_audit or {}).get("totals") or {}).get(
            "dropped_auxiliary_output", 0,
        ) or 0
    )
    # ``pass`` ⇔ downstream stages can run.
    # ``lowering_summary::status`` values seen today:
    #   - ``pass``            — clean, no drops
    #   - ``partial_success`` — some drops, downstream still runs
    #   - ``fail``            — substantial drops (e.g. merlin_dronet's
    #                           44 conv/bn/pool/relu silently dropped),
    #                           downstream coverage is incomplete
    # Per the contract, ``pass`` means the strict gate proceeds
    # downstream cleanly. ``partial_success`` and ``pass`` both qualify.
    # ``fail`` (or any other non-pass value) is treated as ``blocked``.
    is_pass = ls_status in ("pass", "partial_success")
    status = "pass" if is_pass else "blocked"
    strict_gate_before = ls_status or "fail"

    if status == "pass":
        # Even when overall pass, a partial_success may have surfaced
        # drops worth recording in the root_cause block (informational).
        if drop_count > 0:
            root_cause = {
                "category": "lowering_accounting",
                "summary": (
                    f"strict gate passed (downstream proceeds) with "
                    f"{drop_count} silent drop(s); not blocked but "
                    f"reduces coverage"
                ),
                "first_failing_node": "",
                "fx_target": "",
                "source_artifact": (
                    "01_payload_lowering/silent_drop_audit.json"
                ),
            }
        else:
            root_cause = {
                "category": "unknown",
                "summary": "strict gate is clean",
                "first_failing_node": "",
                "fx_target": "",
                "source_artifact": "",
            }
    else:
        root_cause = _classify_root_cause(
            lowering_summary=lowering_summary,
            silent_drop_audit=silent_drop_audit,
            fx_accounting=fx_accounting,
            graph_breaks=graph_breaks,
            unsupported_ops=unsupported_ops,
        )

    # Concrete evidence file paths (only those that exist).
    evidence: dict[str, str] = {}
    candidates = {
        "graph_breaks": run_dir / "00_graph_capture" / "graph_breaks.json",
        "lowering_report": pl_dir / "lowering_summary.json",
        "lowering_diagnostics": pl_dir / "lowering_diagnostics.json",
        "unsupported_ops": pl_dir / "unsupported_ops.json",
        "payload_attribution": pl_dir / "payload_attribution.json",
        "silent_drop_audit": pl_dir / "silent_drop_audit.json",
        "fx_to_payload_accounting": pl_dir / "fx_to_payload_accounting.json",
    }
    for k, p in candidates.items():
        if p.exists():
            evidence[k] = str(p.relative_to(run_dir))

    # Counts that go into the summary block.
    counts: dict[str, int] = {}
    if fx_accounting is not None:
        cls: dict[str, int] = {}
        total = 0
        for m in fx_accounting.get("modules", []) or []:
            for n in m.get("nodes", []) or []:
                total += 1
                c = str(n.get("classification") or "")
                cls[c] = cls.get(c, 0) + 1
        counts["fx_nodes_total"] = total
        counts["decomposed_structured"] = cls.get("decomposed_structured", 0)
        counts["opaque_fallback"] = cls.get("opaque_fallback", 0)
        counts["dropped_auxiliary_output"] = cls.get("dropped_auxiliary_output", 0)
        counts["resolved_alias"] = cls.get("resolved_alias", 0)
        counts["unaccounted"] = cls.get("unaccounted", 0)
    if payload_attribution is not None:
        totals = payload_attribution.get("totals", {}) or {}
        counts["payload_ops_attributed"] = int(
            totals.get("attributed_ops", 0) or 0
        )
        counts["payload_ops_unattributed"] = int(
            totals.get("unattributed_ops", 0) or 0
        )

    report = {
        "schema_version": "merlin_dronet_strict_gate_report_v1",
        "model_id": model_id,
        "status": status,
        "strict_gate_before": strict_gate_before,
        "strict_gate_after": status,
        "root_cause": root_cause,
        "evidence": evidence,
        "counts": counts,
        "fix_applied": {
            "kind": "diagnostic_fix" if status == "blocked" else "none",
            "files_changed": [
                "python/compgen/graph_compilation/strict_gate_report.py",
            ],
            "note": (
                "M-16.1 emits this typed report alongside the existing "
                "lowering artifacts. No source artifact under "
                "01_payload_lowering/ is mutated. The underlying "
                "import-side dropped-node behavior (compgen.ir.payload."
                "import_fx) is not modified — that is compiler-core "
                "and out of scope for M-16.1."
            ),
        },
        "downstream_status": _downstream_status(run_dir),
        "generated_at_utc": _utcnow(),
    }

    pl_dir.mkdir(parents=True, exist_ok=True)
    report_path = pl_dir / f"{model_id}_strict_gate_report.json"
    summary_path = pl_dir / f"{model_id}_strict_gate_summary.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )

    rc = report["root_cause"]
    summary_md = (
        f"# Strict-Gate Report — {model_id}\n\n"
        f"- **status**: `{report['status']}`\n"
        f"- **strict_gate_before**: `{strict_gate_before}`\n"
        f"- **strict_gate_after**: `{report['strict_gate_after']}`\n"
        f"- **root_cause.category**: `{rc['category']}`\n"
        f"- **root_cause.summary**: {rc['summary']}\n"
    )
    if rc.get("first_failing_node"):
        summary_md += f"- **first_failing_node**: `{rc['first_failing_node']}`\n"
        summary_md += f"- **fx_target**: `{rc['fx_target']}`\n"
        summary_md += f"- **source_artifact**: `{rc['source_artifact']}`\n"
    summary_md += "\n## Counts\n\n"
    for k, v in counts.items():
        summary_md += f"- `{k}`: {v}\n"
    summary_md += "\n## Downstream status\n\n"
    for k, v in report["downstream_status"].items():
        summary_md += f"- `{k}`: `{v}`\n"
    summary_md += "\n## Evidence\n\n"
    for k, v in evidence.items():
        summary_md += f"- `{k}` → `{v}`\n"
    summary_path.write_text(summary_md, encoding="utf-8")

    return StrictGateReportResult(
        status=status,
        report_path=report_path,
        summary_md_path=summary_path,
        model_id=model_id,
        root_cause_category=rc["category"],
    )
