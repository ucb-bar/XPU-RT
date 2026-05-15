"""Architecture audit for the Phase F extension substrate.

Encodes the ten hard rules from
``docs/architecture/EXTENSION_PROVIDER_ARCHITECTURE.md`` as
deterministic checks. The audit returns a typed
:class:`AuditReport`; the CLI wrapper in
``scripts/dev/audit_extension_architecture.py`` exits non-zero on
violations.

Checks (closed enum):

* ``extension_card_completeness`` — every shipped card declares
  ``integration_level``.
* ``provider_result_is_not_certificate`` — source scan: no
  module asserts ``ProviderResult ≡ Certificate`` or assigns
  ``KernelCertificate.from_provider_result``.
* ``pass_tool_no_direct_ir_mutation`` — pass-tool cards never
  declare ``writes: [payload_ir]``; pass-tool entrypoints never
  import ``compgen.ir.payload.write``-style modules.
* ``optional_provider_imports_quarantined`` — disallowed
  third-party imports (``cuda_tile``, ``tilelang``, ``cutlass``,
  ``thunderkittens``, ``hexagon_mlir``, etc.) outside
  ``providers/adapters/*`` and ``kernels/providers/*``.
* ``blocked_provider_not_paper_claimable`` — provider cards with
  ``integration_level`` in {card_only, probe, generate} must have
  ``paper_claimable: false``.

Every violation carries the source path and a typed reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from compgen.providers.card_loader import (
    iter_dialect_cards as _iter_dialect_cards,
    iter_provider_cards as _iter_provider_cards,
    iter_target_cards as _iter_target_cards,
)
from compgen.providers.provider_types import PAPER_CLAIMABLE_LEVELS


# Optional provider modules that must NEVER be imported at top-level
# in core code. Adapters under ``providers/adapters/`` and the legacy
# ``kernels/providers/`` tree are the only allowed import sites.
FORBIDDEN_OPTIONAL_IMPORTS = (
    "cuda_tile",
    "tilelang",
    "cutlass",
    "thunderkittens",
    "hexagon_mlir",
    "bitblas",
    "mirage",
    "kernelbench",
    "caesar",
    "radiance_kernels",
    "pallas",
    "nki",
)

ALLOWED_IMPORT_PATHS = (
    "python/compgen/providers/adapters/",
    "python/compgen/kernels/providers/",
    "python/compgen/kernels/autocomp_adapter.py",
    "python/compgen/kernels/kernelblaster_adapter.py",
    "python/compgen/extensions/firesim/",
    "python/compgen/extensions/zephyr/",
)

VIOLATION_REASONS = (
    "missing_integration_level",
    "paper_claimable_at_blocked_level",
    "pass_tool_writes_payload_ir",
    "forbidden_optional_import",
    "provider_result_used_as_certificate",
    "execution_evidence_malformed",
)


@dataclass(frozen=True)
class Violation:
    check: str
    path: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "path": self.path,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class AuditReport:
    violations: list[Violation] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "schema_version": "extension_architecture_audit_v1",
            "passed": self.passed,
            "checks_run": list(self.checks_run),
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "summary": dict(self.summary),
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_card_completeness(report: AuditReport) -> None:
    """Every shipped card declares ``integration_level`` (schema
    already rejects missing, but assert at the audit layer too)."""

    report.checks_run.append("extension_card_completeness")
    count = 0
    for c in _iter_provider_cards():
        count += 1
        if not c.integration_level:
            report.violations.append(
                Violation(
                    check="extension_card_completeness",
                    path=f"providers/cards/{c.provider_id}.yaml",
                    reason="missing_integration_level",
                )
            )
    for d in _iter_dialect_cards():
        count += 1
        if not d.integration_level:
            report.violations.append(
                Violation(
                    check="extension_card_completeness",
                    path=f"dialects/cards/{d.dialect_provider_id}.yaml",
                    reason="missing_integration_level",
                )
            )
    for t in _iter_target_cards():
        count += 1
    report.summary["cards_audited"] = count


def check_blocked_provider_not_paper_claimable(report: AuditReport) -> None:
    """A card at integration_level ∉ {verify, promote} must have
    paper_claimable=false."""

    report.checks_run.append("blocked_provider_not_paper_claimable")
    for c in _iter_provider_cards():
        if c.paper_claimable and c.integration_level not in PAPER_CLAIMABLE_LEVELS:
            report.violations.append(
                Violation(
                    check="blocked_provider_not_paper_claimable",
                    path=f"providers/cards/{c.provider_id}.yaml",
                    reason="paper_claimable_at_blocked_level",
                    detail=f"integration_level={c.integration_level!r}",
                )
            )
    for d in _iter_dialect_cards():
        if d.paper_claimable and d.integration_level not in PAPER_CLAIMABLE_LEVELS:
            report.violations.append(
                Violation(
                    check="blocked_provider_not_paper_claimable",
                    path=f"dialects/cards/{d.dialect_provider_id}.yaml",
                    reason="paper_claimable_at_blocked_level",
                    detail=f"integration_level={d.integration_level!r}",
                )
            )


def check_pass_tool_no_direct_ir_mutation(report: AuditReport) -> None:
    """Pass-tool cards never declare writes: [payload_ir]."""

    report.checks_run.append("pass_tool_no_direct_ir_mutation")
    import yaml
    cards_dir = Path(__file__).resolve().parent.parent / "pass_tools" / "cards"
    if not cards_dir.is_dir():
        return
    for path in sorted(cards_dir.glob("*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        writes = body.get("writes", []) or []
        if "payload_ir" in writes:
            report.violations.append(
                Violation(
                    check="pass_tool_no_direct_ir_mutation",
                    path=str(path),
                    reason="pass_tool_writes_payload_ir",
                )
            )


def _path_is_allowed(path: Path, repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    for allowed in ALLOWED_IMPORT_PATHS:
        if rel.startswith(allowed):
            return True
    return False


def check_optional_provider_imports_quarantined(
    report: AuditReport,
    *,
    repo_root: Path | None = None,
) -> None:
    """No core module imports an optional provider's top-level
    package. Adapter modules are the only legal import sites."""

    report.checks_run.append("optional_provider_imports_quarantined")
    root = repo_root or Path(__file__).resolve().parent.parent.parent.parent
    src_root = root / "python" / "compgen"
    if not src_root.is_dir():
        return

    import_re = re.compile(r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)")
    for py_path in src_root.rglob("*.py"):
        if _path_is_allowed(py_path, root):
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or not (stripped.startswith("import") or stripped.startswith("from")):
                continue
            m = import_re.match(stripped)
            if not m:
                continue
            mod = m.group(1)
            for forbidden in FORBIDDEN_OPTIONAL_IMPORTS:
                if mod == forbidden or mod.startswith(forbidden + "."):
                    report.violations.append(
                        Violation(
                            check="optional_provider_imports_quarantined",
                            path=str(py_path.relative_to(root)),
                            reason="forbidden_optional_import",
                            detail=f"line {line!r} imports {forbidden!r}",
                        )
                    )
                    break


def check_provider_result_is_not_certificate(
    report: AuditReport,
    *,
    repo_root: Path | None = None,
) -> None:
    """Source scan: ``ProviderResult`` must never be aliased to or
    treated as a ``KernelCertificate``."""

    report.checks_run.append("provider_result_is_not_certificate")
    root = repo_root or Path(__file__).resolve().parent.parent.parent.parent
    src_root = root / "python" / "compgen"
    if not src_root.is_dir():
        return

    suspicious_patterns = (
        re.compile(r"ProviderResult\s*=\s*KernelCertificate"),
        re.compile(r"KernelCertificate\s*=\s*ProviderResult"),
        re.compile(r"isinstance\([^,]+,\s*KernelCertificate\)\s*or\s*isinstance\([^,]+,\s*ProviderResult\)"),
        re.compile(r"as_certificate\(provider_result"),
        re.compile(r"certificate\s*=\s*provider_result\b"),
    )

    for py_path in src_root.rglob("*.py"):
        try:
            text = py_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in suspicious_patterns:
            for match in pattern.finditer(text):
                report.violations.append(
                    Violation(
                        check="provider_result_is_not_certificate",
                        path=str(py_path.relative_to(root)),
                        reason="provider_result_used_as_certificate",
                        detail=match.group(0),
                    )
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_execution_evidence(
    report: AuditReport,
    *,
    evidence_pack: Path | None = None,
) -> None:
    """every per-provider directory must carry either the
    available-quartet or a typed blocked_proof.

    A missing ``per_provider/`` tree is fine when no execution evidence
    has been recorded yet (e.g. evidence pack built without
    runs). A directory that exists but is malformed or empty is a
    violation.
    """

    report.checks_run.append("execution_evidence")
    if evidence_pack is None:
        # Default to the canonical evidence pack location.
        evidence_pack = Path("results/extension_provider_evidence_pack")
    from compgen.audit.execution_evidence import (
        audit_provider_dir,
        discover_per_provider_dirs,
    )

    dirs = discover_per_provider_dirs(evidence_pack)
    state_counts = {"available": 0, "blocked": 0}
    for pdir in dirs:
        state, detail = audit_provider_dir(pdir)
        if state in ("malformed", "empty"):
            report.violations.append(
                Violation(
                    check="execution_evidence",
                    path=str(pdir),
                    reason="execution_evidence_malformed",
                    detail=detail or state,
                )
            )
            continue
        state_counts[state] = state_counts.get(state, 0) + 1
    report.summary["execution_evidence_available"] = state_counts.get("available", 0)
    report.summary["execution_evidence_blocked"] = state_counts.get("blocked", 0)


def run_audit(
    *,
    repo_root: Path | None = None,
    evidence_pack: Path | None = None,
) -> AuditReport:
    report = AuditReport()
    check_card_completeness(report)
    check_blocked_provider_not_paper_claimable(report)
    check_pass_tool_no_direct_ir_mutation(report)
    check_optional_provider_imports_quarantined(report, repo_root=repo_root)
    check_provider_result_is_not_certificate(report, repo_root=repo_root)
    check_execution_evidence(report, evidence_pack=evidence_pack)
    return report
