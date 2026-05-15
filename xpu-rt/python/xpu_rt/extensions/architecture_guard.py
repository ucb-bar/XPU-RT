"""spec'd path for the architecture audit.

Re-exports :mod:`compgen.audit.extension_architecture` so user
spec imports of ``compgen.extensions.architecture_guard`` resolve.
"""

from __future__ import annotations

from compgen.audit.extension_architecture import (
    ALLOWED_IMPORT_PATHS,
    FORBIDDEN_OPTIONAL_IMPORTS,
    VIOLATION_REASONS,
    AuditReport,
    Violation,
    check_blocked_provider_not_paper_claimable,
    check_card_completeness,
    check_execution_evidence,
    check_optional_provider_imports_quarantined,
    check_pass_tool_no_direct_ir_mutation,
    check_provider_result_is_not_certificate,
    run_audit,
)

__all__ = [
    "ALLOWED_IMPORT_PATHS",
    "FORBIDDEN_OPTIONAL_IMPORTS",
    "VIOLATION_REASONS",
    "AuditReport",
    "Violation",
    "check_blocked_provider_not_paper_claimable",
    "check_card_completeness",
    "check_execution_evidence",
    "check_optional_provider_imports_quarantined",
    "check_pass_tool_no_direct_ir_mutation",
    "check_provider_result_is_not_certificate",
    "run_audit",
]
