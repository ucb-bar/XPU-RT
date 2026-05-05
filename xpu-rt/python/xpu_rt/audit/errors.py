"""Typed audit errors.

Every audit gate that can fail raises a specific subclass of
:class:`AuditError`. Generic ``RuntimeError`` is reserved for genuine
programmer bugs; an audit failure is never a programmer bug — it is the
gate doing its job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class AuditError(RuntimeError):
    """Base class for every audit failure.

    Catching ``AuditError`` in the trust-report aggregator is fine.
    Catching it anywhere else is a smell: the gate exists to fire.
    """


# ---------------------------------------------------------------------------
# Contracts / caveats / ledger schema
# ---------------------------------------------------------------------------


class RealnessContractError(AuditError):
    """Realness contract is malformed, missing required fields, or stale."""


class CaveatLedgerError(AuditError):
    """Caveat ledger entry is malformed or violates the schema."""


class StaleCaveatError(CaveatLedgerError):
    """A caveat row is older than its required-verification window."""


# ---------------------------------------------------------------------------
# Realness scan / import provenance
# ---------------------------------------------------------------------------


class UnallowlistedStubError(AuditError):
    """Source-level scan found a stub/mock/placeholder not in the allowlist."""


class ForbiddenImportError(AuditError):
    """A production run imported a module that is forbidden on production paths."""


# ---------------------------------------------------------------------------
# Trace replay
# ---------------------------------------------------------------------------


class ReplayHashMismatch(AuditError):
    """Replaying a decision trace produced different artifact hashes than the original."""


class DecisionIdMismatch(AuditError):
    """Agent response decision_id does not match the request's decision_id."""


# ---------------------------------------------------------------------------
# Fresh-agent harness / holdout / perturbation
# ---------------------------------------------------------------------------


class TaskPackIncomplete(AuditError):
    """Generated fresh-agent task pack is missing a required allowlisted file."""


class TaskPackContaminated(AuditError):
    """Generated task pack includes a forbidden file (private context leak)."""


class HoldoutHonestyViolation(AuditError):
    """A holdout model run silently partial-passed instead of verified-or-typed-blocked."""


# ---------------------------------------------------------------------------
# Negative-control placeholders (full implementations land in M-31)
# ---------------------------------------------------------------------------


class MissingPassCard(AuditError):
    """An exposed pass has no pass card (placeholder until M-31)."""


class PreconditionViolation(AuditError):
    """A pass attempted to run on IR that fails its preconditions."""


class StaleAnalysisAudit(AuditError):
    """A consumer used an analysis summary that should have been invalidated."""


class ContractHashMismatch(AuditError):
    """A promoted recipe's contract_hash does not match the caller's region."""


class AppliesWhenViolation(AuditError):
    """A promoted recipe's applies_when predicate failed for the candidate region."""


class CertificateInvalidated(AuditError):
    """A verification certificate was reused across mismatched artifact hashes."""


class EvidencePackSourceMissing(AuditError):
    """Evidence pack builder ran but a declared source artifact was absent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single trust-report gate.

    ``status`` is ``"pass"``, ``"fail"``, or ``"skipped"``. ``skipped`` is
    only valid when the gate's prerequisites are absent (e.g. a hardware-
    dependent gate skipped on a CPU-only host); never as a polite failure.
    """

    name: str
    status: str
    detail: str = ""
    artifact_path: Path | None = None
