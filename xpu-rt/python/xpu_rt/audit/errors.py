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
# Negative-control placeholders (full implementations land )
# ---------------------------------------------------------------------------


class MissingPassCard(AuditError):
    """An exposed pass has no pass card (placeholder until )."""


class PreconditionViolation(AuditError):
    """A pass attempted to run on IR that fails its preconditions."""


class StaleAnalysisAudit(AuditError):
    """A consumer used an analysis summary that should have been invalidated.

    introduced this as a typed-error placeholder. turns it
    into the family root for stale-summary failures — both
    producer-side ("a pass mutated something it didn't declare";
    :class:`UnannouncedInvalidation`) and consumer-side ("a reader
    consumed a summary an upstream pass invalidated"; lands )
    inherit from this class so existing references and
    pytest.raises(StaleAnalysisAudit) calls keep working."""


class UnannouncedInvalidation(StaleAnalysisAudit):
    """A pass mutated an analysis summary it did not declare in its
    ``invalidates`` list. The dependency closure of the claim
    was checked too — this fires when even the transitive closure
    misses the observed mutation."""


class RefinementMonotonicityViolation(AuditError):
    """A recipe claims a refinement (e.g. ``bit_equality``) stronger
    than the weakest refinement preserved across its applied passes
    ."""


class VerificationGateMissing(AuditError):
    """A pass card declares a required verification rung (e.g.
    ``differential``) but no certificate is on disk."""


class VerificationGateFailed(AuditError):
    """A verification certificate exists but reports
    ``status != "pass"``."""


class PhaseTransitionViolation(AuditError):
    """A pass plan attempted to run a phase-N pass before all
    phase-(<N) passes had completed.

    The phase order is strict:
    ``canonicalize → analyze → optimize → verify → emit``. An
    optimize-phase pass cannot be scheduled before canonicalize
    finishes; a verify-phase pass cannot be scheduled while there
    are still optimize-phase passes pending."""


class PairContractViolation(AuditError):
    """A pass plan violated a card's ``requires_after`` or ``excludes``
    contract. A required-after pass was missing from the plan,
    or two mutually-exclusive passes were scheduled together."""


class PassPlanInvalid(AuditError):
    """An agent's ``pass_plan`` field violated a structural invariant —
    duplicate steps, references to unknown pass_ids, references to
    illegal candidate_ids, etc.. Distinct from
    :class:`PhaseTransitionViolation` and :class:`PairContractViolation`
    so the validator can attribute failures precisely."""


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
