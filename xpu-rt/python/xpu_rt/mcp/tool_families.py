"""H4 — Tool families with shared base schemas (Section 11 Dream 4).

Six closed-enum tool families:

* ``dossier_read`` — pure-read inspection of graph dossiers
* ``decision`` — propose / apply / override / list decisions
* ``recipe_edit`` — Recipe-IR mutations
* ``kernel_request`` — kernel codegen / bench / autotune
* ``bench`` — measurement registration + lookup
* ``verification`` — verify_*, etc_conformance_*, explain_verification

Each family declares a ``BaseInput`` + ``BaseOutput`` (frozen dataclass
with ``from_dict`` + ``to_dict``); concrete tool inputs / outputs are
expected to be supersets of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

# ---------------------------------------------------------------------------
# Family enum
# ---------------------------------------------------------------------------

FAMILY_DOSSIER_READ: Final[str] = "dossier_read"
FAMILY_DECISION: Final[str] = "decision"
FAMILY_RECIPE_EDIT: Final[str] = "recipe_edit"
FAMILY_KERNEL_REQUEST: Final[str] = "kernel_request"
FAMILY_BENCH: Final[str] = "bench"
FAMILY_VERIFICATION: Final[str] = "verification"

TOOL_FAMILIES: Final[tuple[str, ...]] = (
    FAMILY_DOSSIER_READ,
    FAMILY_DECISION,
    FAMILY_RECIPE_EDIT,
    FAMILY_KERNEL_REQUEST,
    FAMILY_BENCH,
    FAMILY_VERIFICATION,
)


# ---------------------------------------------------------------------------
# Base I/O dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseInput:
    """Common input fields for every family."""

    session_id: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseInput":
        return cls(session_id=d.get("session_id", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id}


@dataclass(frozen=True)
class BaseOutput:
    """Common output fields for every family."""

    ok: bool = True
    status: str = "ok"
    error: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseOutput":
        return cls(
            ok=bool(d.get("ok", True)),
            status=str(d.get("status", "ok")),
            error=str(d.get("error", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "error": self.error}


# ---------------------------------------------------------------------------
# Per-family specialisations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DossierReadInput(BaseInput):
    """Inputs common to graph/dossier-read tools."""

    focus_region_id: str | None = None


@dataclass(frozen=True)
class DossierReadOutput(BaseOutput):
    """Outputs common to graph/dossier-read tools."""

    region_count: int = 0


@dataclass(frozen=True)
class DecisionInput(BaseInput):
    """Inputs common to decision-family tools."""

    decision_key: str = ""


@dataclass(frozen=True)
class DecisionOutput(BaseOutput):
    """Outputs common to decision-family tools."""

    decision_key: str = ""
    resolution: str = ""


@dataclass(frozen=True)
class RecipeEditInput(BaseInput):
    """Inputs common to recipe-edit tools."""

    target: str = ""
    edit_kind: str = ""


@dataclass(frozen=True)
class RecipeEditOutput(BaseOutput):
    """Outputs common to recipe-edit tools."""

    op_id: str = ""
    applied: bool = False


@dataclass(frozen=True)
class KernelRequestInput(BaseInput):
    """Inputs common to kernel-request tools."""

    op_signature: str = ""


@dataclass(frozen=True)
class KernelRequestOutput(BaseOutput):
    """Outputs common to kernel-request tools."""

    request_id: str = ""
    kernel_id: str = ""


@dataclass(frozen=True)
class BenchInput(BaseInput):
    """Inputs common to bench tools."""

    bench_id: str = ""


@dataclass(frozen=True)
class BenchOutput(BaseOutput):
    """Outputs common to bench tools."""

    bench_id: str = ""
    latency_us: float = 0.0


@dataclass(frozen=True)
class VerificationInput(BaseInput):
    """Inputs common to verification tools."""

    verifier: str = ""


@dataclass(frozen=True)
class VerificationOutput(BaseOutput):
    """Outputs common to verification tools."""

    verdict: str = ""
    counterexample: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Family registry helpers
# ---------------------------------------------------------------------------

FAMILY_BASES: Final[dict[str, tuple[type, type]]] = {
    FAMILY_DOSSIER_READ: (DossierReadInput, DossierReadOutput),
    FAMILY_DECISION: (DecisionInput, DecisionOutput),
    FAMILY_RECIPE_EDIT: (RecipeEditInput, RecipeEditOutput),
    FAMILY_KERNEL_REQUEST: (KernelRequestInput, KernelRequestOutput),
    FAMILY_BENCH: (BenchInput, BenchOutput),
    FAMILY_VERIFICATION: (VerificationInput, VerificationOutput),
}


# Default per-tool family mapping. Read-only catalog; tools not listed
# here inherit no family (audit reports them).
DEFAULT_TOOL_FAMILY: Final[dict[str, str]] = {
    # dossier_read
    "analyze_graph": FAMILY_DOSSIER_READ,
    "get_dossier": FAMILY_DOSSIER_READ,
    "focus_chunk": FAMILY_DOSSIER_READ,
    "get_context_brief": FAMILY_DOSSIER_READ,
    "view_recipe": FAMILY_DOSSIER_READ,
    "diff_recipe": FAMILY_DOSSIER_READ,
    "list_phase_tools": FAMILY_DOSSIER_READ,
    "session_summary": FAMILY_DOSSIER_READ,
    # decision
    "propose_decision": FAMILY_DECISION,
    "apply_decision": FAMILY_DECISION,
    "override_decision": FAMILY_DECISION,
    "list_decisions": FAMILY_DECISION,
    "list_pending_dispatch_decisions": FAMILY_DECISION,
    "request_dispatch_decision": FAMILY_DECISION,
    "register_dispatch_decision": FAMILY_DECISION,
    "lookup_dispatch_decision": FAMILY_DECISION,
    # recipe_edit
    "apply_recipe": FAMILY_RECIPE_EDIT,
    "step_proposal": FAMILY_RECIPE_EDIT,
    "batch_propose": FAMILY_RECIPE_EDIT,
    "propose_invent_slot": FAMILY_RECIPE_EDIT,
    "suggest_proposals": FAMILY_RECIPE_EDIT,
    "synthesize_decomp": FAMILY_RECIPE_EDIT,
    "synthesize_translation": FAMILY_RECIPE_EDIT,
    "resolve_unsupported_op": FAMILY_RECIPE_EDIT,
    # kernel_request
    "request_kernel_codegen": FAMILY_KERNEL_REQUEST,
    "register_kernel_result": FAMILY_KERNEL_REQUEST,
    "lookup_cached_kernel": FAMILY_KERNEL_REQUEST,
    "request_refinement": FAMILY_KERNEL_REQUEST,
    "register_refinement_attempt": FAMILY_KERNEL_REQUEST,
    "request_autotune_trial": FAMILY_KERNEL_REQUEST,
    "register_autotune_pick": FAMILY_KERNEL_REQUEST,
    # bench
    "request_kernel_bench": FAMILY_BENCH,
    "register_bench_result": FAMILY_BENCH,
    "lookup_bench_result": FAMILY_BENCH,
    # verification
    "verify_proposal": FAMILY_VERIFICATION,
    "verify_vendor_package": FAMILY_VERIFICATION,
    "explain_verification": FAMILY_VERIFICATION,
    "etc_conformance_run": FAMILY_VERIFICATION,
    "etc_conformance_summarize": FAMILY_VERIFICATION,
    "etc_megakernel_inspect": FAMILY_VERIFICATION,
}


def is_known_family(family: str) -> bool:
    """True iff ``family`` is one of the six closed-enum names."""

    return family in TOOL_FAMILIES


def family_for_tool(tool_name: str) -> str | None:
    """Look up the default family for ``tool_name`` (None if unmapped)."""

    return DEFAULT_TOOL_FAMILY.get(tool_name)


__all__ = [
    "BaseInput",
    "BaseOutput",
    "BenchInput",
    "BenchOutput",
    "DEFAULT_TOOL_FAMILY",
    "DecisionInput",
    "DecisionOutput",
    "DossierReadInput",
    "DossierReadOutput",
    "FAMILY_BASES",
    "FAMILY_BENCH",
    "FAMILY_DECISION",
    "FAMILY_DOSSIER_READ",
    "FAMILY_KERNEL_REQUEST",
    "FAMILY_RECIPE_EDIT",
    "FAMILY_VERIFICATION",
    "KernelRequestInput",
    "KernelRequestOutput",
    "RecipeEditInput",
    "RecipeEditOutput",
    "TOOL_FAMILIES",
    "VerificationInput",
    "VerificationOutput",
    "family_for_tool",
    "is_known_family",
]
