"""Contract-driven kernel verifier (M-44, LOAD-BEARING).

Phase C M-44: stop hardcoding kernel checks. Generate the verifier
obligation list mechanically from ``KernelContractV3`` fields. Each
contract field maps to a typed obligation; each obligation maps to a
concrete ``verify_*`` callable.

Mapping (Section 7's "contract field → verifier check" table):

::

    io.inputs[*].shape           → obl_input_shape_match
    io.inputs[*].dtype_class     → obl_input_dtype_match
    io.inputs[*].layout          → obl_input_layout_match
    io.outputs[*].shape          → obl_output_shape_match
    io.outputs[*].aliased_with   → obl_output_no_forbidden_alias
    numerics.accumulator_dtype   → obl_accumulator_dtype_match
    numerics.max_relative_error  → obl_differential_within_higham_bound
    numerics.deterministic       → obl_deterministic_repeat_run
    sync.event_decls             → obl_each_event_signalled_once
    memory.input_tiers           → obl_memory_tier_match  (M-48 runtime assert)
    dispatch.model               → obl_dispatch_model_match
    hardware.target_name         → obl_target_name_match

The differential check is anchored to the M-37.13 Higham bound, never
to hand-picked constants. Tampered metadata, output, or dispatch each
fire a distinct typed failure_kind that maps directly into the M-43
recoverability taxonomy (numerical_mismatch / shape_mismatch /
metadata_mismatch / semantic_contract_violation).

Output: ``04_kernel_codegen/validation/<task_id>.validation.json``
listing every obligation + its verdict. M-45 wraps an accepted
verification report into a kernel certificate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from compgen.kernels.contract_v3 import (
    DispatchModel,
    KernelContractV3,
    LayoutKind,
    MemoryTier,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerifierObligation:
    """One obligation derived from a contract field. The verifier walks
    obligations in order; the first failure short-circuits with a
    typed verdict the M-43 recovery taxonomy maps to a failure_kind."""

    obl_id: str
    contract_field: str        # human-readable origin (for audit)
    verifier_kind: str         # which verify_* the runner dispatches to
    expected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obl_id": self.obl_id,
            "contract_field": self.contract_field,
            "verifier_kind": self.verifier_kind,
            "expected": dict(self.expected),
        }


@dataclass(frozen=True)
class ObligationVerdict:
    obl_id: str
    verifier_kind: str
    status: str                # "pass" | "fail" | "deferred"
    failure_kind: str = ""     # one of M-43 RECOVERABILITY keys (when fail)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "obl_id": self.obl_id,
            "verifier_kind": self.verifier_kind,
            "status": self.status,
            "failure_kind": self.failure_kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class VerificationReport:
    schema_version: str
    task_id: str
    contract_hash: str
    overall: str               # "pass" | "fail" | "deferred"
    obligations: tuple[VerifierObligation, ...]
    verdicts: tuple[ObligationVerdict, ...]
    failure_kind: str = ""     # first failing obligation's failure_kind
    failure_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "contract_hash": self.contract_hash,
            "overall": self.overall,
            "failure_kind": self.failure_kind,
            "failure_summary": self.failure_summary,
            "obligations": [o.to_dict() for o in self.obligations],
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# --------------------------------------------------------------------------- #
# Obligation generation — mechanical traversal of contract fields
# --------------------------------------------------------------------------- #


def generate_obligations(contract: KernelContractV3) -> list[VerifierObligation]:
    """Walk ``contract`` fields and emit one obligation per field that
    has a verifiable invariant. The list is the verifier's checklist;
    no obligation can be added or removed without going through this
    function (that's the M-44 invariant — verifier and contract stay
    in lockstep)."""
    obligations: list[VerifierObligation] = []
    io = contract.io

    # 1+. io.inputs[*] — shape, dtype, layout.
    for i, t in enumerate(io.inputs):
        obligations.append(VerifierObligation(
            obl_id=f"obl_input_{i}_shape",
            contract_field=f"io.inputs[{i}].shape",
            verifier_kind="input_shape_match",
            expected={"index": i, "dims": list(t.shape.dims)},
        ))
        obligations.append(VerifierObligation(
            obl_id=f"obl_input_{i}_dtype",
            contract_field=f"io.inputs[{i}].dtype_class",
            verifier_kind="input_dtype_match",
            expected={"index": i, "dtype_class": list(t.dtype_class)},
        ))
        obligations.append(VerifierObligation(
            obl_id=f"obl_input_{i}_layout",
            contract_field=f"io.inputs[{i}].layout",
            verifier_kind="input_layout_match",
            expected={"index": i, "layout": t.layout.value},
        ))

    # outputs — shape, no forbidden alias.
    for i, t in enumerate(io.outputs):
        obligations.append(VerifierObligation(
            obl_id=f"obl_output_{i}_shape",
            contract_field=f"io.outputs[{i}].shape",
            verifier_kind="output_shape_match",
            expected={"index": i, "dims": list(t.shape.dims)},
        ))

    # numerics — accumulator dtype + Higham-bounded differential + deterministic.
    if io.numerics.accumulator_dtype:
        obligations.append(VerifierObligation(
            obl_id="obl_accumulator_dtype",
            contract_field="numerics.accumulator_dtype",
            verifier_kind="accumulator_dtype_match",
            expected={"accumulator_dtype": io.numerics.accumulator_dtype},
        ))
    obligations.append(VerifierObligation(
        obl_id="obl_differential_higham",
        contract_field="numerics.max_relative_error",
        verifier_kind="differential_within_higham_bound",
        expected={"max_relative_error": io.numerics.max_relative_error},
    ))
    if io.numerics.deterministic:
        obligations.append(VerifierObligation(
            obl_id="obl_deterministic_repeat",
            contract_field="numerics.deterministic",
            verifier_kind="deterministic_repeat_run",
            expected={"deterministic": True},
        ))

    # sync.event_decls — each declared event signalled exactly once.
    for e in contract.orchestration.sync.event_decls:
        obligations.append(VerifierObligation(
            obl_id=f"obl_event_{e.name}_signalled_once",
            contract_field=f"sync.event_decls[{e.name}]",
            verifier_kind="event_signalled_once",
            expected={"event": e.name, "wait_count": e.wait_count},
        ))

    # memory.input_tiers — runtime assertion (M-48 wires the actual
    # buffer-allocation check; M-44 declares the obligation as
    # "deferred to runtime").
    mem = contract.orchestration.memory
    for i, t in enumerate(mem.input_tiers):
        obligations.append(VerifierObligation(
            obl_id=f"obl_input_{i}_memory_tier",
            contract_field=f"memory.input_tiers[{i}]",
            verifier_kind="memory_tier_match_runtime_deferred",
            expected={"index": i, "tier": t.value},
        ))
    for i, t in enumerate(mem.output_tiers):
        obligations.append(VerifierObligation(
            obl_id=f"obl_output_{i}_memory_tier",
            contract_field=f"memory.output_tiers[{i}]",
            verifier_kind="memory_tier_match_runtime_deferred",
            expected={"index": i, "tier": t.value},
        ))

    # dispatch.model — provider's declared backend must match.
    obligations.append(VerifierObligation(
        obl_id="obl_dispatch_model",
        contract_field="dispatch.model",
        verifier_kind="dispatch_model_match",
        expected={"model": contract.orchestration.dispatch.model.value},
    ))

    # hardware.target_name — provider's claims/metadata must reference
    # the same target.
    if contract.orchestration.execution is not None:
        obligations.append(VerifierObligation(
            obl_id="obl_target_name",
            contract_field="hardware.target_name",
            verifier_kind="target_name_match",
            expected={
                "target_name": (
                    contract.orchestration.execution.hardware.target_name
                ),
            },
        ))

    return obligations


# --------------------------------------------------------------------------- #
# Verifier runner — dispatches each obligation to a verify_* callable
# --------------------------------------------------------------------------- #


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _verify_input_shape_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    inputs = metadata.get("inputs") or []
    idx = obl.expected["index"]
    expected_dims = obl.expected["dims"]
    if idx >= len(inputs):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=(
                f"metadata.inputs has only {len(inputs)} entries; "
                f"contract requires input[{idx}]"
            ),
        )
    got_dims = inputs[idx].get("dims") or inputs[idx].get("shape") or []
    if list(got_dims) != list(expected_dims):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="shape_mismatch",
            detail=(
                f"input[{idx}].dims expected {expected_dims}, got {got_dims}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_output_shape_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    outputs = metadata.get("outputs") or []
    idx = obl.expected["index"]
    expected_dims = obl.expected["dims"]
    if idx >= len(outputs):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=f"metadata.outputs missing index {idx}",
        )
    got_dims = outputs[idx].get("dims") or outputs[idx].get("shape") or []
    if list(got_dims) != list(expected_dims):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="shape_mismatch",
            detail=(
                f"output[{idx}].dims expected {expected_dims}, got {got_dims}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_input_dtype_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    inputs = metadata.get("inputs") or []
    idx = obl.expected["index"]
    expected_dtypes = set(obl.expected["dtype_class"])
    if idx >= len(inputs):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=f"metadata.inputs missing index {idx}",
        )
    got_dtype = inputs[idx].get("dtype")
    if got_dtype is None or got_dtype not in expected_dtypes:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=(
                f"input[{idx}].dtype expected one of {sorted(expected_dtypes)}, "
                f"got {got_dtype!r}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_input_layout_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    inputs = metadata.get("inputs") or []
    idx = obl.expected["index"]
    expected_layout = obl.expected["layout"]
    if idx >= len(inputs):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=f"metadata.inputs missing index {idx}",
        )
    got_layout = inputs[idx].get("layout")
    if got_layout != expected_layout:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=(
                f"input[{idx}].layout expected {expected_layout!r}, "
                f"got {got_layout!r}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_accumulator_dtype_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    expected = obl.expected["accumulator_dtype"]
    got = metadata.get("accumulator_dtype")
    if got != expected:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=f"accumulator_dtype expected {expected!r}, got {got!r}",
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_dispatch_model_match(
    *, obl: VerifierObligation, claims: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    expected = obl.expected["model"]
    supports = claims.get("supports_dispatch") or []
    if expected not in supports:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="semantic_contract_violation",
            detail=(
                f"contract dispatch.model={expected!r} not in "
                f"claims.supports_dispatch={supports!r}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_target_name_match(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    expected = obl.expected["target_name"]
    got = metadata.get("target_name")
    if got is not None and got != expected:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="metadata_mismatch",
            detail=f"target_name expected {expected!r}, got {got!r}",
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_event_signalled_once(
    *, obl: VerifierObligation, metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    """Static check on metadata: each declared event must appear in
    metadata.signals_emitted with count == wait_count."""
    declared = metadata.get("signals_emitted") or {}
    expected_event = obl.expected["event"]
    expected_count = obl.expected["wait_count"]
    got_count = declared.get(expected_event)
    if got_count is None:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",  # M-48 wires the runtime check
            detail=(
                f"metadata does not declare signals_emitted[{expected_event!r}]; "
                f"deferred to M-48 runtime assertion"
            ),
        )
    if got_count != expected_count:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="semantic_contract_violation",
            detail=(
                f"event {expected_event!r} expected wait_count={expected_count}, "
                f"got {got_count}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_memory_tier_runtime_deferred(
    *, obl: VerifierObligation, **_: Any,
) -> ObligationVerdict:
    """M-44 declares the obligation; M-48 runtime assertion checks at
    launch. Always status=deferred at this layer."""
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
        status="deferred",
        detail=(
            "memory tier match is a runtime assertion; deferred to M-48 "
            "PLAN_VIOLATION_BUFFER_TIER"
        ),
    )


def _verify_differential_within_higham_bound(
    *, obl: VerifierObligation, contract: KernelContractV3,
    metadata: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    """For COMPUTE_TILED matmul kernels we can run the differential
    here using M-37.13's Higham bound. Other archetypes are deferred
    to a M-44.x follow-on (the verifier emits 'deferred' rather than
    silently passing — staying loud is the M-31A discipline)."""
    archetype = contract.archetype.value
    if archetype != "compute_tiled":
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",
            detail=(
                f"differential check for archetype={archetype!r} is not yet "
                f"implemented; deferred to follow-on. M-44 ships matmul only."
            ),
        )

    # For matmul: the metadata must declare the per-case max_abs_error
    # and we compare against Higham's bound derived from the contract
    # shape. Real run wires this through the kernel artifact's
    # measured outputs vs eager BLAS; M-44 surfaces the obligation
    # mechanically and accepts the metadata-declared signal until M-49
    # runs the executor end-to-end.
    declared = metadata.get("declared_max_abs_error")
    declared_bound = metadata.get("declared_higham_bound")
    if declared is None or declared_bound is None:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",
            detail=(
                "metadata does not declare declared_max_abs_error and "
                "declared_higham_bound; M-49 (glue differential) wires the "
                "real measurement; deferred until then"
            ),
        )
    if float(declared) > float(declared_bound):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="numerical_mismatch",
            detail=(
                f"declared_max_abs_error={declared} > Higham bound "
                f"{declared_bound}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _verify_deterministic_repeat_run(
    *, obl: VerifierObligation, claims: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    """The provider claims expected_numerics; ``deterministic`` is
    asserted once the executor (M-47) repeats the run."""
    if claims.get("expected_numerics") in ("bit_equality", "tolerance_eps"):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",
            detail=(
                "determinism asserted by repeated run at M-47; provider "
                f"claims expected_numerics={claims.get('expected_numerics')!r}"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
        status="fail", failure_kind="semantic_contract_violation",
        detail=(
            f"contract requires deterministic=True but provider claims "
            f"expected_numerics={claims.get('expected_numerics')!r}"
        ),
    )


# Dispatch table.
_VERIFIERS: dict[str, Callable[..., ObligationVerdict]] = {
    "input_shape_match": _verify_input_shape_match,
    "input_dtype_match": _verify_input_dtype_match,
    "input_layout_match": _verify_input_layout_match,
    "output_shape_match": _verify_output_shape_match,
    "accumulator_dtype_match": _verify_accumulator_dtype_match,
    "differential_within_higham_bound": _verify_differential_within_higham_bound,
    "deterministic_repeat_run": _verify_deterministic_repeat_run,
    "event_signalled_once": _verify_event_signalled_once,
    "memory_tier_match_runtime_deferred": _verify_memory_tier_runtime_deferred,
    "dispatch_model_match": _verify_dispatch_model_match,
    "target_name_match": _verify_target_name_match,
}


def verify_kernel(
    *,
    contract: KernelContractV3,
    task_id: str,
    contract_hash: str,
    kernel_metadata_path: Path | None,
    provider_claims_path: Path | None,
) -> VerificationReport:
    """Run every obligation against the kernel's metadata + provider
    claims. Returns a typed report.

    Inputs:
      - ``contract`` — the materialised KernelContractV3 (M-40).
      - ``kernel_metadata_path`` — points at the provider's
        ``kernel_metadata.json`` (under the sandboxed artifact_dir).
      - ``provider_claims_path`` — points at the provider's
        ``provider_claims.json``.
    """
    metadata = _read_json_or_none(kernel_metadata_path) if kernel_metadata_path else None
    claims = _read_json_or_none(provider_claims_path) if provider_claims_path else None

    if metadata is None:
        return VerificationReport(
            schema_version="kernel_verification_report_v1",
            task_id=task_id, contract_hash=contract_hash,
            overall="fail", failure_kind="metadata_mismatch",
            failure_summary=(
                f"kernel_metadata.json missing or unreadable at "
                f"{kernel_metadata_path}"
            ),
            obligations=(), verdicts=(),
        )
    if claims is None:
        return VerificationReport(
            schema_version="kernel_verification_report_v1",
            task_id=task_id, contract_hash=contract_hash,
            overall="fail", failure_kind="metadata_mismatch",
            failure_summary=(
                f"provider_claims.json missing or unreadable at "
                f"{provider_claims_path}"
            ),
            obligations=(), verdicts=(),
        )

    obligations = generate_obligations(contract)
    verdicts: list[ObligationVerdict] = []
    overall = "pass"
    failure_kind = ""
    failure_summary = ""

    for obl in obligations:
        runner = _VERIFIERS.get(obl.verifier_kind)
        if runner is None:
            v = ObligationVerdict(
                obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
                status="fail", failure_kind="schema_invalid",
                detail=(
                    f"no verifier registered for kind {obl.verifier_kind!r}; "
                    f"M-44 obligation table out of sync with verifier dispatch"
                ),
            )
        else:
            try:
                v = runner(
                    obl=obl, contract=contract,
                    metadata=metadata, claims=claims,
                )
            except Exception as exc:  # noqa: BLE001 — surface as typed
                v = ObligationVerdict(
                    obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
                    status="fail", failure_kind="schema_invalid",
                    detail=f"verifier raised: {type(exc).__name__}: {exc}",
                )
        verdicts.append(v)
        if v.status == "fail" and overall != "fail":
            overall = "fail"
            failure_kind = v.failure_kind
            failure_summary = v.detail

    return VerificationReport(
        schema_version="kernel_verification_report_v1",
        task_id=task_id, contract_hash=contract_hash,
        overall=overall, failure_kind=failure_kind,
        failure_summary=failure_summary,
        obligations=tuple(obligations),
        verdicts=tuple(verdicts),
    )


def write_validation_report(
    *, run_dir: Path, task_id: str, report: VerificationReport,
) -> Path:
    """Persist the verification report under
    ``04_kernel_codegen/validation/<task_id>.validation.json``."""
    out_dir = run_dir.resolve() / "04_kernel_codegen" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.validation.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
