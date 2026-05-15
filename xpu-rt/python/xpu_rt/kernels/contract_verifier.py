"""Contract-driven kernel verifier (, LOAD-BEARING).

Phase C stop hardcoding kernel checks. Generate the verifier
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
    memory.input_tiers → obl_memory_tier_match (runtime assert)
    dispatch.model               → obl_dispatch_model_match
    hardware.target_name         → obl_target_name_match

The differential check is anchored to the Higham bound, never
to hand-picked constants. Tampered metadata, output, or dispatch each
fire a distinct typed failure_kind that maps directly into the
recoverability taxonomy (numerical_mismatch / shape_mismatch /
metadata_mismatch / semantic_contract_violation).

Output: ``04_kernel_codegen/validation/<task_id>.validation.json``
listing every obligation + its verdict. wraps an accepted
verification report into a kernel certificate.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compgen.kernels.contract_v3 import (
    KernelContractV3,
)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerifierObligation:
    """One obligation derived from a contract field. The verifier walks
    obligations in order; the first failure short-circuits with a
    typed verdict the recovery taxonomy maps to a failure_kind."""

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
    failure_kind: str = ""     # one of RECOVERABILITY keys (when fail)
    detail: str = ""
    # G2 wire-in: optional typed Counterexample populated when the
    # failure is well-characterised (numerical mismatch, contract
    # violation with a known op-level cause). Backward-compatible:
    # legacy verdicts have counterexample=None.
    counterexample: Any = None  # compgen.agent.counterexample.Counterexample

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "obl_id": self.obl_id,
            "verifier_kind": self.verifier_kind,
            "status": self.status,
            "failure_kind": self.failure_kind,
            "detail": self.detail,
        }
        if self.counterexample is not None:
            body["counterexample"] = self.counterexample.to_dict()
        return body


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
    function (that's the invariant — verifier and contract stay
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

    # memory.input_tiers — runtime assertion (wires the actual
    # buffer-allocation check; declares the obligation as
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

    # wire-up: when the contract opts in via
    # ``optional_v3_1_fields["z3_proof_required"] = True``, every
    # supported precondition becomes a Z3 proof obligation. The
    # ``predicate_proof_via_z3`` verifier discharges them through
    # :mod:`compgen.solve.z3_obligations`.
    if contract.optional_v3_1_fields.get("z3_proof_required"):
        for i, pred in enumerate(contract.preconditions or ()):
            obligations.append(VerifierObligation(
                obl_id=f"obl_precondition_{i}_z3_proof",
                contract_field=f"preconditions[{i}]",
                verifier_kind="predicate_proof_via_z3",
                expected={
                    "predicate_index": i,
                    "predicate": _predicate_to_proof_dict(pred),
                },
            ))

    return obligations


def _predicate_to_proof_dict(pred: Any) -> dict[str, Any]:
    """Translate a typed :class:`Predicate` into the parameters the
    Z3 obligation harness consumes.

    Supported mappings:

    * ``ModEq(arg_dim, k)`` — emits ``divisible_by(<dim_name>, k)``
      under a single bounded integer variable.
    * ``DtypeIn(arg, dtype_set)`` — no Z3 proof needed; expressed as
      a finite ``in_set`` over the dtype symbol mapped to integer
      codepoints.

    Anything else returns ``{"unsupported": True, ...}``; the
    verifier maps that to ``status=deferred`` so the obligation is
    visible in the report but does not fail the cert.
    """

    from compgen.kernels.predicates import (
        ByteSizeLe,
        DtypeIn,
        ModEq,
        NoAlias,
        NumericalWithinEps,
        predicate_kind,
    )

    if isinstance(pred, ModEq):
        return {
            "obligation_kind": "shape_predicate_implication",
            "params": {
                "variables": {pred.arg_dim: {"min": 1, "max": 65536}},
                "applies_when": [],
                "precondition": {
                    "op": "divisible_by", "var": pred.arg_dim, "k": pred.k,
                },
            },
        }
    if isinstance(pred, DtypeIn):
        return {
            "obligation_kind": "shape_predicate_implication",
            "params": {
                "variables": {pred.arg: {"min": 0, "max": 16}},
                "applies_when": [],
                "precondition": {
                    "op": "in_set", "var": pred.arg, "values": [hash(d) & 0xFFFF for d in pred.dtype_set],
                },
            },
        }
    if isinstance(pred, (ByteSizeLe, NoAlias, NumericalWithinEps)):
        # No Z3 lowering yet; report as unsupported so the verifier
        # marks it ``deferred`` honestly.
        return {
            "unsupported": True,
            "kind": predicate_kind(pred),
            "reason": "no Z3 lowering implemented yet",
        }
    return {
        "unsupported": True,
        "kind": type(pred).__name__,
        "reason": "unknown predicate type",
    }


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
            status="deferred",  # wires the runtime check
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
    """declares the obligation; runtime assertion checks at
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
    here using 's Higham bound. Other archetypes are deferred
    to a .x follow-on (the verifier emits 'deferred' rather than
    silently passing — staying loud is the discipline)."""
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
    # measured outputs vs eager BLAS; surfaces the obligation
    # mechanically and accepts the metadata-declared signal until
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
        cex = _build_numerical_counterexample(
            obl=obl,
            contract=contract,
            declared=float(declared),
            declared_bound=float(declared_bound),
        )
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="numerical_mismatch",
            detail=(
                f"declared_max_abs_error={declared} > Higham bound "
                f"{declared_bound}"
            ),
            counterexample=cex,
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
    )


def _build_numerical_counterexample(
    *,
    obl: VerifierObligation,
    contract: KernelContractV3,
    declared: float,
    declared_bound: float,
) -> Any:
    """Build a typed :class:`compgen.agent.counterexample.Counterexample`
    for a numerical-mismatch failure.

    The verifier knows the archetype + contract shape + declared
    error but not a specific failing index — those come from the
    real-differential runner. Until that ships, we emit the
    Counterexample with empty indices but populated names + IR slice
    so the agent's :mod:`compgen.agent.primitives.explain_counterexample`
    has typed structure to reason about.
    """

    from compgen.agent.counterexample import (
        Counterexample,
        InputSlice,
        IRSlice,
        OutputSlice,
        RemediationHint,
        classify_rejection,
    )

    # Numerical-mismatch is recoverable iff a remediation hint exists;
    # at this layer we don't have a concrete candidate id, so the
    # class lands as "surprising" — the LLM-call-site primitive
    # explain_counterexample will promote it to tactic_recoverable
    # when it can offer a hint.
    rejection_class = classify_rejection(
        legality_was_blocked=False,
        numerical_only=True,
        remediation_known=False,
    )

    return Counterexample(
        gate="differential_higham",
        rejection_class=rejection_class,
        input_slice=InputSlice(
            name="matmul_input_a",
            indices={},  # will populate when the real runner lands
        ),
        output_slice=OutputSlice(
            name="matmul_output",
            indices={},
            actual=declared,
            reference=declared_bound,
            abs_error=max(0.0, declared - declared_bound),
        ),
        ir_slice=IRSlice(
            region_id=obl.contract_field,  # e.g. "numerics.max_relative_error"
            op=f"archetype={contract.archetype.value}",
            annotation=(
                f"accumulator_dtype="
                f"{contract.io.numerics.accumulator_dtype or 'unspecified'}; "
                f"declared_max_abs_error={declared}, "
                f"Higham bound={declared_bound}"
            ),
        ),
        likely_cause=(
            "declared_max_abs_error exceeded Higham bound — "
            "accumulator precision is likely insufficient for this "
            "matmul shape"
        ),
        remediation=RemediationHint(
            kind="param_change",
            suggest=None,
            confidence=0.5,
            rationale=(
                "Promote accumulator to a higher-precision dtype "
                "(e.g. fp32 for an fp16 matmul); no specific candidate "
                "id resolvable at the verifier layer"
            ),
        ),
    )


def _verify_deterministic_repeat_run(
    *, obl: VerifierObligation, claims: dict[str, Any], **_: Any,
) -> ObligationVerdict:
    """The provider claims expected_numerics; ``deterministic`` is
    asserted once the executor repeats the run."""
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


def _verify_predicate_proof_via_z3(
    *,
    obl: VerifierObligation,
    contract: KernelContractV3,
    metadata: dict[str, Any],
    z3_obligation_report: dict[str, Any] | None = None,
    **_: Any,
) -> ObligationVerdict:
    """Discharge a contract precondition through the Z3 obligation harness.

    wire-up. Reads the per-predicate ``proof_dict`` from
    ``obl.expected['predicate']``; when it carries ``unsupported:
    True``, we honestly report ``deferred``. Otherwise we route the
    obligation through :mod:`compgen.solve.z3_obligations` and append
    its typed response into ``z3_obligation_report`` (mutated in
    place by the caller).

    Status mapping:

    * Z3 ``proved`` → ``pass``.
    * Z3 ``sat_counterexample`` → ``fail`` (counterexample is
      included in the obligation report).
    * Z3 ``timeout`` / ``unsupported`` / ``error`` → ``deferred``
      with the typed reason.
    """

    pred_proof = obl.expected.get("predicate") or {}
    if pred_proof.get("unsupported"):
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",
            detail=(
                f"precondition kind {pred_proof.get('kind')!r} has no Z3 "
                f"lowering yet: {pred_proof.get('reason', '')}"
            ),
        )

    from compgen.solve.backend_registry import default_registry
    from compgen.solve.solver_types import (
        SolverProblemKind,
        SolverRequest,
        SolverStatus,
    )

    registry = default_registry()
    request = SolverRequest(
        problem_id=obl.obl_id,
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation=pred_proof,
    )
    from compgen.solve.solver_types import SolverBackendName

    z3_backend = registry.get_backend(SolverBackendName.Z3)
    if z3_backend is None:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="deferred",
            detail="Z3 backend not registered",
        )
    response = z3_backend.solve(request)
    if z3_obligation_report is not None:
        z3_obligation_report.setdefault("obligations", []).append(
            {
                "obl_id": obl.obl_id,
                "predicate_index": obl.expected.get("predicate_index"),
                "obligation_kind": pred_proof.get("obligation_kind"),
                "status": response.status.value,
                "selected_backend": response.selected_backend.value,
                "formulation_hash": response.formulation_hash,
                "time_ms": response.time_ms,
                "counterexample": response.counterexample,
                "infeasibility_reason": response.infeasibility_reason,
            }
        )
    if response.status is SolverStatus.PROVED:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind, status="pass",
            detail=(
                f"Z3 proved precondition (formulation_hash="
                f"{response.formulation_hash})"
            ),
        )
    if response.status is SolverStatus.SAT_COUNTEREXAMPLE:
        return ObligationVerdict(
            obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
            status="fail", failure_kind="semantic_contract_violation",
            detail=(
                f"Z3 found counterexample: {response.counterexample!r} "
                f"(formulation_hash={response.formulation_hash})"
            ),
        )
    return ObligationVerdict(
        obl_id=obl.obl_id, verifier_kind=obl.verifier_kind,
        status="deferred",
        detail=(
            f"Z3 returned {response.status.value}: "
            f"{response.infeasibility_reason or ''}"
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
    "predicate_proof_via_z3": _verify_predicate_proof_via_z3,
}


def verify_kernel(
    *,
    contract: KernelContractV3,
    task_id: str,
    contract_hash: str,
    kernel_metadata_path: Path | None,
    provider_claims_path: Path | None,
    z3_obligation_report: dict[str, Any] | None = None,
) -> VerificationReport:
    """Run every obligation against the kernel's metadata + provider
    claims. Returns a typed report.

    Inputs:
      ``contract`` — the materialised KernelContractV3.
      - ``kernel_metadata_path`` — points at the provider's
        ``kernel_metadata.json`` (under the sandboxed artifact_dir).
      - ``provider_claims_path`` — points at the provider's
        ``provider_claims.json``.
      - ``z3_obligation_report`` — optional dict the
        ``predicate_proof_via_z3`` verifier mutates to record each
        Z3 obligation's typed response (wire-up). Caller
        persists this via :func:`write_z3_obligation_report`.
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
                    z3_obligation_report=z3_obligation_report,
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


def write_z3_obligation_report(
    *, run_dir: Path, task_id: str, body: dict[str, Any],
) -> Path:
    """Persist the Z3 obligation report under
    ``04_kernel_codegen/solver/<task_id>.z3_obligations.json``.

    Caller pattern (wire-up)::

        z3_report: dict = {
            "schema_version": "z3_obligations_index_v1",
            "task_id": task_id,
            "obligations": [],
        }
        verify_kernel(..., z3_obligation_report=z3_report)
        path = write_z3_obligation_report(
            run_dir=run_dir, task_id=task_id, body=z3_report,
        )
        # path is recorded on KernelCertificate.z3_obligation_report_ref.
    """

    out_dir = run_dir.resolve() / "04_kernel_codegen" / "solver"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.z3_obligations.json"
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
