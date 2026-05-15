"""Tests for the G2 wire-in: typed Counterexample on contract-verifier
numerical-mismatch failure.

Coverage:

* A failing differential-higham obligation produces a verdict whose
  `counterexample` is a typed :class:`compgen.agent.counterexample.Counterexample`.
* The verdict's `to_dict()` serializes the counterexample so downstream
  evidence packs can consume the typed payload.
* A passing obligation leaves `counterexample=None` (backward-compat).
* The `_build_numerical_counterexample` helper preserves Higham bound
  + declared error in the OutputSlice.
* Deferred verdicts (e.g. metadata missing, non-compute_tiled archetype)
  emit no counterexample.
* The legacy `ObligationVerdict.to_dict()` shape is preserved when
  counterexample=None.

These tests use a lightweight SimpleNamespace stub for the contract —
the verifier only reads `contract.archetype.value` and
`contract.io.numerics.accumulator_dtype`, so constructing a full
KernelContractV3 envelope is unnecessary overhead for testing the
G2 wire-in surface.
"""

from __future__ import annotations

from types import SimpleNamespace

from compgen.agent.counterexample import (
    Counterexample,
    IRSlice,
    RemediationHint,
)
from compgen.kernels.contract_verifier import (
    ObligationVerdict,
    VerifierObligation,
    _build_numerical_counterexample,
    _verify_differential_within_higham_bound,
)


def _stub_contract(
    archetype_value: str = "compute_tiled",
    accumulator_dtype: str | None = "fp16",
) -> SimpleNamespace:
    return SimpleNamespace(
        archetype=SimpleNamespace(value=archetype_value),
        io=SimpleNamespace(
            numerics=SimpleNamespace(accumulator_dtype=accumulator_dtype),
        ),
    )


def _obl() -> VerifierObligation:
    return VerifierObligation(
        obl_id="obl_differential_higham",
        contract_field="numerics.max_relative_error",
        verifier_kind="differential_within_higham_bound",
        expected={"max_relative_error": 1e-3},
    )


def test_failing_differential_emits_typed_counterexample():
    contract = _stub_contract()
    metadata = {
        "declared_max_abs_error": 0.02,
        "declared_higham_bound": 0.01,
    }
    verdict = _verify_differential_within_higham_bound(
        obl=_obl(), contract=contract, metadata=metadata,
    )
    assert verdict.status == "fail"
    assert verdict.failure_kind == "numerical_mismatch"
    assert isinstance(verdict.counterexample, Counterexample)
    cex = verdict.counterexample
    assert cex.gate == "differential_higham"
    assert cex.rejection_class == "surprising"
    assert cex.output_slice.actual == 0.02
    assert cex.output_slice.reference == 0.01
    assert cex.output_slice.abs_error == 0.01
    assert isinstance(cex.ir_slice, IRSlice)
    assert "accumulator_dtype" in cex.ir_slice.annotation
    assert "fp16" in cex.ir_slice.annotation
    assert isinstance(cex.remediation, RemediationHint)
    assert cex.remediation.kind == "param_change"


def test_passing_verdict_has_no_counterexample():
    contract = _stub_contract()
    metadata = {
        "declared_max_abs_error": 0.005,
        "declared_higham_bound": 0.01,
    }
    verdict = _verify_differential_within_higham_bound(
        obl=_obl(), contract=contract, metadata=metadata,
    )
    assert verdict.status == "pass"
    assert verdict.counterexample is None
    body = verdict.to_dict()
    assert "counterexample" not in body


def test_failing_verdict_to_dict_serializes_counterexample():
    contract = _stub_contract()
    metadata = {"declared_max_abs_error": 1.0, "declared_higham_bound": 0.1}
    verdict = _verify_differential_within_higham_bound(
        obl=_obl(), contract=contract, metadata=metadata,
    )
    body = verdict.to_dict()
    assert body["status"] == "fail"
    assert "counterexample" in body
    cex_body = body["counterexample"]
    assert cex_body["gate"] == "differential_higham"
    assert cex_body["rejection_class"] in {
        "tactic_fatal", "tactic_recoverable", "surprising",
    }
    assert "output_slice" in cex_body
    assert cex_body["output_slice"]["abs_error"] >= 0.0


def test_build_numerical_counterexample_helper_directly():
    contract = _stub_contract(accumulator_dtype="fp32")
    cex = _build_numerical_counterexample(
        obl=_obl(), contract=contract,
        declared=0.5, declared_bound=0.1,
    )
    assert isinstance(cex, Counterexample)
    assert cex.output_slice.abs_error == 0.4
    assert "fp32" in cex.ir_slice.annotation


def test_deferred_path_when_metadata_missing():
    contract = _stub_contract()
    verdict = _verify_differential_within_higham_bound(
        obl=_obl(), contract=contract, metadata={},
    )
    assert verdict.status == "deferred"
    assert verdict.counterexample is None


def test_deferred_path_when_archetype_not_compute_tiled():
    contract = _stub_contract(archetype_value="memory_bound")
    verdict = _verify_differential_within_higham_bound(
        obl=_obl(), contract=contract,
        metadata={"declared_max_abs_error": 0.5, "declared_higham_bound": 0.1},
    )
    assert verdict.status == "deferred"
    assert verdict.counterexample is None


def test_obligation_verdict_to_dict_omits_counterexample_when_none():
    v = ObligationVerdict(
        obl_id="obl_legacy",
        verifier_kind="input_shape_match",
        status="pass",
    )
    body = v.to_dict()
    assert set(body.keys()) == {
        "obl_id", "verifier_kind", "status", "failure_kind", "detail",
    }


def test_obligation_verdict_with_counterexample_serializes():
    contract = _stub_contract()
    cex = _build_numerical_counterexample(
        obl=_obl(), contract=contract,
        declared=0.5, declared_bound=0.1,
    )
    v = ObligationVerdict(
        obl_id="obl_x",
        verifier_kind="differential_within_higham_bound",
        status="fail",
        failure_kind="numerical_mismatch",
        detail="x",
        counterexample=cex,
    )
    body = v.to_dict()
    assert "counterexample" in body
    assert body["counterexample"]["gate"] == "differential_higham"


def test_abs_error_is_nonneg_when_reference_above_actual():
    """Defensive: if the bound is somehow above the declared value
    (shouldn't happen on a fail path, but check the helper)."""

    contract = _stub_contract()
    cex = _build_numerical_counterexample(
        obl=_obl(), contract=contract,
        declared=0.1, declared_bound=0.5,
    )
    # max(0.0, 0.1 - 0.5) = 0.0
    assert cex.output_slice.abs_error == 0.0
