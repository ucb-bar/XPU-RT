"""Tests for the P2.3 typed Counterexample + delta_debug module."""

from __future__ import annotations

import pytest
from compgen.agent.counterexample import (
    REJECTION_CLASSES,
    REMEDIATION_KINDS,
    Counterexample,
    CounterexampleError,
    InputSlice,
    IRSlice,
    OutputSlice,
    RemediationHint,
    classify_rejection,
    delta_debug_input,
)


def _cex(**overrides):
    return Counterexample(
        gate=overrides.get("gate", "differential"),
        rejection_class=overrides.get("rejection_class", "tactic_recoverable"),
        input_slice=InputSlice(
            name="hidden_states",
            indices={"batch": 3, "seq": 12, "head": 2},
        ),
        output_slice=OutputSlice(
            name="attn_logits",
            indices={"batch": 3, "seq": 12, "query": 2, "key": 4},
            actual=0.51,
            reference=0.43,
            abs_error=0.08,
        ),
        ir_slice=IRSlice(
            region_id="023",
            op="linalg.matmul %2, %3",
            annotation="fp16 accumulator, FuseElementwise(softmax,matmul)",
        ),
        likely_cause=overrides.get("likely_cause", "softmax overflow when fused before matmul"),
        remediation=overrides.get(
            "remediation",
            RemediationHint(
                kind="tactic_change",
                suggest="cand_reorder_fuse",
                confidence=0.7,
                rationale="reorder so matmul comes first",
            ),
        ),
    )


# ---------- Positive --------------------------------------------------


def test_counterexample_roundtrips():
    body = _cex().to_dict()
    assert body["gate"] == "differential"
    assert body["rejection_class"] == "tactic_recoverable"
    assert body["output_slice"]["abs_error"] == 0.08
    assert body["remediation"]["kind"] == "tactic_change"


def test_remediation_none_allowed():
    cex = _cex(remediation=None)
    assert cex.to_dict()["remediation"] is None


def test_classify_rejection_tactic_fatal():
    assert classify_rejection(
        legality_was_blocked=True,
        numerical_only=False,
        remediation_known=False,
    ) == "tactic_fatal"


def test_classify_rejection_tactic_recoverable():
    assert classify_rejection(
        legality_was_blocked=False,
        numerical_only=True,
        remediation_known=True,
    ) == "tactic_recoverable"


def test_classify_rejection_surprising():
    assert classify_rejection(
        legality_was_blocked=False,
        numerical_only=False,
        remediation_known=False,
    ) == "surprising"
    # Numerical but no remediation → surprising
    assert classify_rejection(
        legality_was_blocked=False,
        numerical_only=True,
        remediation_known=False,
    ) == "surprising"


def test_remediation_kinds_closed_enum():
    assert set(REMEDIATION_KINDS) == {"tactic_change", "param_change", "abandon_tactic"}


def test_rejection_classes_closed_enum():
    assert set(REJECTION_CLASSES) == {"tactic_fatal", "tactic_recoverable", "surprising"}


# ---------- delta_debug -----------------------------------------------


def test_delta_debug_reduces_to_single_element():
    """Predicate fires only when the input contains a specific marker."""

    full = list(range(32))
    target = 17

    def predicate(xs):
        return target in xs

    minimised = delta_debug_input(full, failing_predicate=predicate)
    assert target in minimised
    assert len(minimised) < len(full)


def test_delta_debug_already_minimal():
    """Single-element failing input — nothing to halve."""

    out = delta_debug_input([42], failing_predicate=lambda xs: 42 in xs)
    assert out == [42]


def test_delta_debug_rejects_non_failing_input():
    """If the seed doesn't fail, there is nothing to minimise."""

    with pytest.raises(CounterexampleError, match="does not fail"):
        delta_debug_input([1, 2, 3], failing_predicate=lambda xs: 99 in xs)


def test_delta_debug_respects_iteration_cap():
    """A pathological predicate (every nonempty slice fails) terminates."""

    out = delta_debug_input(
        list(range(64)),
        failing_predicate=lambda xs: len(xs) > 0,
        max_iterations=8,
    )
    # Must terminate; halving 64 down by 8 iters leaves at least 1.
    assert 1 <= len(out) <= 64


# ---------- Negative controls ----------------------------------------


def test_unknown_rejection_class_rejected():
    with pytest.raises(CounterexampleError, match="rejection_class"):
        Counterexample(
            gate="differential",
            rejection_class="not_a_real_class",
            input_slice=InputSlice(name="x", indices={}),
            output_slice=OutputSlice(name="y", indices={}, actual=0.0, reference=0.0, abs_error=0.0),
            ir_slice=IRSlice(region_id="r", op="op"),
        )


def test_unknown_remediation_kind_rejected():
    with pytest.raises(CounterexampleError, match="remediation kind"):
        RemediationHint(kind="bogus_kind", suggest=None, confidence=0.5)


def test_remediation_confidence_out_of_range_rejected():
    with pytest.raises(CounterexampleError, match="confidence"):
        RemediationHint(kind="tactic_change", suggest=None, confidence=1.5)
    with pytest.raises(CounterexampleError, match="confidence"):
        RemediationHint(kind="tactic_change", suggest=None, confidence=-0.1)
