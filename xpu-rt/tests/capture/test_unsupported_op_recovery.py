"""Comprehensive tests for the unsupported-op recovery pipeline.

Exercises every stage: detect, introspect, classify, synthesize
(translation, decomp, fake), verify, promote, and the full end-to-end
``recover_unsupported_operators()`` orchestrator.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from compgen.capture.unsupported import recover_unsupported_operators
from compgen.capture.unsupported.classify import UnsupportedClassification, classify_operator_issue
from compgen.capture.unsupported.detect import UnsupportedOperatorIssue, detect_unsupported_operators
from compgen.capture.unsupported.introspect import (
    ExampleTensorInfo,
    UnsupportedOpDossier,
    build_operator_dossier,
    parse_target,
)
from compgen.capture.unsupported.promote import PromotionRecord, build_promotion_record
from compgen.capture.unsupported.synthesize_decomp import (
    SynthesizedDecomposition,
    synthesize_export_decomposition,
)
from compgen.capture.unsupported.synthesize_fake import (
    SynthesizedFakeKernel,
    synthesize_fake_kernel,
)
from compgen.capture.unsupported.synthesize_translation import (
    SynthesizedPayloadTranslation,
    synthesize_payload_translation,
)
from compgen.capture.unsupported.verify import UnsupportedVerification, verify_unsupported_resolution

# ---------------------------------------------------------------------------
# Helpers -- mock FX graph structures
# ---------------------------------------------------------------------------


def _make_fake_tensor(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a meta tensor with the given shape and dtype."""
    return torch.empty(shape, dtype=dtype, device="meta")


def _make_fx_node(
    target: str,
    name: str,
    input_shapes: list[tuple[int, ...]],
    output_shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    """Create a minimal FX-like node with meta values."""
    args = []
    for shape in input_shapes:
        arg = SimpleNamespace(meta={"val": _make_fake_tensor(shape, dtype)})
        args.append(arg)

    return SimpleNamespace(
        op="call_function",
        target=target,
        name=name,
        args=tuple(args),
        meta={"val": _make_fake_tensor(output_shape, dtype)},
    )


def _make_exported_program(nodes: list[SimpleNamespace]) -> SimpleNamespace:
    """Wrap nodes in a minimal exported-program-like object."""
    return SimpleNamespace(graph=SimpleNamespace(nodes=nodes))


# ---------------------------------------------------------------------------
# 1. detect_unsupported_operators
# ---------------------------------------------------------------------------


class TestDetectUnsupportedOperators:
    """Tests for the detection stage."""

    def test_no_unsupported_ops_when_all_supported(self) -> None:
        """If all targets are in supported_targets, no issues are found."""
        node = _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        issues = detect_unsupported_operators(program, supported_targets={"aten.mm.default"})
        assert issues == []

    def test_detects_unsupported_op(self) -> None:
        """A node not in supported_targets should be detected."""
        node = _make_fx_node("aten.addmm.default", "addmm_0", [(16,), (4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        issues = detect_unsupported_operators(program, supported_targets=set())
        assert len(issues) == 1
        assert issues[0].target == "aten.addmm.default"
        assert issues[0].count == 1
        assert "addmm_0" in issues[0].node_names

    def test_groups_multiple_occurrences(self) -> None:
        """Multiple nodes with the same target should be grouped."""
        nodes = [
            _make_fx_node("aten.silu.default", "silu_0", [(4, 8)], (4, 8)),
            _make_fx_node("aten.silu.default", "silu_1", [(4, 8)], (4, 8)),
        ]
        program = _make_exported_program(nodes)

        issues = detect_unsupported_operators(program, supported_targets=set())
        assert len(issues) == 1
        assert issues[0].count == 2
        assert "silu_0" in issues[0].node_names
        assert "silu_1" in issues[0].node_names

    def test_explicit_targets_excluded(self) -> None:
        """Explicit targets should be excluded even if not in supported_targets."""
        node = _make_fx_node("custom.my_op.default", "my_op_0", [(4, 8)], (4, 8))
        program = _make_exported_program([node])

        issues = detect_unsupported_operators(
            program, supported_targets=set(), explicit_targets={"custom.my_op.default"}
        )
        assert issues == []

    def test_non_call_function_nodes_skipped(self) -> None:
        """Nodes with op != 'call_function' should be skipped."""
        node = SimpleNamespace(op="placeholder", target="x", name="x", args=(), meta={})
        program = _make_exported_program([node])

        issues = detect_unsupported_operators(program, supported_targets=set())
        assert issues == []

    def test_example_inputs_captured(self) -> None:
        """Detection should capture example tensor info from node args."""
        node = _make_fx_node("aten.addmm.default", "addmm_0", [(16,), (4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        issues = detect_unsupported_operators(program, supported_targets=set())
        assert len(issues) == 1
        assert len(issues[0].example_inputs) == 3
        assert issues[0].example_inputs[0].shape == (16,)
        assert issues[0].example_output is not None
        assert issues[0].example_output.shape == (4, 16)


# ---------------------------------------------------------------------------
# 2. build_operator_dossier / parse_target
# ---------------------------------------------------------------------------


class TestIntrospect:
    """Tests for operator introspection."""

    def test_parse_target_three_parts(self) -> None:
        """Three-part target should parse correctly."""
        ns, op, overload = parse_target("aten.addmm.default")
        assert ns == "aten"
        assert op == "addmm"
        assert overload == "default"

    def test_parse_target_two_parts(self) -> None:
        """Two-part target should default overload to 'default'."""
        ns, op, overload = parse_target("aten.mm")
        assert ns == "aten"
        assert op == "mm"
        assert overload == "default"

    def test_parse_target_one_part(self) -> None:
        """Single-part target should have empty namespace."""
        ns, op, overload = parse_target("unknown")
        assert ns == ""
        assert op == "unknown"
        assert overload == "default"

    def test_build_dossier_aten_op(self) -> None:
        """Dossier for an ATen op should populate schema fields."""
        dossier = build_operator_dossier("aten.mm.default")
        assert dossier.target == "aten.mm.default"
        assert dossier.namespace == "aten"
        assert dossier.operator == "mm"
        assert dossier.overload == "default"
        assert dossier.is_aten is True
        assert dossier.is_custom is False
        # torch.ops.aten.mm.default should resolve
        assert dossier.schema != ""

    def test_build_dossier_custom_op(self) -> None:
        """Dossier for a custom op should flag is_custom."""
        dossier = build_operator_dossier("custom_ns.my_op.default")
        assert dossier.is_custom is True
        assert dossier.is_aten is False

    def test_build_dossier_with_sample_args(self) -> None:
        """Dossier should incorporate sample arg tensor info."""
        sample_input = ExampleTensorInfo(shape=(4, 8), dtype="float32")
        sample_output = ExampleTensorInfo(shape=(4, 16), dtype="float32")
        dossier = build_operator_dossier(
            "aten.mm.default",
            sample_args=(sample_input,),
            sample_output=sample_output,
        )
        assert len(dossier.example_inputs) == 1
        assert dossier.example_inputs[0].shape == (4, 8)
        assert dossier.example_output is not None
        assert dossier.example_output.shape == (4, 16)


# ---------------------------------------------------------------------------
# 3. classify_operator_issue
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for operator classification."""

    def test_classify_known_payload_decomposition(self) -> None:
        """Op with registered payload decomposition -> known_payload_decomposition."""
        issue = UnsupportedOperatorIssue(
            target="aten.mm.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = build_operator_dossier("aten.mm.default", payload_decomposition_registered=True)
        classification = classify_operator_issue(issue, dossier)
        assert classification.strategy == "known_payload_decomposition"
        assert classification.bucket == "payload_decomposition"
        assert classification.confidence == "high"

    def test_classify_torchao_like(self) -> None:
        """TorchAO-like ops -> explicit_blackbox."""
        issue = UnsupportedOperatorIssue(
            target="torchao.quant_op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="torchao.quant_op.default",
            namespace="torchao",
            operator="quant_op",
            overload="default",
            schema="",
            is_torchao_like=True,
            is_custom=True,
        )
        classification = classify_operator_issue(issue, dossier)
        assert classification.strategy == "explicit_blackbox"
        assert classification.bucket == "quantization_wrapper"

    def test_classify_custom_op(self) -> None:
        """Custom namespace ops -> explicit_blackbox."""
        issue = UnsupportedOperatorIssue(
            target="custom_ns.my_op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="custom_ns.my_op.default",
            namespace="custom_ns",
            operator="my_op",
            overload="default",
            schema="",
            is_custom=True,
        )
        classification = classify_operator_issue(issue, dossier)
        assert classification.strategy == "explicit_blackbox"
        assert classification.bucket == "opaque_custom_op"

    def test_classify_simple_aten_tensor_op(self) -> None:
        """Simple ATen tensor op -> synthesized_external_call."""
        issue = UnsupportedOperatorIssue(
            target="aten.mm.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = build_operator_dossier("aten.mm.default")
        # The schema for mm should contain "-> Tensor" and have <=3 Tensor mentions
        classification = classify_operator_issue(issue, dossier)
        assert classification.strategy == "synthesized_external_call"
        assert classification.bucket == "payload_decomposition"

    def test_classify_fallback_blackbox(self) -> None:
        """Unrecognized op without simple schema -> explicit_blackbox."""
        issue = UnsupportedOperatorIssue(
            target="aten.complex_op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="aten.complex_op.default",
            namespace="aten",
            operator="complex_op",
            overload="default",
            schema="(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e) -> Tensor",
            is_aten=True,
        )
        classification = classify_operator_issue(issue, dossier)
        assert classification.strategy == "explicit_blackbox"
        assert classification.confidence == "low"


# ---------------------------------------------------------------------------
# 4. synthesize_payload_translation
# ---------------------------------------------------------------------------


class TestSynthesizePayloadTranslation:
    """Tests for payload translation synthesis."""

    def test_synthesize_for_external_call_strategy(self) -> None:
        """Should produce a translation for synthesized_external_call strategy."""
        issue = UnsupportedOperatorIssue(
            target="aten.mm.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = build_operator_dossier("aten.mm.default")
        classification = UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="medium",
            reason="test",
        )
        translation = synthesize_payload_translation(issue, dossier, classification)
        assert translation is not None
        assert isinstance(translation, SynthesizedPayloadTranslation)
        assert translation.kind == "external_call"
        assert "mm" in translation.callee_name

    def test_returns_none_for_blackbox_strategy(self) -> None:
        """Should return None when strategy is not synthesized_external_call."""
        issue = UnsupportedOperatorIssue(
            target="custom.op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="custom.op.default",
            namespace="custom",
            operator="op",
            overload="default",
            schema="",
            is_custom=True,
        )
        classification = UnsupportedClassification(
            bucket="opaque_custom_op",
            strategy="explicit_blackbox",
            confidence="medium",
            reason="test",
        )
        translation = synthesize_payload_translation(issue, dossier, classification)
        assert translation is None


# ---------------------------------------------------------------------------
# 5. synthesize_export_decomposition
# ---------------------------------------------------------------------------


class TestSynthesizeExportDecomposition:
    """Tests for export-level decomposition synthesis."""

    def test_returns_decomp_for_known_aten_addmm(self) -> None:
        """Should return a callable decomposition for aten.addmm.default."""
        dossier = build_operator_dossier("aten.addmm.default")
        result = synthesize_export_decomposition("aten.addmm.default", dossier)
        assert result is not None
        assert isinstance(result, SynthesizedDecomposition)
        assert result.target == "aten.addmm.default"
        assert result.description != ""
        # The decomp_fn should be callable and produce correct output
        bias = torch.randn(16)
        mat1 = torch.randn(4, 8)
        mat2 = torch.randn(8, 16)
        out = result.decomp_fn(bias, mat1, mat2)
        expected = torch.mm(mat1, mat2) + bias
        assert torch.allclose(out, expected)

    def test_returns_decomp_for_known_aten_silu(self) -> None:
        """Should return a callable decomposition for aten.silu.default."""
        dossier = build_operator_dossier("aten.silu.default")
        result = synthesize_export_decomposition("aten.silu.default", dossier)
        assert result is not None
        x = torch.randn(4, 8)
        out = result.decomp_fn(x)
        expected = x * torch.sigmoid(x)
        assert torch.allclose(out, expected)

    def test_returns_decomp_for_known_aten_linear(self) -> None:
        """Should return a callable decomposition for aten.linear.default."""
        dossier = build_operator_dossier("aten.linear.default")
        result = synthesize_export_decomposition("aten.linear.default", dossier)
        assert result is not None
        x = torch.randn(4, 8)
        w = torch.randn(16, 8)
        b = torch.randn(16)
        out = result.decomp_fn(x, w, b)
        expected = torch.mm(x, w.t()) + b
        assert torch.allclose(out, expected, atol=1e-6)

    def test_returns_decomp_for_leaky_relu(self) -> None:
        """Should return a callable decomposition for aten.leaky_relu.default."""
        dossier = build_operator_dossier("aten.leaky_relu.default")
        result = synthesize_export_decomposition("aten.leaky_relu.default", dossier)
        assert result is not None
        x = torch.randn(4, 8)
        out = result.decomp_fn(x)
        expected = torch.where(x >= 0, x, x * 0.01)
        assert torch.allclose(out, expected)

    def test_returns_decomp_for_hardswish(self) -> None:
        """Should return a callable decomposition for aten.hardswish.default."""
        dossier = build_operator_dossier("aten.hardswish.default")
        result = synthesize_export_decomposition("aten.hardswish.default", dossier)
        assert result is not None
        x = torch.randn(4, 8)
        out = result.decomp_fn(x)
        expected = x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0
        assert torch.allclose(out, expected)

    def test_returns_none_for_unknown_aten_op(self) -> None:
        """Should return None for an ATen op not on the allow-list."""
        dossier = build_operator_dossier("aten.some_exotic_op.default")
        result = synthesize_export_decomposition("aten.some_exotic_op.default", dossier)
        assert result is None

    def test_returns_none_for_non_aten_op(self) -> None:
        """Should return None for a non-ATen namespace op."""
        dossier = UnsupportedOpDossier(
            target="custom.my_op.default",
            namespace="custom",
            operator="my_op",
            overload="default",
            schema="",
            is_custom=True,
            is_aten=False,
        )
        result = synthesize_export_decomposition("custom.my_op.default", dossier)
        assert result is None


# ---------------------------------------------------------------------------
# 6. synthesize_fake_kernel
# ---------------------------------------------------------------------------


class TestSynthesizeFakeKernel:
    """Tests for fake kernel synthesis."""

    def test_returns_fake_for_valid_dossier(self) -> None:
        """Should return a fake kernel when the dossier has example output."""
        dossier = UnsupportedOpDossier(
            target="custom.my_op.default",
            namespace="custom",
            operator="my_op",
            overload="default",
            schema="",
            is_custom=True,
            example_output=ExampleTensorInfo(shape=(4, 16), dtype="float32"),
        )
        result = synthesize_fake_kernel("custom.my_op.default", dossier)
        assert result is not None
        assert isinstance(result, SynthesizedFakeKernel)
        assert result.output_shape == (4, 16)
        assert result.output_dtype == "float32"

    def test_fake_fn_produces_correct_shape_and_dtype(self) -> None:
        """The fake function should return a tensor with the correct shape/dtype."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(2, 3, 4), dtype="float16"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is not None
        out = result.fake_fn(torch.randn(2, 3))
        assert out.shape == (2, 3, 4)
        assert out.dtype == torch.float16

    def test_fake_fn_inherits_device_from_input(self) -> None:
        """The fake function should use the device of the first tensor arg."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(4, 8), dtype="float32"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is not None
        cpu_input = torch.randn(4, 8, device="cpu")
        out = result.fake_fn(cpu_input)
        assert out.device.type == "cpu"

    def test_returns_none_when_no_example_output(self) -> None:
        """Should return None when dossier has no example output."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=None,
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is None

    def test_returns_none_when_empty_shape(self) -> None:
        """Should return None when example output has empty shape."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(), dtype="float32"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is None

    def test_returns_none_when_zero_dimension(self) -> None:
        """Should return None when example output has a zero-sized dimension."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(4, 0, 8), dtype="float32"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is None

    def test_handles_int_dtype(self) -> None:
        """Fake kernel should handle integer dtypes correctly."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(4,), dtype="int64"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is not None
        out = result.fake_fn()
        assert out.shape == (4,)
        assert out.dtype == torch.int64

    def test_handles_bool_dtype(self) -> None:
        """Fake kernel should handle bool dtype correctly."""
        dossier = UnsupportedOpDossier(
            target="aten.my_op.default",
            namespace="aten",
            operator="my_op",
            overload="default",
            schema="",
            is_aten=True,
            example_output=ExampleTensorInfo(shape=(2, 3), dtype="bool"),
        )
        result = synthesize_fake_kernel("aten.my_op.default", dossier)
        assert result is not None
        out = result.fake_fn()
        assert out.shape == (2, 3)
        assert out.dtype == torch.bool


# ---------------------------------------------------------------------------
# 7. verify_unsupported_resolution
# ---------------------------------------------------------------------------


class TestVerifyUnsupportedResolution:
    """Tests for verification of recovery artifacts."""

    def test_verify_with_known_aten_op(self) -> None:
        """Verification should succeed for a known ATen op with valid examples."""
        issue = UnsupportedOperatorIssue(
            target="aten.mm.default",
            stage="payload_import",
            reason="test",
            count=1,
            example_inputs=(
                ExampleTensorInfo(shape=(4, 8), dtype="float32"),
                ExampleTensorInfo(shape=(8, 16), dtype="float32"),
            ),
            example_output=ExampleTensorInfo(shape=(4, 16), dtype="float32"),
        )
        dossier = build_operator_dossier(
            "aten.mm.default",
            sample_args=(
                ExampleTensorInfo(shape=(4, 8), dtype="float32"),
                ExampleTensorInfo(shape=(8, 16), dtype="float32"),
            ),
            sample_output=ExampleTensorInfo(shape=(4, 16), dtype="float32"),
        )
        verification = verify_unsupported_resolution(issue, dossier, translation=None)
        assert verification.schema_ok is True
        assert verification.eager_reference_ok is True

    def test_verify_reports_missing_schema(self) -> None:
        """Verification should flag missing schema."""
        issue = UnsupportedOperatorIssue(
            target="unknown.op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="unknown.op.default",
            namespace="unknown",
            operator="op",
            overload="default",
            schema="",
        )
        verification = verify_unsupported_resolution(issue, dossier, translation=None)
        assert verification.schema_ok is False
        assert "missing operator schema" in verification.messages

    def test_verify_reports_missing_reference(self) -> None:
        """Verification should flag missing reference callable."""
        issue = UnsupportedOperatorIssue(
            target="unknown.op.default",
            stage="payload_import",
            reason="test",
            count=1,
        )
        dossier = UnsupportedOpDossier(
            target="unknown.op.default",
            namespace="unknown",
            operator="op",
            overload="default",
            schema="(Tensor a) -> Tensor",
            reference_callable=None,
        )
        verification = verify_unsupported_resolution(issue, dossier, translation=None)
        assert verification.eager_reference_ok is False
        assert any("missing eager reference" in m for m in verification.messages)

    def test_verify_notes_synthesized_translation(self) -> None:
        """Verification should note that a translation was synthesized."""
        issue = UnsupportedOperatorIssue(
            target="aten.mm.default",
            stage="payload_import",
            reason="test",
            count=1,
            example_inputs=(
                ExampleTensorInfo(shape=(4, 8), dtype="float32"),
                ExampleTensorInfo(shape=(8, 16), dtype="float32"),
            ),
        )
        dossier = build_operator_dossier(
            "aten.mm.default",
            sample_args=(
                ExampleTensorInfo(shape=(4, 8), dtype="float32"),
                ExampleTensorInfo(shape=(8, 16), dtype="float32"),
            ),
        )
        translation = SynthesizedPayloadTranslation(
            target="aten.mm.default",
            kind="external_call",
            translator=lambda *_: None,  # type: ignore[arg-type]
            callee_name="aten_mm_default",
        )
        verification = verify_unsupported_resolution(issue, dossier, translation=translation)
        assert any("synthesized" in m for m in verification.messages)


# ---------------------------------------------------------------------------
# 8. build_promotion_record
# ---------------------------------------------------------------------------


class TestBuildPromotionRecord:
    """Tests for promotion record creation."""

    def test_builds_stable_cache_key(self) -> None:
        """Cache key should be deterministic for the same inputs."""
        dossier = build_operator_dossier("aten.mm.default")
        classification = UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="medium",
            reason="test",
        )
        versions = {"torch": "2.4.0"}
        record1 = build_promotion_record(dossier, classification, versions)
        record2 = build_promotion_record(dossier, classification, versions)
        assert record1.cache_key == record2.cache_key
        assert isinstance(record1, PromotionRecord)
        assert record1.policy == "cache-first"

    def test_different_versions_produce_different_keys(self) -> None:
        """Different runtime versions should produce different cache keys."""
        dossier = build_operator_dossier("aten.mm.default")
        classification = UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="medium",
            reason="test",
        )
        record1 = build_promotion_record(dossier, classification, {"torch": "2.4.0"})
        record2 = build_promotion_record(dossier, classification, {"torch": "2.5.0"})
        assert record1.cache_key != record2.cache_key


# ---------------------------------------------------------------------------
# 9. Full end-to-end: recover_unsupported_operators
# ---------------------------------------------------------------------------


class TestRecoverUnsupportedOperatorsE2E:
    """End-to-end tests for the full recovery pipeline."""

    def test_full_pipeline_with_single_unsupported_op(self) -> None:
        """Full pipeline should detect, classify, verify, and promote for a single op."""
        node = _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        resolutions = recover_unsupported_operators(
            program,
            supported_targets=set(),
            runtime_versions={"torch": "2.4.0"},
        )
        assert len(resolutions) == 1

        r = resolutions[0]
        assert r.target == "aten.mm.default"
        assert r.issue.count == 1
        assert r.dossier.is_aten is True
        assert r.classification.strategy in {"synthesized_external_call", "known_payload_decomposition"}
        assert isinstance(r.verification, UnsupportedVerification)
        assert isinstance(r.promotion, PromotionRecord)

    def test_full_pipeline_with_multiple_ops(self) -> None:
        """Full pipeline should handle multiple different unsupported ops."""
        nodes = [
            _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16)),
            _make_fx_node("custom_ns.my_op.default", "my_op_0", [(4, 8)], (4, 8)),
        ]
        program = _make_exported_program(nodes)

        resolutions = recover_unsupported_operators(
            program,
            supported_targets=set(),
            runtime_versions={"torch": "2.4.0"},
        )
        assert len(resolutions) == 2

        targets = {r.target for r in resolutions}
        assert "aten.mm.default" in targets
        assert "custom_ns.my_op.default" in targets

        # The custom op should be classified as blackbox
        custom_res = next(r for r in resolutions if r.target == "custom_ns.my_op.default")
        assert custom_res.classification.strategy == "explicit_blackbox"
        assert custom_res.approved_blackbox is True

    def test_full_pipeline_no_unsupported_ops(self) -> None:
        """Full pipeline should return empty list when all ops are supported."""
        node = _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        resolutions = recover_unsupported_operators(
            program,
            supported_targets={"aten.mm.default"},
            runtime_versions={"torch": "2.4.0"},
        )
        assert resolutions == []

    def test_full_pipeline_with_payload_decomposition(self) -> None:
        """Ops already registered in supported_targets as payload decompositions."""
        node = _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        # mm is in supported_targets, so it should not appear as unsupported
        resolutions = recover_unsupported_operators(
            program,
            supported_targets={"aten.mm.default"},
            runtime_versions={"torch": "2.4.0"},
        )
        assert len(resolutions) == 0

    def test_resolution_dataclass_fields(self) -> None:
        """UnsupportedOpResolution should expose all fields."""
        node = _make_fx_node("aten.mm.default", "mm_0", [(4, 8), (8, 16)], (4, 16))
        program = _make_exported_program([node])

        resolutions = recover_unsupported_operators(
            program,
            supported_targets=set(),
            runtime_versions={"torch": "2.4.0"},
        )
        assert len(resolutions) == 1
        r = resolutions[0]
        # All fields should be present and of the right type
        assert isinstance(r.issue, UnsupportedOperatorIssue)
        assert isinstance(r.dossier, UnsupportedOpDossier)
        assert isinstance(r.classification, UnsupportedClassification)
        assert isinstance(r.verification, UnsupportedVerification)
        assert isinstance(r.promotion, PromotionRecord)
        assert isinstance(r.approved_blackbox, bool)


# ---------------------------------------------------------------------------
# 10. Import smoke test
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """Verify that public API can be imported without errors."""

    def test_import_synthesize_decomp(self) -> None:
        """synthesize_decomp module should be importable."""
        from compgen.capture.unsupported.synthesize_decomp import (
            SynthesizedDecomposition,
            synthesize_export_decomposition,
        )

        assert callable(synthesize_export_decomposition)
        assert SynthesizedDecomposition is not None

    def test_import_synthesize_fake(self) -> None:
        """synthesize_fake module should be importable."""
        from compgen.capture.unsupported.synthesize_fake import (
            SynthesizedFakeKernel,
            synthesize_fake_kernel,
        )

        assert callable(synthesize_fake_kernel)
        assert SynthesizedFakeKernel is not None

    def test_top_level_recover_import(self) -> None:
        """Top-level recover_unsupported_operators should be importable."""
        from compgen.capture.unsupported import recover_unsupported_operators

        assert callable(recover_unsupported_operators)
