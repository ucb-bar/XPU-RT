"""Bucket classification for unsupported operators."""

from __future__ import annotations

from dataclasses import dataclass

from compgen.capture.unsupported.detect import UnsupportedOperatorIssue
from compgen.capture.unsupported.introspect import UnsupportedOpDossier


@dataclass(frozen=True)
class UnsupportedClassification:
    """Classifier output for one unsupported operator."""

    bucket: str
    strategy: str
    confidence: str
    reason: str


def _schema_is_simple_tensor_op(schema: str) -> bool:
    return "-> Tensor" in schema and schema.count("Tensor") <= 3


def classify_operator_issue(
    issue: UnsupportedOperatorIssue,
    dossier: UnsupportedOpDossier,
) -> UnsupportedClassification:
    """Classify an unsupported operator into a bounded recovery bucket."""

    if dossier.payload_decomposition_registered:
        return UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="known_payload_decomposition",
            confidence="high",
            reason="Payload lowering is already registered",
        )

    if dossier.is_torchao_like:
        return UnsupportedClassification(
            bucket="quantization_wrapper",
            strategy="explicit_blackbox",
            confidence="medium",
            reason="Quantized or TorchAO-like operator should preserve reference semantics first",
        )

    if dossier.is_custom:
        return UnsupportedClassification(
            bucket="opaque_custom_op",
            strategy="explicit_blackbox",
            confidence="medium",
            reason="Custom namespace should be isolated until a dedicated lowering exists",
        )

    if dossier.is_aten and _schema_is_simple_tensor_op(dossier.schema):
        return UnsupportedClassification(
            bucket="payload_decomposition",
            strategy="synthesized_external_call",
            confidence="medium",
            reason="Simple ATen tensor operator can be translated automatically into a typed external call",
        )

    return UnsupportedClassification(
        bucket="blackbox_boundary",
        strategy="explicit_blackbox",
        confidence="low",
        reason=f"{issue.target} requires an explicit blackbox boundary until a richer lowering exists",
    )


__all__ = ["UnsupportedClassification", "classify_operator_issue"]
