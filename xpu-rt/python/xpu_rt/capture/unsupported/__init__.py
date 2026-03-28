"""Unsupported-operator recovery pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compgen.capture.unsupported.classify import UnsupportedClassification, classify_operator_issue
from compgen.capture.unsupported.detect import UnsupportedOperatorIssue, detect_unsupported_operators
from compgen.capture.unsupported.introspect import UnsupportedOpDossier, build_operator_dossier
from compgen.capture.unsupported.promote import PromotionRecord, build_promotion_record
from compgen.capture.unsupported.synthesize_translation import (
    SynthesizedPayloadTranslation,
    synthesize_payload_translation,
)
from compgen.capture.unsupported.verify import UnsupportedVerification, verify_unsupported_resolution


@dataclass(frozen=True)
class UnsupportedOpResolution:
    """Full recovery result for one unsupported operator target."""

    issue: UnsupportedOperatorIssue
    dossier: UnsupportedOpDossier
    classification: UnsupportedClassification
    verification: UnsupportedVerification
    promotion: PromotionRecord
    translation: SynthesizedPayloadTranslation | None = None
    approved_blackbox: bool = False

    @property
    def target(self) -> str:
        return self.issue.target


def recover_unsupported_operators(
    exported_program: Any,
    *,
    supported_targets: set[str],
    runtime_versions: dict[str, str],
    explicit_targets: set[str] | None = None,
) -> list[UnsupportedOpResolution]:
    """Run detection, introspection, classification, synthesis, and verification."""

    resolutions: list[UnsupportedOpResolution] = []
    for issue in detect_unsupported_operators(
        exported_program,
        supported_targets=supported_targets,
        explicit_targets=explicit_targets,
    ):
        dossier = build_operator_dossier(
            issue.target,
            sample_args=tuple(issue.example_inputs),
            sample_output=issue.example_output,
            payload_decomposition_registered=issue.target in supported_targets,
        )
        classification = classify_operator_issue(issue, dossier)
        translation = synthesize_payload_translation(issue, dossier, classification)
        verification = verify_unsupported_resolution(issue, dossier, translation)
        promotion = build_promotion_record(dossier, classification, runtime_versions)
        approved_blackbox = classification.strategy == "explicit_blackbox"
        resolutions.append(UnsupportedOpResolution(
            issue=issue,
            dossier=dossier,
            classification=classification,
            verification=verification,
            promotion=promotion,
            translation=translation,
            approved_blackbox=approved_blackbox,
        ))
    return resolutions


__all__ = [
    "SynthesizedPayloadTranslation",
    "UnsupportedClassification",
    "UnsupportedOpDossier",
    "UnsupportedOpResolution",
    "UnsupportedOperatorIssue",
    "UnsupportedVerification",
    "build_operator_dossier",
    "classify_operator_issue",
    "detect_unsupported_operators",
    "recover_unsupported_operators",
    "verify_unsupported_resolution",
]
