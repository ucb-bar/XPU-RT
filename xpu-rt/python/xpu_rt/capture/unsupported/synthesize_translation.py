"""Synthesize bounded Payload translations for unsupported operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type, StringAttr, TensorType
from xdsl.dialects.func import CallOp
from xdsl.ir import SSAValue

from compgen.capture.unsupported.classify import UnsupportedClassification
from compgen.capture.unsupported.detect import UnsupportedOperatorIssue
from compgen.capture.unsupported.introspect import UnsupportedOpDossier, parse_target
from compgen.ir.payload.decompositions import DecompFn, DecompResult


@dataclass(frozen=True)
class SynthesizedPayloadTranslation:
    """A synthesized Payload-level translation for an unsupported operator."""

    target: str
    kind: str
    translator: DecompFn
    callee_name: str


def _torch_dtype_to_xdsl(dtype: torch.dtype | None) -> Any:
    mapping = {
        torch.float16: Float16Type,
        torch.float32: Float32Type,
        torch.float64: Float64Type,
    }
    factory = mapping.get(dtype, Float32Type)
    return factory()


def _result_type_from_meta(meta: dict[str, Any]) -> TensorType:
    value = meta.get("val")
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return TensorType(_torch_dtype_to_xdsl(value.dtype), list(value.shape))
    return TensorType(Float32Type(), [1])


def _make_external_call_translation(target: str) -> SynthesizedPayloadTranslation:
    namespace, operator, overload = parse_target(target)
    sanitized = f"{namespace}_{operator.replace('.', '_')}_{overload}".strip("_")
    region_prefix = operator.split(".")[-1].replace("_", "") or "unsupported"

    def translate(
        operands: list[SSAValue],
        meta: dict[str, Any],
        node_name: str,
    ) -> DecompResult:
        result_type = _result_type_from_meta(meta)
        call = CallOp(sanitized, operands, [result_type])
        call.attributes["compgen.region_id"] = StringAttr(f"{region_prefix}_{node_name}")
        return DecompResult(ops=[call], result=call.res[0], region_ids=[f"{region_prefix}_{node_name}"])

    return SynthesizedPayloadTranslation(
        target=target,
        kind="external_call",
        translator=translate,
        callee_name=sanitized,
    )


def synthesize_payload_translation(
    issue: UnsupportedOperatorIssue,
    dossier: UnsupportedOpDossier,
    classification: UnsupportedClassification,
) -> SynthesizedPayloadTranslation | None:
    """Synthesize a bounded translation strategy when classification allows it."""

    if classification.strategy != "synthesized_external_call":
        return None

    # Restrict automatic synthesis to simple tensor-returning ATen ops.
    if "-> Tensor" not in dossier.schema:
        return None
    if dossier.schema.count("Tensor") > 3:
        return None
    return _make_external_call_translation(issue.target)


__all__ = ["SynthesizedPayloadTranslation", "synthesize_payload_translation"]
