"""Verification helpers for unsupported-op recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from compgen.capture.unsupported.detect import UnsupportedOperatorIssue
from compgen.capture.unsupported.introspect import ExampleTensorInfo, UnsupportedOpDossier
from compgen.capture.unsupported.synthesize_translation import SynthesizedPayloadTranslation


@dataclass(frozen=True)
class UnsupportedVerification:
    """Verification result for a synthesized or blackbox recovery."""

    schema_ok: bool
    eager_reference_ok: bool
    meta_reference_ok: bool
    messages: tuple[str, ...] = ()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }.get(name, torch.float32)


def _make_tensor(example: ExampleTensorInfo, *, device: str = "cpu") -> torch.Tensor:
    shape = tuple(dim if dim > 0 else 1 for dim in example.shape)
    dtype = _dtype_from_name(example.dtype)
    if dtype.is_floating_point:
        return torch.randn(*shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=device)
    return torch.randint(0, 7, shape, dtype=dtype, device=device)


def _invoke_reference(dossier: UnsupportedOpDossier, examples: tuple[ExampleTensorInfo, ...], *, device: str) -> Any:
    reference = dossier.reference_callable
    if reference is None:
        raise RuntimeError("no reference callable available")

    schema = getattr(reference, "_schema", None)
    args: list[Any] = []
    example_iter = iter(examples)
    if schema is None:
        return reference(*[_make_tensor(example, device=device) for example in examples])

    for arg in schema.arguments:
        arg_type = str(getattr(arg, "type", ""))
        default_value = getattr(arg, "default_value", None)
        if "Tensor" in arg_type:
            try:
                example = next(example_iter)
            except StopIteration as exc:  # pragma: no cover - defensive
                raise RuntimeError("missing tensor example for schema argument") from exc
            args.append(_make_tensor(example, device=device))
        elif default_value is not None and not getattr(arg, "kwarg_only", False):
            args.append(default_value)
        elif getattr(arg, "kwarg_only", False):
            continue
        else:
            raise RuntimeError(f"unsupported required non-tensor argument: {arg.name}")
    return reference(*args)


def verify_unsupported_resolution(
    issue: UnsupportedOperatorIssue,
    dossier: UnsupportedOpDossier,
    translation: SynthesizedPayloadTranslation | None,
) -> UnsupportedVerification:
    """Verify the installed eager operator can be used as a recovery oracle."""

    messages: list[str] = []
    schema_ok = bool(dossier.schema)
    eager_reference_ok = False
    meta_reference_ok = False

    if not schema_ok:
        messages.append("missing operator schema")

    if dossier.reference_callable is None:
        messages.append("missing eager reference callable")
    else:
        try:
            eager_out = _invoke_reference(dossier, issue.example_inputs or dossier.example_inputs, device="cpu")
            eager_reference_ok = True
            if dossier.example_output is not None and hasattr(eager_out, "shape"):
                if tuple(int(dim) for dim in eager_out.shape) != dossier.example_output.shape:
                    eager_reference_ok = False
                    messages.append(
                        f"eager output shape mismatch: expected {dossier.example_output.shape}, "
                        f"got {tuple(int(dim) for dim in eager_out.shape)}"
                    )
        except Exception as exc:
            messages.append(f"eager reference failed: {exc}")

        if dossier.has_meta_kernel:
            try:
                _invoke_reference(dossier, issue.example_inputs or dossier.example_inputs, device="meta")
                meta_reference_ok = True
            except Exception as exc:
                messages.append(f"meta reference failed: {exc}")
        else:
            messages.append("no meta kernel registered")

    if translation is not None:
        messages.append(f"synthesized {translation.kind} translation: {translation.callee_name}")

    return UnsupportedVerification(
        schema_ok=schema_ok,
        eager_reference_ok=eager_reference_ok,
        meta_reference_ok=meta_reference_ok,
        messages=tuple(messages),
    )


__all__ = ["UnsupportedVerification", "verify_unsupported_resolution"]
