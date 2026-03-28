"""Runtime operator introspection for unsupported-op recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

import torch


@dataclass(frozen=True)
class ExampleTensorInfo:
    """Shape, dtype, and layout observed for one example tensor."""

    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...] = ()


@dataclass(frozen=True)
class UnsupportedOpDossier:
    """Installed-runtime dossier for one operator target."""

    target: str
    namespace: str
    operator: str
    overload: str
    schema: str
    tags: tuple[str, ...] = ()
    is_aten: bool = False
    is_custom: bool = False
    is_torchao_like: bool = False
    is_view: bool = False
    has_any_kernel: bool = False
    has_meta_kernel: bool = False
    export_decomposition_registered: bool = False
    payload_decomposition_registered: bool = False
    python_module: str = ""
    source_file: str = ""
    source_line: int | None = None
    example_inputs: tuple[ExampleTensorInfo, ...] = ()
    example_output: ExampleTensorInfo | None = None
    reference_callable: Any = None
    extra_metadata: dict[str, str] = field(default_factory=dict)


def parse_target(target: str) -> tuple[str, str, str]:
    """Split a target string like ``aten.addmm.default``."""

    parts = target.split(".")
    if len(parts) >= 3:
        namespace = parts[0]
        overload = parts[-1]
        operator = ".".join(parts[1:-1])
        return namespace, operator, overload
    if len(parts) == 2:
        return parts[0], parts[1], "default"
    return "", target, "default"


def resolve_reference_callable(target: str) -> Any | None:
    """Resolve ``torch.ops`` callable for a target string."""

    namespace, operator, overload = parse_target(target)
    if not namespace or not operator:
        return None
    try:
        namespace_obj = getattr(torch.ops, namespace)
        packet = namespace_obj
        for component in operator.split("."):
            packet = getattr(packet, component)
        return getattr(packet, overload)
    except Exception:
        return None


def _tensor_example(value: Any) -> ExampleTensorInfo | None:
    if isinstance(value, ExampleTensorInfo):
        return value
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    stride_attr = getattr(value, "stride", None)
    stride = tuple(stride_attr()) if callable(stride_attr) else tuple(stride_attr or ())
    return ExampleTensorInfo(
        shape=tuple(int(dim) for dim in value.shape),
        dtype=str(value.dtype).replace("torch.", ""),
        stride=stride,
    )


def _source_info(reference: Any) -> tuple[str, str, int | None]:
    """Best-effort Python source lookup for a runtime callable."""

    if reference is None:
        return "", "", None

    python_module = getattr(reference, "__module__", "") or ""
    try:
        source_file = inspect.getsourcefile(reference) or inspect.getfile(reference)
    except Exception:
        source_file = ""
    try:
        _, line = inspect.getsourcelines(reference)
    except Exception:
        line = None
    return python_module, source_file, line


def _dispatch_flags(reference: Any) -> tuple[bool, bool]:
    """Return ``(has_any_kernel, has_meta_kernel)`` when possible."""

    if reference is None:
        return False, False

    has_any_kernel = False
    has_meta_kernel = False
    try:
        has_any_kernel = bool(reference.has_kernel_for_any_dispatch_key())
    except Exception:
        has_any_kernel = False
    try:
        dispatch_key = torch._C.DispatchKey.Meta  # type: ignore[attr-defined]
        has_meta_kernel = bool(reference.has_kernel_for_dispatch_key(dispatch_key))
    except Exception:
        has_meta_kernel = False
    return has_any_kernel, has_meta_kernel


def build_operator_dossier(
    target: str,
    *,
    sample_args: tuple[Any, ...] = (),
    sample_output: Any = None,
    export_decomposition_registered: bool = False,
    payload_decomposition_registered: bool = False,
) -> UnsupportedOpDossier:
    """Build a dossier for an operator target from the installed runtime."""

    namespace, operator, overload = parse_target(target)
    reference = resolve_reference_callable(target)
    python_module, source_file, source_line = _source_info(reference)
    has_any_kernel, has_meta_kernel = _dispatch_flags(reference)
    tags = tuple(str(tag) for tag in (getattr(reference, "tags", None) or ()))
    schema = str(getattr(reference, "_schema", "")) if reference is not None else ""
    is_view = bool(getattr(reference, "is_view", False)) if reference is not None else False
    example_inputs = tuple(
        info for arg in sample_args if (info := _tensor_example(arg)) is not None
    )
    example_output_info = _tensor_example(sample_output)

    return UnsupportedOpDossier(
        target=target,
        namespace=namespace,
        operator=operator,
        overload=overload,
        schema=schema,
        tags=tags,
        is_aten=namespace == "aten",
        is_custom=namespace not in {"", "aten", "prims"},
        is_torchao_like="torchao" in python_module or "torchao" in target.lower() or "quant" in target.lower(),
        is_view=is_view,
        has_any_kernel=has_any_kernel,
        has_meta_kernel=has_meta_kernel,
        export_decomposition_registered=export_decomposition_registered,
        payload_decomposition_registered=payload_decomposition_registered,
        python_module=python_module,
        source_file=source_file,
        source_line=source_line,
        example_inputs=example_inputs,
        example_output=example_output_info,
        reference_callable=reference,
        extra_metadata={
            "overloadpacket": str(getattr(reference, "overloadpacket", "")) if reference is not None else "",
        },
    )


def runtime_versions() -> dict[str, str]:
    """Return installed runtime versions relevant to the capture boundary."""

    versions = {"torch": getattr(torch, "__version__", "unknown")}
    try:
        import torchao  # type: ignore
    except Exception:
        return versions
    versions["torchao"] = getattr(torchao, "__version__", "unknown")
    return versions


__all__ = [
    "ExampleTensorInfo",
    "UnsupportedOpDossier",
    "build_operator_dossier",
    "parse_target",
    "resolve_reference_callable",
    "runtime_versions",
]
