"""Lowering from ukernel dialect to actual kernel calls.

Converts UkernelCallOp into concrete function calls appropriate for
the kernel backend (Triton, C, CUDA, NKI, vendor library).

Invariants:
    - Lowering is backend-specific (selected by calling_convention).
    - Lowered calls include workspace allocation.
    - Lowering preserves scheduling metadata for the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.ir.ukernel.ops import UkernelCallOp

log = structlog.get_logger()

# Backend prefixes for function name generation
_BACKEND_PREFIXES: dict[str, str] = {
    "c": "extern_",
    "triton": "triton_kernel_",
    "cuda": "cuda_launch_",
    "nki": "nki_",
}


@dataclass(frozen=True)
class LoweredCall:
    """A ukernel call lowered to a concrete function call.

    Attributes:
        function_name: Target function name.
        backend: Backend that will execute this call.
        operands: Operand references.
        results: Result references.
        workspace_bytes: Workspace allocation needed before the call.
        metadata: Scheduling metadata preserved from the original op.
    """

    function_name: str
    backend: str
    operands: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    workspace_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoweringResult:
    """Result of lowering a module's ukernel ops.

    Attributes:
        lowered_calls: List of lowered function call descriptions.
        diagnostics: Warnings or errors encountered during lowering.
    """

    lowered_calls: list[LoweredCall] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def lower_ukernel_to_call(module: Any, backend: str = "c") -> LoweringResult:
    """Lower ukernel ops to concrete function calls.

    Args:
        module: An iterable of ukernel ops (or any container with UkernelCallOp instances).
        backend: Target backend for lowering ("c", "triton", "cuda", "nki").

    Returns:
        LoweringResult with lowered call descriptions and diagnostics.
    """
    result = LoweringResult()
    prefix = _BACKEND_PREFIXES.get(backend)

    if prefix is None:
        result.diagnostics.append(f"Unknown backend '{backend}', falling back to 'c'")
        prefix = _BACKEND_PREFIXES["c"]
        backend = "c"

    # Walk the module for UkernelCallOp instances
    ops = module if isinstance(module, (list, tuple)) else [module]

    for op in ops:
        if not isinstance(op, UkernelCallOp):
            continue

        function_name = f"{prefix}{op.kernel_name}"

        lowered = LoweredCall(
            function_name=function_name,
            backend=backend,
            operands=list(op.operands),
            results=list(op.results),
            workspace_bytes=op.workspace_bytes,
            metadata=dict(op.metadata),
        )
        result.lowered_calls.append(lowered)

        log.debug(
            "ukernel.lower.lowered_call",
            kernel=op.kernel_name,
            function=function_name,
            backend=backend,
            workspace=op.workspace_bytes,
        )

    return result


def lower_ukernel_with_body(
    module: Any,
    registry: Any,
    target_family: str = "any",
    backend: str = "c",
) -> LoweringResult:
    """Lower ukernel ops with body-awareness from the registry.

    Selects the best body for each kernel call and lowers accordingly:
    - transparent + mlir/xdsl: keep visible, mark as inline-able
    - opaque + c/cpp: extern_ prefix (same as lower_ukernel_to_call)
    - opaque + library: vendor-specific call
    - opaque + triton: Triton JIT launch call
    - opaque + binary: function pointer call

    Args:
        module: Iterable of ukernel ops.
        registry: UkernelRegistry for body lookup.
        target_family: Target family for body selection.
        backend: Fallback backend if no body found.

    Returns:
        LoweringResult with body-aware lowered calls.
    """
    result = LoweringResult()
    ops = module if isinstance(module, (list, tuple)) else [module]

    # Body-kind specific prefixes
    _body_prefixes: dict[str, str] = {
        "c": "extern_",
        "cpp": "extern_",
        "triton": "triton_kernel_",
        "library": "lib_",
        "binary": "fptr_",
        "mlir": "inline_",
        "xdsl": "inline_",
        "python": "py_",
    }

    for op in ops:
        if not isinstance(op, UkernelCallOp):
            continue

        # Try to find a body from the registry
        body = None
        body_kind = backend
        if registry is not None:
            body = registry.select_body(op.kernel_name, target_family)
            if body is not None:
                body_kind = body.body_kind

        prefix = _body_prefixes.get(body_kind, _BACKEND_PREFIXES.get(backend, "extern_"))
        function_name = f"{prefix}{op.kernel_name}"

        # For library bodies, use source_ref as the function name if available
        if body and body_kind == "library" and body.source_ref:
            function_name = body.source_ref

        lowered = LoweredCall(
            function_name=function_name,
            backend=body_kind,
            operands=list(op.operands),
            results=list(op.results),
            workspace_bytes=op.workspace_bytes,
            metadata={
                **dict(op.metadata),
                "body_kind": body_kind,
                "transparency": body.transparency if body else "opaque",
                "target_family": body.target_family if body else target_family,
            },
        )
        result.lowered_calls.append(lowered)

        log.debug(
            "ukernel.lower.body_aware",
            kernel=op.kernel_name,
            function=function_name,
            body_kind=body_kind,
            transparency=body.transparency if body else "opaque",
        )

    return result


__all__ = ["LoweredCall", "LoweringResult", "lower_ukernel_to_call", "lower_ukernel_with_body"]
