"""torch-mlir bridge: nn.Module / ExportedProgram -> xDSL ModuleOp.

Mirrors the `hexagon-mlir` production pattern exactly (see
`tmp/hexagon-mlir/test/python/torch-mlir/utils.py:28-34` +
`qcom_hexagon_backend/backend/torch_mlir_hexagon_launcher.py`):

    torch_mlir.fx.export_and_import(model, *inputs,
                                    output_type="linalg-on-tensors")

The returned MLIR text is then parsed back into an xDSL ModuleOp so
the rest of CompGen's passes can operate on it.

When torch-mlir is not installed (no `cp312` wheel on PyPI today; it
ships as a source build), the bridge falls back to CompGen's own
``FXImporter``. The caller gets a diagnostic string telling them
which path was taken.

The bridge is deliberately thin: zero business logic, zero op-level
translation. The whole point is to delegate `ATen -> linalg` to
torch-mlir when available (which handles hundreds of ops correctly)
instead of us expanding our own decomposition table.

Usage:

    from compgen.capture.torch_mlir_bridge import bridge_fx_graph
    result = bridge_fx_graph(model, example_inputs)
    if result.module is not None:
        # run downstream passes on result.module
        ...
    else:
        raise RuntimeError(result.diagnostics)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import structlog
import torch
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

log = structlog.get_logger()


@dataclass
class BridgeResult:
    """Outcome of running the FX->MLIR bridge.

    Attributes:
        module: the parsed xDSL ModuleOp, or ``None`` on failure.
        path_taken: ``"torch_mlir"`` when the torch-mlir path succeeded,
            ``"fx_importer"`` when the CompGen FXImporter fallback ran,
            ``"failed"`` when both paths failed.
        output_type: the ``output_type`` the torch-mlir path used
            (``"linalg-on-tensors"`` by default).
        mlir_text: raw MLIR text produced by torch-mlir (empty when
            the fallback was used).
        diagnostics: list of human-readable diagnostic strings.
    """

    module: ModuleOp | None = None
    path_taken: str = "failed"
    output_type: str = ""
    mlir_text: str = ""
    diagnostics: list[str] = field(default_factory=list)


def _try_torch_mlir_import() -> Any:
    """Return the torch-mlir ``fx`` module, or ``None`` when unavailable.

    Lazily imports so CompGen packages that never call the bridge pay
    zero import cost.
    """
    try:
        from torch_mlir import fx as _fx  # type: ignore
    except ImportError:
        return None
    return _fx


def _parse_mlir_text_to_xdsl(mlir_text: str) -> ModuleOp | None:
    """Parse linalg-on-tensors MLIR text back into an xDSL ModuleOp.

    xDSL's Parser understands most of builtin/linalg/arith/tensor/func
    out of the box. Returns ``None`` if parsing fails -- typically
    because the MLIR text uses a dialect xDSL doesn't register by
    default.
    """
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Func)
    ctx.load_dialect(Linalg)
    ctx.load_dialect(Math)
    ctx.load_dialect(Tensor)

    # Register CompGen's own dialects so they round-trip if present.
    try:
        from compgen.ir.linalg_ext import LinalgExt
        from compgen.ir.quant import Quant
        from compgen.ir.tensor_ext import TensorExt
        ctx.load_dialect(LinalgExt)
        ctx.load_dialect(Quant)
        ctx.load_dialect(TensorExt)
    except Exception:
        pass  # Optional — bridge works without them.

    parser = Parser(ctx, mlir_text)
    try:
        module = parser.parse_module()
    except Exception as exc:  # noqa: BLE001
        log.warning("torch_mlir_bridge.parse_failed", error=str(exc))
        return None
    return module


def bridge_fx_graph(
    model: torch.nn.Module | Any,
    example_inputs: tuple[torch.Tensor, ...],
    *,
    func_name: str = "forward",
    output_type: str = "linalg-on-tensors",
    allow_fallback: bool = True,
) -> BridgeResult:
    """Convert ``model`` + ``example_inputs`` into an xDSL ModuleOp.

    Args:
        model: a ``torch.nn.Module``. torch-mlir also accepts an
            ``ExportedProgram`` via the same API.
        example_inputs: the tuple of example tensors (same shapes the
            compiled artifact will be called with).
        func_name: name of the public func in the emitted MLIR.
        output_type: torch-mlir output dialect. ``"linalg-on-tensors"``
            is the right choice for CompGen (linalg is our downstream
            substrate). Pass ``"torch"`` for the higher-level Torch
            dialect.
        allow_fallback: when ``True`` (default), fall back to CompGen's
            ``FXImporter`` if torch-mlir is unavailable or its import
            fails. When ``False``, a torch-mlir failure becomes a hard
            error and the returned ``module`` is ``None``.
    """
    result = BridgeResult(output_type=output_type)

    fx_module = _try_torch_mlir_import()
    if fx_module is not None:
        try:
            mlir_module = fx_module.export_and_import(
                model,
                *example_inputs,
                output_type=output_type,
                func_name=func_name,
            )
            # torch-mlir returns an MlirModule; serialize to text and
            # parse into xDSL.
            mlir_text = mlir_module.operation.get_asm(
                binary=False,
                large_elements_limit=64,
                enable_debug_info=False,
            )
            result.mlir_text = mlir_text
            module = _parse_mlir_text_to_xdsl(mlir_text)
            if module is not None:
                result.module = module
                result.path_taken = "torch_mlir"
                result.diagnostics.append(
                    f"torch-mlir path produced {len(mlir_text)} chars of "
                    f"{output_type} MLIR; parsed into xDSL"
                )
                log.info(
                    "torch_mlir_bridge.ok",
                    path="torch_mlir",
                    mlir_bytes=len(mlir_text),
                )
                return result
            result.diagnostics.append(
                "torch-mlir produced MLIR text but xDSL could not parse it"
            )
        except Exception as exc:  # noqa: BLE001
            result.diagnostics.append(f"torch-mlir path raised: {exc}")
            log.warning("torch_mlir_bridge.torch_mlir_failed", error=str(exc))
    else:
        result.diagnostics.append(
            "torch-mlir not installed; falling back to CompGen FXImporter"
        )

    if not allow_fallback:
        result.diagnostics.append("allow_fallback=False; returning no module")
        return result

    # Fallback: use CompGen's FXImporter via torch.export capture.
    try:
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import FXImporter

        exported = capture_model(model, example_inputs)
        importer = FXImporter()
        module = importer.import_graph(exported)
        errors = [d for d in importer.diagnostics if d.level == "error"]
        if errors:
            result.diagnostics.append(
                f"FXImporter fallback produced {len(errors)} errors: "
                f"{[d.message for d in errors[:3]]}"
            )
            return result
        result.module = module
        result.path_taken = "fx_importer"
        result.diagnostics.append(
            f"FXImporter fallback succeeded with "
            f"{importer.decomposed_count} decomposed ops, "
            f"{importer.opaque_count} opaque"
        )
        log.info(
            "torch_mlir_bridge.ok",
            path="fx_importer",
            decomposed=importer.decomposed_count,
            opaque=importer.opaque_count,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result.diagnostics.append(f"FXImporter fallback raised: {exc}")
        log.error("torch_mlir_bridge.both_failed", error=str(exc))
        return result


def bridge_fx_graph_or_raise(
    model: torch.nn.Module | Any,
    example_inputs: tuple[torch.Tensor, ...],
    **kwargs: Any,
) -> ModuleOp:
    """Raise-on-failure wrapper around :func:`bridge_fx_graph`."""
    result = bridge_fx_graph(model, example_inputs, **kwargs)
    if result.module is None:
        raise RuntimeError(
            "FX -> xDSL bridge failed for both paths:\n  "
            + "\n  ".join(result.diagnostics)
        )
    return result.module


def module_to_text(module: ModuleOp) -> str:
    """Pretty-print an xDSL ModuleOp as MLIR text (for debugging)."""
    buf = io.StringIO()
    Printer(stream=buf).print(module)
    return buf.getvalue()


def torch_mlir_available() -> bool:
    """Whether the torch-mlir path can be used."""
    return _try_torch_mlir_import() is not None


__all__ = [
    "BridgeResult",
    "bridge_fx_graph",
    "bridge_fx_graph_or_raise",
    "module_to_text",
    "torch_mlir_available",
]
