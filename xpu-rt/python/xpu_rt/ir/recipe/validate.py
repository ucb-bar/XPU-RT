"""Structural validation of Recipe IR programs.

Validates Recipe IR using xDSL's built-in verification infrastructure plus
custom structural checks:
    - xDSL ``verify_()`` on each op (catches tile sizes <= 0, etc.).
    - All symbol references resolve to defined symbols.
    - No conflicting device assignments for the same region.
    - Search budgets are positive (via verify_()).

Invariants:
    - Validation never modifies the Recipe IR.
    - All errors are collected (not fail-fast) for batch reporting.
    - The old ``validate_recipe(list[RecipeOp])`` API is preserved for
      backward compatibility by converting through ``compat.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Operation
from xdsl.utils.exceptions import VerifyException

from compgen.ir.recipe.ops import RecipeOp
from compgen.ir.recipe.ops_candidate import PlaceOnDeviceOp
from compgen.ir.recipe.ops_scope import AnchorOp, RecipeGuardOp, RecipeRegionOp, SegmentOp

log = structlog.get_logger()


@dataclass(frozen=True)
class RecipeValidationError:
    """A single validation error in a Recipe IR program."""

    op_index: int
    op_type: str
    message: str


@dataclass(frozen=True)
class RecipeValidationResult:
    """Result of Recipe IR validation."""

    valid: bool
    errors: list[RecipeValidationError] = field(default_factory=list)


def _collect_defined_symbols(module: ModuleOp) -> set[str]:
    """Walk the module and collect all defined symbol names.

    Args:
        module: The xDSL ModuleOp to inspect.

    Returns:
        Set of symbol name strings defined by RecipeRegionOp, SegmentOp,
        and AnchorOp instances in the module.
    """
    symbols: set[str] = set()
    for op in module.walk():
        if isinstance(op, (RecipeRegionOp, SegmentOp, AnchorOp, RecipeGuardOp)):
            symbols.add(op.sym_name.data)
            continue
        if hasattr(op, "sym_name"):
            sym_name = getattr(op, "sym_name")
            if isinstance(sym_name, StringAttr):
                symbols.add(sym_name.data)
    return symbols


def _collect_symbol_refs_from_attr(attr: Attribute) -> list[str]:
    refs: list[str] = []
    if isinstance(attr, SymbolRefAttr):
        refs.append(attr.root_reference.data)
    elif isinstance(attr, ArrayAttr):
        for item in attr.data:
            refs.extend(_collect_symbol_refs_from_attr(item))
    elif hasattr(attr, "parameters"):
        for param in getattr(attr, "parameters", ()):
            if isinstance(param, Attribute):
                refs.extend(_collect_symbol_refs_from_attr(param))
    return refs


def _collect_symbol_refs(op: Operation) -> list[str]:
    """Extract all SymbolRefAttr root references from an operation's properties.

    Args:
        op: The operation to inspect.

    Returns:
        List of root reference strings from any SymbolRefAttr properties.
    """
    refs: list[str] = []
    if not hasattr(op, "properties"):
        return refs
    for attr in op.properties.values():
        refs.extend(_collect_symbol_refs_from_attr(attr))
    return refs


def validate_recipe_module(module: ModuleOp) -> RecipeValidationResult:
    """Validate a Recipe IR module using xDSL verification and custom checks.

    Steps:
        1. Run ``module.verify()`` -- calls ``verify_()`` on each op (catches
           tile sizes <= 0, search budget <= 0, etc.).
        2. Collect all defined symbols (RecipeRegionOp, SegmentOp, AnchorOp
           ``sym_name`` values).
        3. Check all SymbolRefAttr references resolve to defined symbols.
        4. Detect conflicting PlaceOnDeviceOp (same ``region_ref``, different
           device).

    Args:
        module: An xDSL ModuleOp containing Recipe IR operations.

    Returns:
        RecipeValidationResult with collected errors.
    """
    errors: list[RecipeValidationError] = []

    # --- Step 1: xDSL structural verification ---------------------------------
    try:
        module.verify()
    except VerifyException as exc:
        errors.append(RecipeValidationError(
            op_index=-1,
            op_type="ModuleOp",
            message=f"xDSL verification failed: {exc}",
        ))
        log.warning("validate.xdsl_verify_failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        errors.append(RecipeValidationError(
            op_index=-1,
            op_type="ModuleOp",
            message=f"Unexpected verification error: {exc}",
        ))
        log.error("validate.unexpected_verify_error", error=str(exc))

    # --- Step 2: Collect defined symbols --------------------------------------
    defined_symbols = _collect_defined_symbols(module)

    # --- Step 3: Check symbol references resolve ------------------------------
    for i, op in enumerate(module.walk()):
        if isinstance(op, ModuleOp):
            continue
        op_type_name = type(op).__name__
        refs = _collect_symbol_refs(op)
        for ref in refs:
            if ref and ref not in defined_symbols:
                errors.append(RecipeValidationError(
                    op_index=i,
                    op_type=op_type_name,
                    message=f"Unresolved symbol reference: @{ref}",
                ))

    # --- Step 4: Detect conflicting device assignments ------------------------
    device_assignments: dict[str, tuple[int, int]] = {}  # region_ref -> (device_index, first_op_index)
    for i, op in enumerate(module.walk()):
        if isinstance(op, PlaceOnDeviceOp):
            region_id = op.region_ref.root_reference.data
            device_index = op.device.index.value.data
            if region_id in device_assignments:
                prev_device, prev_idx = device_assignments[region_id]
                if prev_device != device_index:
                    errors.append(RecipeValidationError(
                        op_index=i,
                        op_type="PlaceOnDeviceOp",
                        message=(
                            f"Conflicting device assignment for region @{region_id}: "
                            f"was device {prev_device} (op {prev_idx}), now device {device_index}"
                        ),
                    ))
            else:
                device_assignments[region_id] = (device_index, i)

    valid = len(errors) == 0
    log.info(
        "validate.recipe_module",
        valid=valid,
        error_count=len(errors),
    )
    return RecipeValidationResult(valid=valid, errors=errors)


def validate_recipe(ops: list[RecipeOp]) -> RecipeValidationResult:
    """Validate a Recipe IR program (backward-compatible entry point).

    Converts the legacy ``list[RecipeOp]`` to an xDSL ``ModuleOp`` via
    ``compat.recipe_list_to_module`` and delegates to
    ``validate_recipe_module``.

    Args:
        ops: List of legacy Recipe IR operations.

    Returns:
        RecipeValidationResult.
    """
    # Import here to avoid circular imports at module level
    from compgen.ir.recipe.compat import recipe_list_to_module

    module = recipe_list_to_module(ops)
    return validate_recipe_module(module)


__all__ = [
    "RecipeValidationError",
    "RecipeValidationResult",
    "validate_recipe",
    "validate_recipe_module",
]
