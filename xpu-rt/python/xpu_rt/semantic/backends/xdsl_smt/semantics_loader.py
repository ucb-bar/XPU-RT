"""Semantics loader for CompGen's custom dialects.

Registers Z3-lowering functions for CompGen's Tile IR, Accel dialect,
and other custom ops so they can participate in translation validation.

This is the CompGen-side equivalent of the artifact's
``load_vanilla_semantics()`` — it extends the ArithZ3Lowerer with
handlers for CompGen-specific operations.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()

# Registry of op-type → Z3 lowering function
# Each function has signature: (lowerer, op, prefix) -> None
_COMPGEN_OP_HANDLERS: dict[type, Any] = {}


def register_compgen_semantics() -> None:
    """Register CompGen dialect semantics with the ArithZ3Lowerer.

    Call this once at startup to make Tile/Accel ops verifiable.
    Currently a stub — filled in Phase 6.
    """
    log.info("smt.semantics.register", num_handlers=len(_COMPGEN_OP_HANDLERS))


def register_op_handler(op_type: type, handler: Any) -> None:
    """Register a Z3 lowering handler for a custom op type.

    Args:
        op_type: The xDSL Operation subclass.
        handler: Callable(lowerer, op, prefix) -> None that populates
                 lowerer._values for the op's results.
    """
    _COMPGEN_OP_HANDLERS[op_type] = handler


def get_op_handler(op_type: type) -> Any | None:
    """Get the registered Z3 lowering handler for an op type."""
    return _COMPGEN_OP_HANDLERS.get(op_type)


__all__ = ["get_op_handler", "register_compgen_semantics", "register_op_handler"]
