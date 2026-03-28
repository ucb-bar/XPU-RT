"""Pass 8: Specialize layout encodings for the target.

Calls the ``LayoutResolver.specialize()`` protocol to convert generic
layout encodings (e.g., ``tiled_128x64``) into target-specific
``PackSpecAttr`` values.

If no resolver is provided, uses the DefaultLayoutResolver which
keeps generic encodings unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

if TYPE_CHECKING:
    from compgen.targets.capability import CapabilitySpec

log = structlog.get_logger()

SPECIALIZED_ATTR = "compgen.layout_specialized"
PACK_SPEC_ATTR = "compgen.pack_spec"


def specialize_layouts(
    module: ModuleOp,
    *,
    resolver: Any | None = None,
    capabilities: CapabilitySpec | None = None,
) -> ModuleOp:
    """Specialize generic layout encodings for the target.

    For each op with a layout encoding:
    - Call resolver.specialize(encoding, capabilities) if available.
    - If specialization returns a PackSpecAttr, attach it to the op.
    - Otherwise, mark as specialized with the generic encoding.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    specialized = 0

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if SPECIALIZED_ATTR in op.attributes:
            continue

        # Find the op's encoding
        encoding_str = None
        for attr_key in ("compgen.propagated_encoding", "compgen.layout_hint", "compgen.encoding"):
            attr = op.attributes.get(attr_key)
            if attr and hasattr(attr, "data"):
                encoding_str = attr.data
                break

        if not encoding_str:
            continue

        # Consult ukernel tile_family hint if present
        tile_hint = op.attributes.get("compgen.ukernel_tile_family")
        if tile_hint and hasattr(tile_hint, "data") and tile_hint.data:
            encoding_str = f"{encoding_str}:{tile_hint.data}"

        # Try target-specific specialization
        pack_spec_str = None
        if resolver is not None and capabilities is not None:
            try:
                result = resolver.specialize(encoding_str, capabilities)
                if result is not None:
                    pack_spec_str = str(result)
            except Exception:
                pass

        if pack_spec_str:
            op.attributes[PACK_SPEC_ATTR] = StringAttr(pack_spec_str)

        op.attributes[SPECIALIZED_ATTR] = StringAttr(encoding_str)
        specialized += 1

    log.debug("layout.specialize_layouts", specialized=specialized)
    return module


__all__ = ["specialize_layouts"]
