"""Layout IR operations.

Four operations for virtual layout encoding and physical materialization:
    - SetLayoutOp: annotate a tensor with a virtual layout encoding
    - UnsetLayoutOp: mark a boundary where layout must materialize
    - PackOp: materialize a layout encoding into a physical pack
    - UnpackOp: materialize a layout encoding into a physical unpack
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr

# Reuse ProvenanceAttr from Recipe IR for lineage tracking
from compgen.ir.recipe.attrs import ProvenanceAttr


@irdl_op_definition
class SetLayoutOp(IRDLOperation):
    """Annotate a tensor with a virtual layout encoding.

    Attaches a layout encoding to a tensor reference, declaring its
    logical layout without materializing any data movement.
    """

    name = "layout.set_layout"

    encoding = prop_def(LayoutEncodingAttr)
    source_ref = prop_def(SymbolRefAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class UnsetLayoutOp(IRDLOperation):
    """Mark a boundary where layout encoding must materialize.

    Removes a virtual layout encoding, signaling that physical
    pack/transpose operations are needed at this boundary.
    """

    name = "layout.unset_layout"

    source_ref = prop_def(SymbolRefAttr)
    boundary_reason = opt_prop_def(StringAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class PackOp(IRDLOperation):
    """Materialize a layout encoding into a physical pack operation.

    Applies a pack specification to transform tensor data from its
    original layout into the target tiled/blocked layout.
    """

    name = "layout.pack"

    source_ref = prop_def(SymbolRefAttr)
    pack_spec = prop_def(PackSpecAttr)
    is_prepack = prop_def(IntegerAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        val = self.is_prepack.value.data
        # i1 stores 1 as -1 (signed 1-bit); accept both representations
        if val not in (0, 1, -1):
            raise VerifyException(f"layout.pack is_prepack must be 0 or 1, got {val}")


@irdl_op_definition
class UnpackOp(IRDLOperation):
    """Materialize a layout encoding into a physical unpack operation.

    Reverses a pack operation, transforming tensor data from the
    tiled/blocked layout back to the original layout.
    """

    name = "layout.unpack"

    source_ref = prop_def(SymbolRefAttr)
    pack_spec = prop_def(PackSpecAttr)
    provenance = opt_prop_def(ProvenanceAttr)

    traits = traits_def(Pure())


__all__ = [
    "PackOp",
    "SetLayoutOp",
    "UnpackOp",
    "UnsetLayoutOp",
]
