"""Recipe IR custom attributes.

Defines ParametrizedAttribute types used by Recipe IR operations:
    - ShapeSummaryAttr: tensor shape/dtype summary
    - EffectClassAttr: side-effect classification
    - CostAttr: cost estimate with confidence
    - ProvenanceAttr: creation provenance (agent/eqsat/template/seed)
    - DeviceRefAttr: device reference by index+name
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def


@irdl_attr_definition
class ShapeSummaryAttr(ParametrizedAttribute):
    """Summary of tensor shape and dtype for a payload region."""

    name = "recipe.shape_summary"
    dims: ArrayAttr = param_def(ArrayAttr)
    dtype: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        dims: list[int] | ArrayAttr,
        dtype: str | StringAttr,
    ) -> None:
        if isinstance(dims, list):
            dims = ArrayAttr(
                [IntegerAttr(d, IntegerType(64)) for d in dims],
            )
        if isinstance(dtype, str):
            dtype = StringAttr(dtype)
        super().__init__(dims, dtype)


@irdl_attr_definition
class EffectClassAttr(ParametrizedAttribute):
    """Side-effect classification for a region.

    Valid kinds: "pure", "read", "write", "readwrite".
    """

    name = "recipe.effect_class"
    kind: StringAttr = param_def(StringAttr)

    def __init__(self, kind: str | StringAttr) -> None:
        if isinstance(kind, str):
            kind = StringAttr(kind)
        super().__init__(kind)


@irdl_attr_definition
class CostAttr(ParametrizedAttribute):
    """Cost/latency estimate with confidence level.

    Attributes:
        value_us: Estimated cost in microseconds.
        confidence: "measured", "estimated", or "unknown".
    """

    name = "recipe.cost"
    value_us: IntegerAttr = param_def(IntegerAttr)
    confidence: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        value_us: int | IntegerAttr,
        confidence: str | StringAttr,
    ) -> None:
        if isinstance(value_us, int):
            value_us = IntegerAttr(value_us, IntegerType(64))
        if isinstance(confidence, str):
            confidence = StringAttr(confidence)
        super().__init__(value_us, confidence)


@irdl_attr_definition
class ProvenanceAttr(ParametrizedAttribute):
    """Provenance metadata: who created this and at what iteration.

    Attributes:
        source: "agent", "eqsat", "template", or "seed".
        iteration: Generation/iteration number.
    """

    name = "recipe.provenance"
    source: StringAttr = param_def(StringAttr)
    iteration: IntegerAttr = param_def(IntegerAttr)

    def __init__(
        self,
        source: str | StringAttr,
        iteration: int | IntegerAttr,
    ) -> None:
        if isinstance(source, str):
            source = StringAttr(source)
        if isinstance(iteration, int):
            iteration = IntegerAttr(iteration, IntegerType(64))
        super().__init__(source, iteration)


@irdl_attr_definition
class DeviceRefAttr(ParametrizedAttribute):
    """Reference to a device by index and name.

    Attributes:
        index: Device index into the target profile's device list.
        device_name: Human-readable device name.
    """

    name = "recipe.device_ref"
    index: IntegerAttr = param_def(IntegerAttr)
    device_name: StringAttr = param_def(StringAttr)

    def __init__(
        self,
        index: int | IntegerAttr,
        device_name: str | StringAttr,
    ) -> None:
        if isinstance(index, int):
            index = IntegerAttr(index, IntegerType(64))
        if isinstance(device_name, str):
            device_name = StringAttr(device_name)
        super().__init__(index, device_name)


__all__ = [
    "CostAttr",
    "DeviceRefAttr",
    "EffectClassAttr",
    "ProvenanceAttr",
    "ShapeSummaryAttr",
]
