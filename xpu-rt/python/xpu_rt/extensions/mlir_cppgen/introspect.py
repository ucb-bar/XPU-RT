"""Introspect live xDSL Dialect objects into generation-ready dataclasses.

Reads xDSL ``Dialect`` instances via ``get_irdl_definition()`` and
produces ``DialectInfo`` records that downstream emitters consume to
generate TableGen, C++, and CMake files.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation

# ---------------------------------------------------------------------------
# Type mapping: xDSL attr class name → TableGen type string
# ---------------------------------------------------------------------------

_BUILTIN_ATTR_MAP: dict[str, str] = {
    "StringAttr": "StrAttr",
    "IntegerAttr": "I64Attr",
    "SymbolRefAttr": "FlatSymbolRefAttr",
    "ArrayAttr": "ArrayAttr",
    "IntegerType": "I64Attr",
}


# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrFieldInfo:
    """One parameter of a custom ParametrizedAttribute."""

    name: str
    xdsl_type: str
    tablegen_type: str
    is_optional: bool = False


@dataclass(frozen=True)
class AttrInfo:
    """Introspected custom attribute (e.g. LayoutEncodingAttr)."""

    class_name: str
    mlir_mnemonic: str
    dialect_prefix: str
    cpp_class: str
    fields: list[AttrFieldInfo]


@dataclass(frozen=True)
class PropInfo:
    """One property of an op."""

    name: str
    xdsl_type: str
    tablegen_type: str
    is_optional: bool


@dataclass(frozen=True)
class VerifierInfo:
    """Verifier metadata extracted from an op class."""

    kind: str  # "enum_check", "range_check", "dimension_check", "custom"
    property_name: str
    valid_values: frozenset[str] | None = None
    min_dims: int | None = None


@dataclass(frozen=True)
class OpInfo:
    """Introspected xDSL IRDL operation."""

    class_name: str
    mnemonic: str
    properties: list[PropInfo]
    has_region: bool
    traits: list[str]
    verifier: VerifierInfo | None
    summary: str


@dataclass
class DialectInfo:
    """Complete introspected dialect ready for code generation."""

    name: str
    cpp_namespace: str
    prefix: str  # TableGen prefix (e.g. "Layout")
    ops: list[OpInfo] = field(default_factory=list)
    attrs: list[AttrInfo] = field(default_factory=list)
    dep_dialect_prefixes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

# Cache of custom attr type → (dialect_prefix, tablegen_type)
_custom_attr_registry: dict[str, tuple[str, str]] = {}


def register_custom_attr(class_name: str, dialect_prefix: str) -> str:
    """Register a custom attr so property introspection can resolve it."""
    tg_name = f"{dialect_prefix}_{class_name}"
    _custom_attr_registry[class_name] = (dialect_prefix, tg_name)
    return tg_name


def _resolve_tablegen_type(xdsl_type_name: str) -> str:
    """Resolve an xDSL type name to a TableGen type string."""
    if xdsl_type_name in _BUILTIN_ATTR_MAP:
        return _BUILTIN_ATTR_MAP[xdsl_type_name]
    if xdsl_type_name in _custom_attr_registry:
        return _custom_attr_registry[xdsl_type_name][1]
    return xdsl_type_name


def _dialect_prefix(name: str) -> str:
    """Convert dialect name to CamelCase prefix: 'layout' → 'Layout'."""
    parts = name.replace(".", "_").split("_")
    return "".join(p.capitalize() for p in parts)


def _mnemonic_from_name(op_name: str) -> str:
    """Extract mnemonic from qualified name: 'layout.set_layout' → 'set_layout'."""
    if "." in op_name:
        return op_name.rsplit(".", 1)[1]
    return op_name


# ---------------------------------------------------------------------------
# Attribute introspection (uses get_irdl_definition().parameters)
# ---------------------------------------------------------------------------


def introspect_attr(attr_cls: type, dialect_prefix: str) -> AttrInfo:
    """Introspect a ParametrizedAttribute subclass."""
    mlir_name = getattr(attr_cls, "name", "")
    mnemonic = mlir_name.rsplit(".", 1)[-1] if "." in mlir_name else mlir_name

    fields: list[AttrFieldInfo] = []

    # Use xDSL's IRDL definition API
    irdl_def = attr_cls.get_irdl_definition()
    for param_name, param_def in irdl_def.parameters:
        # Extract the constraint type (BaseAttr wrapping the actual type)
        xdsl_type = _extract_constraint_type_name(param_def.constr)
        tg_type = _resolve_tablegen_type(xdsl_type)
        fields.append(
            AttrFieldInfo(
                name=param_name,
                xdsl_type=xdsl_type,
                tablegen_type=tg_type,
            )
        )

    cpp_class = f"{dialect_prefix}_{attr_cls.__name__}"
    register_custom_attr(attr_cls.__name__, dialect_prefix)

    return AttrInfo(
        class_name=attr_cls.__name__,
        mlir_mnemonic=mnemonic,
        dialect_prefix=dialect_prefix,
        cpp_class=cpp_class,
        fields=fields,
    )


def _extract_constraint_type_name(constr: Any) -> str:
    """Extract the type name from an xDSL IRDL constraint (e.g. BaseAttr)."""
    # BaseAttr has an .attr field pointing to the actual class
    if hasattr(constr, "attr"):
        return constr.attr.__name__
    # AnyOf, AllOf, etc. — fall back to string
    return str(constr)


# ---------------------------------------------------------------------------
# Operation introspection (uses get_irdl_definition().properties)
# ---------------------------------------------------------------------------


def introspect_op(op_cls: type, dialect_prefix: str) -> OpInfo:
    """Introspect an IRDLOperation subclass."""
    op_name = getattr(op_cls, "name", "")
    mnemonic = _mnemonic_from_name(op_name)

    properties = _extract_properties(op_cls)
    has_region = _has_region(op_cls)
    traits = _extract_traits(op_cls)
    verifier = _extract_verifier(op_cls)
    summary = (op_cls.__doc__ or "").strip().split("\n")[0]

    return OpInfo(
        class_name=op_cls.__name__,
        mnemonic=mnemonic,
        properties=properties,
        has_region=has_region,
        traits=traits,
        verifier=verifier,
        summary=summary,
    )


def _extract_properties(op_cls: type) -> list[PropInfo]:
    """Extract properties from the IRDL definition."""
    props: list[PropInfo] = []

    irdl_def = op_cls.get_irdl_definition()
    for prop_name, prop_def in irdl_def.properties.items():
        type_name = type(prop_def).__name__
        is_optional = "Opt" in type_name

        xdsl_type = _extract_constraint_type_name(prop_def.constr)
        tablegen_type = _resolve_tablegen_type(xdsl_type)

        props.append(
            PropInfo(
                name=prop_name,
                xdsl_type=xdsl_type,
                tablegen_type=tablegen_type,
                is_optional=is_optional,
            )
        )

    return props


def _has_region(op_cls: type) -> bool:
    """Check if the op has regions."""
    irdl_def = op_cls.get_irdl_definition()
    return bool(irdl_def.regions)


def _extract_traits(op_cls: type) -> list[str]:
    """Extract trait names from the op class."""
    traits_obj = getattr(op_cls, "traits", None)
    if traits_obj is None:
        return []

    # xDSL OpTraits is iterable
    trait_names = []
    try:
        traits_set = traits_obj.traits if hasattr(traits_obj, "traits") else traits_obj
        if isinstance(traits_set, (set, frozenset)):
            for trait in traits_set:
                name = type(trait).__name__
                if name == "Pure":
                    trait_names.append("Pure")
                elif "SymbolOp" in name:
                    trait_names.append("Symbol")
                elif "NoTerminator" in name:
                    trait_names.append("NoTerminator")
                else:
                    trait_names.append(name)
    except (TypeError, AttributeError):
        pass

    return sorted(trait_names)


def _extract_verifier(op_cls: type) -> VerifierInfo | None:
    """Extract verifier metadata from the op class.

    Detects common patterns:
    1. Enum check: ``self.X.data not in self._VALID_*``
    2. Range check: ``self.X.value.data not in (0, 1 ...)``
    3. Dimension check: ``len(dims) < N``
    """
    # The @irdl_op_definition decorator adds verify_ to ALL ops.
    # Check if the original class SOURCE contains a user-defined verify_.
    try:
        cls_source = inspect.getsource(op_cls)
        if "    def verify_(self)" not in cls_source:
            return None
    except (OSError, TypeError):
        return None

    # Get the verify_ method source (from the class source)
    try:
        verify_source = _get_verify_source(cls_source)
    except (OSError, TypeError):
        verify_source = ""

    # Look for _VALID_* class variables (enum check pattern)
    for attr_name in dir(op_cls):
        if attr_name.startswith("_VALID_"):
            valid_set = getattr(op_cls, attr_name)
            if isinstance(valid_set, (set, frozenset)):
                prop_name = _find_validated_property(verify_source)
                return VerifierInfo(
                    kind="enum_check",
                    property_name=prop_name or "unknown",
                    valid_values=frozenset(str(v) for v in valid_set),
                )

    source = verify_source

    if "len(dims)" in source or "len(" in source:
        prop = _find_validated_property(source)
        return VerifierInfo(
            kind="dimension_check",
            property_name=prop or "shape",
            min_dims=2,
        )

    if "not in (0, 1" in source or "not in {0, 1" in source:
        prop = _find_validated_property(source)
        return VerifierInfo(
            kind="range_check",
            property_name=prop or "unknown",
            valid_values=frozenset({"0", "1"}),
        )

    return VerifierInfo(kind="custom", property_name="")


def _get_verify_source(cls_source: str) -> str:
    """Extract just the verify_ method body from class source."""
    import re

    match = re.search(r"(    def verify_\(self\).*?)(?=\n    \w|\n\n__|\Z)", cls_source, re.DOTALL)
    return match.group(1) if match else ""


def _find_validated_property(source: str) -> str | None:
    """Heuristic: find property name from ``self.X.data`` in verify_ source."""
    import re

    matches = re.findall(r"self\.(\w+)\.data", source)
    for m in matches:
        if not m.startswith("_"):
            return m
    # Fallback: any self.X. reference
    matches = re.findall(r"self\.(\w+)\.", source)
    for m in matches:
        if not m.startswith("_"):
            return m
    return None


# ---------------------------------------------------------------------------
# Dialect introspection
# ---------------------------------------------------------------------------


def introspect_dialect(
    dialect: Dialect,
    *,
    attr_classes: list[type] | None = None,
    op_classes: list[type] | None = None,
) -> DialectInfo:
    """Introspect an xDSL Dialect into a DialectInfo.

    Args:
        dialect: The xDSL Dialect object.
        attr_classes: Explicit list of attr classes.
        op_classes: Explicit list of op classes.

    Returns:
        DialectInfo ready for code generation.
    """
    name = dialect.name
    prefix = _dialect_prefix(name)
    cpp_ns = "compgen::" + name.replace(".", "_")

    info = DialectInfo(
        name=name,
        cpp_namespace=cpp_ns,
        prefix=prefix,
    )

    # Introspect attributes first (so ops can reference them)
    for attr_cls in attr_classes or []:
        info.attrs.append(introspect_attr(attr_cls, prefix))

    # Introspect operations
    ops = op_classes or []
    if not ops:
        for op_cls in getattr(dialect, "_ops", []):
            if isinstance(op_cls, type) and issubclass(op_cls, IRDLOperation):
                ops.append(op_cls)

    for op_cls in ops:
        info.ops.append(introspect_op(op_cls, prefix))

    return info


# ---------------------------------------------------------------------------
# Convenience: introspect well-known CompGen dialects
# ---------------------------------------------------------------------------


def _ensure_recipe_base_registered() -> None:
    """Ensure RecipeBase shared attrs are registered for cross-dialect references."""
    if "ProvenanceAttr" not in _custom_attr_registry:
        introspect_recipe_base()


def introspect_layout_dialect() -> DialectInfo:
    """Introspect the Layout IR dialect."""
    _ensure_recipe_base_registered()

    from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr
    from compgen.ir.layout.dialect import Layout
    from compgen.ir.layout.ops import PackOp, SetLayoutOp, UnpackOp, UnsetLayoutOp

    return introspect_dialect(
        Layout,
        attr_classes=[LayoutEncodingAttr, PackSpecAttr],
        op_classes=[SetLayoutOp, UnsetLayoutOp, PackOp, UnpackOp],
    )


def introspect_tile_dialect() -> DialectInfo:
    """Introspect the Tile IR dialect."""
    _ensure_recipe_base_registered()

    from compgen.ir.tile.attrs import FragmentLayoutAttr, MemoryClassAttr, TileShapeAttr
    from compgen.ir.tile.dialect import Tile
    from compgen.ir.tile.ops import (
        TileAsyncCopyOp,
        TileBarrierOp,
        TileElementwiseOp,
        TileLoadOp,
        TileMMAOp,
        TileReduceOp,
        TileStoreOp,
    )

    return introspect_dialect(
        Tile,
        attr_classes=[MemoryClassAttr, FragmentLayoutAttr, TileShapeAttr],
        op_classes=[
            TileLoadOp,
            TileStoreOp,
            TileMMAOp,
            TileElementwiseOp,
            TileReduceOp,
            TileBarrierOp,
            TileAsyncCopyOp,
        ],
    )


def introspect_accel_dialect() -> DialectInfo:
    """Introspect the Accel IR dialect."""
    from compgen.ir.accel.dialect import AccelDialect
    from compgen.ir.accel.ops import (
        AccelBarrierIROp,
        AccelDMAStartIROp,
        AccelDMAWaitIROp,
        AccelMatrixEngineIROp,
        AccelTileLoadIROp,
        AccelTileStoreIROp,
    )

    return introspect_dialect(
        AccelDialect,
        attr_classes=[],
        op_classes=[
            AccelTileLoadIROp,
            AccelTileStoreIROp,
            AccelDMAStartIROp,
            AccelDMAWaitIROp,
            AccelMatrixEngineIROp,
            AccelBarrierIROp,
        ],
    )


def introspect_recipe_base() -> DialectInfo:
    """Introspect shared Recipe IR attributes (ProvenanceAttr, DeviceRefAttr)."""
    from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr

    info = DialectInfo(
        name="recipe_base",
        cpp_namespace="compgen::recipe_base",
        prefix="RecipeBase",
    )
    info.attrs.append(introspect_attr(ProvenanceAttr, "RecipeBase"))
    info.attrs.append(introspect_attr(DeviceRefAttr, "RecipeBase"))
    return info


__all__ = [
    "AttrFieldInfo",
    "AttrInfo",
    "DialectInfo",
    "OpInfo",
    "PropInfo",
    "VerifierInfo",
    "introspect_accel_dialect",
    "introspect_attr",
    "introspect_dialect",
    "introspect_layout_dialect",
    "introspect_op",
    "introspect_recipe_base",
    "introspect_tile_dialect",
]
