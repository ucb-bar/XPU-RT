"""Guard-family specifications used by the synthesis layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.ir import Operation

from compgen.ir.recipe.ops_candidate import FuseOp, PlaceOnDeviceOp, TileOp, VectorizeOp

FUSION_FAMILY = "fusion"
LOCAL_MEM_FAMILY = "local_mem"


@dataclass(frozen=True)
class GuardFamilySpec:
    """Description of a guard family supported by the synthesis layer."""

    family: str
    guard_kind: str
    experimental: bool = False

    def matches_candidate(self, op: Operation) -> bool:
        return False

    def label(self, env: dict[str, Any]) -> tuple[bool, bool]:
        return False, False


@dataclass(frozen=True)
class FusionGuardSpec(GuardFamilySpec):
    family: str = FUSION_FAMILY
    guard_kind: str = "legality"
    experimental: bool = False

    def matches_candidate(self, op: Operation) -> bool:
        return isinstance(op, FuseOp)

    def label(self, env: dict[str, Any]) -> tuple[bool, bool]:
        safe = bool(
            env.get("fusible")
            and env.get("graph_break_free", True)
            and env.get("export_issue_free", True)
            and env.get("fusion_region_count", 0) >= 2
        )
        profitable = bool(
            safe
            and (
                env.get("backend_triton")
                or env.get("target_is_accel_native")
                or env.get("target_is_ukernel_runtime")
            )
            and env.get("estimated_flops", 0) > 0
        )
        return safe, profitable


@dataclass(frozen=True)
class LocalMemGuardSpec(GuardFamilySpec):
    family: str = LOCAL_MEM_FAMILY
    guard_kind: str = "placement"
    experimental: bool = False

    def matches_candidate(self, op: Operation) -> bool:
        return isinstance(op, (TileOp, PlaceOnDeviceOp))

    def label(self, env: dict[str, Any]) -> tuple[bool, bool]:
        safe = bool(
            env.get("graph_break_free", True)
            and env.get("export_issue_free", True)
            and env.get("local_mem_fit", False)
        )
        profitable = bool(
            safe
            and (
                env.get("target_is_accel_native")
                or env.get("target_is_ukernel_runtime")
                or env.get("backend_triton")
            )
            and env.get("estimated_flops", 0) > 0
        )
        return safe, profitable


@dataclass(frozen=True)
class VectorizationGuardSpec(GuardFamilySpec):
    family: str = "vectorization"
    guard_kind: str = "legality"
    experimental: bool = True

    def matches_candidate(self, op: Operation) -> bool:
        return isinstance(op, VectorizeOp)


@dataclass(frozen=True)
class RangeNoWrapSpec(GuardFamilySpec):
    family: str = "range"
    guard_kind: str = "analysis"
    experimental: bool = True


@dataclass(frozen=True)
class QuantizationLegalitySpec(GuardFamilySpec):
    family: str = "quantization"
    guard_kind: str = "legality"
    experimental: bool = True


class FusionSoundnessSpec:
    """Sufficient-condition SMT spec for guarded fusion."""

    def build_vars(self) -> dict[str, Any]:
        import z3

        return {
            "fusible": z3.Bool("fusible"),
            "graph_break_free": z3.Bool("graph_break_free"),
            "export_issue_free": z3.Bool("export_issue_free"),
            "fusion_region_count": z3.Int("fusion_region_count"),
        }

    def sound_formula(self, vars: dict[str, Any]) -> Any:
        import z3

        return z3.And(
            vars["fusible"],
            vars["graph_break_free"],
            vars["export_issue_free"],
            vars["fusion_region_count"] >= 2,
        )


class LocalMemSoundnessSpec:
    """Sufficient-condition SMT spec for guarded local-memory decisions."""

    def build_vars(self) -> dict[str, Any]:
        import z3

        return {
            "local_mem_fit": z3.Bool("local_mem_fit"),
            "graph_break_free": z3.Bool("graph_break_free"),
            "export_issue_free": z3.Bool("export_issue_free"),
        }

    def sound_formula(self, vars: dict[str, Any]) -> Any:
        import z3

        return z3.And(
            vars["local_mem_fit"],
            vars["graph_break_free"],
            vars["export_issue_free"],
        )


PROMOTED_FAMILIES: dict[str, GuardFamilySpec] = {
    FUSION_FAMILY: FusionGuardSpec(),
    LOCAL_MEM_FAMILY: LocalMemGuardSpec(),
}

EXPERIMENTAL_FAMILIES: dict[str, GuardFamilySpec] = {
    "vectorization": VectorizationGuardSpec(),
    "range": RangeNoWrapSpec(),
    "quantization": QuantizationLegalitySpec(),
}


def get_family_spec(family: str) -> GuardFamilySpec:
    if family in PROMOTED_FAMILIES:
        return PROMOTED_FAMILIES[family]
    if family in EXPERIMENTAL_FAMILIES:
        return EXPERIMENTAL_FAMILIES[family]
    raise KeyError(f"unknown guard family: {family}")


def get_soundness_spec(family: str) -> Any:
    if family == FUSION_FAMILY:
        return FusionSoundnessSpec()
    if family == LOCAL_MEM_FAMILY:
        return LocalMemSoundnessSpec()
    return None


__all__ = [
    "EXPERIMENTAL_FAMILIES",
    "FUSION_FAMILY",
    "FusionGuardSpec",
    "GuardFamilySpec",
    "LOCAL_MEM_FAMILY",
    "LocalMemGuardSpec",
    "PROMOTED_FAMILIES",
    "QuantizationLegalitySpec",
    "RangeNoWrapSpec",
    "FusionSoundnessSpec",
    "LocalMemSoundnessSpec",
    "VectorizationGuardSpec",
    "get_family_spec",
    "get_soundness_spec",
]
