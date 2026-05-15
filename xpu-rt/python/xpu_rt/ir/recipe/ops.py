"""Recipe IR operations.

Each op represents a single optimization decision the LLM can make.
Ops are frozen dataclasses -- compact, canonical, easy to generate,
easy to diff, easy to replay.

Invariants:
    - Every op has a ``region`` field identifying what it applies to.
    - Ops are order-independent unless explicitly sequenced.
    - Unknown or invalid ops are rejected by ``validate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.llm.base import Objective


@dataclass(frozen=True)
class MatchRegion:
    """Select a region in the Payload IR for optimization.

    Attributes:
        region_id: Identifier for the IR region (op name, subgraph hash, etc.).
        op_filter: Optional op-type filter (e.g., "linalg.matmul").
        metadata: Additional selection criteria.
    """

    region_id: str
    op_filter: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SetObjective:
    """Set the optimization objective for a recipe.

    Attributes:
        objective: The optimization objective.
        weight: Priority weight (for multi-objective).
    """

    objective: Objective
    weight: float = 1.0


@dataclass(frozen=True)
class ChooseTransformFamily:
    """Choose a transform family to apply to a matched region.

    Attributes:
        region_id: Region to transform.
        family: Transform family name (e.g., "tile", "fuse", "vectorize", "interchange").
        priority: Selection priority among competing families.
    """

    region_id: str
    family: str
    priority: int = 0


@dataclass(frozen=True)
class SetTileParams:
    """Set tiling parameters for a region.

    Attributes:
        region_id: Region to tile.
        tile_sizes: Tile sizes per dimension.
        interchange: Optional dimension interchange order.
    """

    region_id: str
    tile_sizes: tuple[int, ...] = ()
    interchange: tuple[int, ...] | None = None


@dataclass(frozen=True)
class AssignDevice:
    """Assign a region to a specific device.

    Attributes:
        region_id: Region to place.
        device_index: Index into the target profile's device list.
        reason: Placement rationale (for audit).
    """

    region_id: str
    device_index: int
    reason: str = ""


@dataclass(frozen=True)
class InsertCopyBoundary:
    """Insert a copy boundary between two regions on different devices.

    Attributes:
        src_region: Source region ID.
        dst_region: Destination region ID.
        tensor_name: Name of the tensor to copy.
        async_: Whether the copy can be asynchronous.
    """

    src_region: str
    dst_region: str
    tensor_name: str
    async_: bool = True


@dataclass(frozen=True)
class RequestKernelSearch:
    """Request a kernel search for a region via Autocomp/Triton.

    Attributes:
        region_id: Region needing a custom kernel.
        backend: Preferred kernel backend ("triton", "autocomp", "vendor").
        search_budget: Max iterations for the search.
        constraints: Additional search constraints.
    """

    region_id: str
    backend: str = "autocomp"
    search_budget: int = 50
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequireCheck:
    """Attach a verification obligation to a recipe step.

    Attributes:
        region_id: Region the check applies to.
        check_type: Type of check ("structural", "differential", "translation_validation", "peephole").
        tolerance: Tolerance for numerical checks.
    """

    region_id: str
    check_type: str = "differential"
    tolerance: float = 1e-5


@dataclass(frozen=True)
class PromoteIfVerified:
    """Mark a recipe for promotion if all verification obligations pass.

    Attributes:
        recipe_name: Name for the promoted recipe.
        require_all_checks: Whether ALL RequireCheck obligations must pass.
    """

    recipe_name: str
    require_all_checks: bool = True


# Union of all Recipe IR ops
RecipeOp = (
    MatchRegion
    | SetObjective
    | ChooseTransformFamily
    | SetTileParams
    | AssignDevice
    | InsertCopyBoundary
    | RequestKernelSearch
    | RequireCheck
    | PromoteIfVerified
)

__all__ = [
    "AssignDevice",
    "ChooseTransformFamily",
    "InsertCopyBoundary",
    "MatchRegion",
    "PromoteIfVerified",
    "RecipeOp",
    "RequestKernelSearch",
    "RequireCheck",
    "SetObjective",
    "SetTileParams",
]
