"""Layout planner -- deterministic layout assignment for network regions.

Given a NetworkAnalysis and a TargetProfile, the planner assigns preferred
operand/output layouts, identifies prepack and transpose-absorption
candidates, and optionally selects a tile encoding.  All decisions are
deterministic (no LLM calls).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from compgen.agent.analyzer import NetworkAnalysis, PatternCluster, RegionDossier
from compgen.targets.schema import TargetProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern-type keywords used for layout heuristics
# ---------------------------------------------------------------------------
_MATMUL_KEYWORDS = ("matmul", "linear", "addmm", "mlp", "gemm")
_ELEMENTWISE_KEYWORDS = ("elementwise", "relu", "gelu", "add", "mul", "sigmoid", "tanh")
_ATTENTION_KEYWORDS = ("attention", "sdpa", "gqa", "mha")
_TRANSPOSE_KEYWORDS = ("transpose", "permute")

# ---------------------------------------------------------------------------
# Tile encoding defaults (target-dependent)
# ---------------------------------------------------------------------------
_DEFAULT_TILE_ENCODING = "blocked_64x64x32"
_TENSOR_CORE_TILE_ENCODING = "blocked_128x128x32"


def _has_tensor_cores(target: TargetProfile) -> bool:
    """Check whether any device in the target advertises tensor cores."""
    for device in target.devices:
        for feature in device.features:
            if "tensor_core" in feature.lower():
                return True
        for cu in device.compute_units:
            if "tensor_core" in cu.name.lower():
                return True
    return False


def _matches_any(value: str, keywords: tuple[str, ...]) -> bool:
    """Return True if *value* contains any of the given keywords."""
    lower = value.lower()
    return any(kw in lower for kw in keywords)


# ---------------------------------------------------------------------------
# LayoutPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutPlan:
    """Recommended layout configuration for a single region.

    Attributes:
        region_id: Identifier of the region this plan applies to.
        preferred_operand_layouts: Mapping of operand name to layout string.
        preferred_output_layout: Recommended output layout string.
        prepack_candidates: Operand names worth prepacking.
        transpose_absorption_candidates: Ops that can absorb transposes.
        tile_encoding: Tile encoding string if applicable.
        notes: Planning notes and rationale.
    """

    region_id: str
    preferred_operand_layouts: dict[str, str] = field(default_factory=dict)
    preferred_output_layout: str = "row_major"
    prepack_candidates: list[str] = field(default_factory=list)
    transpose_absorption_candidates: list[str] = field(default_factory=list)
    tile_encoding: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LayoutPlanner
# ---------------------------------------------------------------------------


class LayoutPlanner:
    """Assigns preferred layouts to every region in a NetworkAnalysis.

    The planner is fully deterministic.  It inspects region kinds, layout
    candidates from the dossier, and target device features to decide on
    operand/output layouts, prepack candidates, and tile encodings.
    """

    def plan(
        self,
        analysis: NetworkAnalysis,
        target: TargetProfile,
    ) -> dict[str, LayoutPlan]:
        """Produce a layout plan for each region.

        Args:
            analysis: The network analysis produced by ``NetworkAnalyzer``.
            target: The deployment target profile.

        Returns:
            Dict mapping region_id to its ``LayoutPlan``.
        """
        tensor_cores = _has_tensor_cores(target)
        plans: dict[str, LayoutPlan] = {}

        if analysis.dossier is not None and analysis.dossier.regions:
            for region in analysis.dossier.regions:
                plans[region.region_id] = self._plan_from_dossier(
                    region,
                    tensor_cores,
                )
        else:
            for cluster in analysis.clusters:
                plans[cluster.cluster_id] = self._plan_from_cluster(
                    cluster,
                    tensor_cores,
                )

        logger.debug("LayoutPlanner produced %d plans", len(plans))
        return plans

    # -- dossier-based planning ---------------------------------------------

    def _plan_from_dossier(
        self,
        region: RegionDossier,
        tensor_cores: bool,
    ) -> LayoutPlan:
        """Derive a LayoutPlan from a RegionDossier entry."""
        kind = region.kind
        layout_candidates = region.layout_candidates
        notes: list[str] = []

        # Determine preferred layouts per operand and output
        operand_layouts: dict[str, str] = {}
        output_layout = "row_major"
        prepack: list[str] = []
        absorb: list[str] = []
        tile: str | None = None

        if _matches_any(kind, _MATMUL_KEYWORDS):
            output_layout, tile = self._matmul_layouts(
                layout_candidates,
                tensor_cores,
                notes,
            )
            # Mark RHS (second operand) as a prepack candidate
            if len(region.node_names) >= 2:
                rhs_name = region.node_names[1]
                prepack.append(rhs_name)
                notes.append(f"RHS operand '{rhs_name}' marked as prepack candidate")
            for name in region.node_names:
                operand_layouts[name] = output_layout

        elif _matches_any(kind, _ATTENTION_KEYWORDS):
            output_layout, tile = self._attention_layouts(
                layout_candidates,
                tensor_cores,
                notes,
            )
            # QK matmul benefits from tiled; softmax is row-major
            for idx, name in enumerate(region.node_names):
                lower = name.lower()
                if "softmax" in lower or "exp" in lower:
                    operand_layouts[name] = "row_major"
                else:
                    operand_layouts[name] = output_layout

        elif _matches_any(kind, _ELEMENTWISE_KEYWORDS):
            output_layout = "row_major"
            for name in region.node_names:
                operand_layouts[name] = "row_major"
            notes.append("Elementwise region: row_major preferred")

        elif _matches_any(kind, _TRANSPOSE_KEYWORDS):
            output_layout = "row_major"
            for name in region.node_names:
                operand_layouts[name] = "row_major"
                absorb.append(name)
            notes.append("Transpose region: candidates for absorption")

        else:
            # Fallback: honour first layout candidate if available
            if layout_candidates:
                output_layout = layout_candidates[0]
                notes.append(f"Fallback to first layout candidate: {output_layout}")
            for name in region.node_names:
                operand_layouts[name] = output_layout

        return LayoutPlan(
            region_id=region.region_id,
            preferred_operand_layouts=operand_layouts,
            preferred_output_layout=output_layout,
            prepack_candidates=prepack,
            transpose_absorption_candidates=absorb,
            tile_encoding=tile,
            notes=notes,
        )

    # -- cluster-based planning (fallback when dossier is absent) -----------

    def _plan_from_cluster(
        self,
        cluster: PatternCluster,
        tensor_cores: bool,
    ) -> LayoutPlan:
        """Derive a LayoutPlan from a PatternCluster (no dossier)."""
        kind = cluster.pattern_type
        notes: list[str] = ["Derived from PatternCluster (no dossier)"]

        operand_layouts: dict[str, str] = {}
        output_layout = "row_major"
        prepack: list[str] = []
        absorb: list[str] = []
        tile: str | None = None

        if _matches_any(kind, _MATMUL_KEYWORDS):
            output_layout, tile = self._matmul_layouts((), tensor_cores, notes)
            if len(cluster.node_names) >= 2:
                rhs_name = cluster.node_names[1]
                prepack.append(rhs_name)
                notes.append(f"RHS operand '{rhs_name}' marked as prepack candidate")
            for name in cluster.node_names:
                operand_layouts[name] = output_layout

        elif _matches_any(kind, _ATTENTION_KEYWORDS):
            output_layout, tile = self._attention_layouts((), tensor_cores, notes)
            for name in cluster.node_names:
                lower = name.lower()
                if "softmax" in lower or "exp" in lower:
                    operand_layouts[name] = "row_major"
                else:
                    operand_layouts[name] = output_layout

        elif _matches_any(kind, _ELEMENTWISE_KEYWORDS):
            output_layout = "row_major"
            for name in cluster.node_names:
                operand_layouts[name] = "row_major"
            notes.append("Elementwise cluster: row_major preferred")

        elif _matches_any(kind, _TRANSPOSE_KEYWORDS):
            output_layout = "row_major"
            for name in cluster.node_names:
                operand_layouts[name] = "row_major"
                absorb.append(name)
            notes.append("Transpose cluster: candidates for absorption")

        else:
            for name in cluster.node_names:
                operand_layouts[name] = output_layout
            notes.append("Unrecognized pattern type: defaulting to row_major")

        return LayoutPlan(
            region_id=cluster.cluster_id,
            preferred_operand_layouts=operand_layouts,
            preferred_output_layout=output_layout,
            prepack_candidates=prepack,
            transpose_absorption_candidates=absorb,
            tile_encoding=tile,
            notes=notes,
        )

    # -- shared layout helpers -----------------------------------------------

    def _matmul_layouts(
        self,
        layout_candidates: tuple[str, ...],
        tensor_cores: bool,
        notes: list[str],
    ) -> tuple[str, str | None]:
        """Return (preferred_layout, tile_encoding) for matmul-like regions."""
        if tensor_cores:
            tile = _TENSOR_CORE_TILE_ENCODING
            layout = "tiled"
            notes.append(f"Tensor cores detected: tiled layout with {tile}")
        elif layout_candidates:
            layout = layout_candidates[0]
            tile = _DEFAULT_TILE_ENCODING
            notes.append(f"Matmul region: using candidate layout '{layout}'")
        else:
            layout = "tiled"
            tile = _DEFAULT_TILE_ENCODING
            notes.append("Matmul region: default tiled layout")
        return layout, tile

    def _attention_layouts(
        self,
        layout_candidates: tuple[str, ...],
        tensor_cores: bool,
        notes: list[str],
    ) -> tuple[str, str | None]:
        """Return (preferred_layout, tile_encoding) for attention-like regions."""
        if tensor_cores:
            tile = _TENSOR_CORE_TILE_ENCODING
            layout = "tiled"
            notes.append(f"Tensor cores detected: tiled attention layout with {tile}")
        elif layout_candidates:
            layout = layout_candidates[0]
            tile = _DEFAULT_TILE_ENCODING
            notes.append(f"Attention region: using candidate layout '{layout}'")
        else:
            layout = "tiled"
            tile = _DEFAULT_TILE_ENCODING
            notes.append("Attention region: default tiled layout for QK matmul")
        return layout, tile


__all__ = ["LayoutPlan", "LayoutPlanner"]
