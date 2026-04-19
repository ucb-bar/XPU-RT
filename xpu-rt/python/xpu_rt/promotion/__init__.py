"""Recipe promotion and caching subpackage.

Manages the transition from LLM-generated candidates to promoted,
deterministic recipes in the recipe library. Key principle:

    LLM proposes -> verification filters -> promotion hardens

Only verified artifacts get promoted. Once promoted, a recipe is
deterministic and does not require LLM involvement to use.

Recipes are keyed by: hash(target_profile) + hash(model_ir) + hash(objective)
"""

from __future__ import annotations

from compgen.promotion.lineage import (
    LineageGraph,
    LineageNode,
    build_lineage_graph,
    find_lineage_siblings,
    get_promotion_history,
)
from compgen.promotion.pattern_graduation import (
    PatternAppearance,
    PatternIdentity,
    PatternPromotionRequest,
    build_promotion_requests,
    graduate_from_transcripts,
    scan_transcripts,
)

__all__: list[str] = [
    "LineageGraph",
    "LineageNode",
    "PatternAppearance",
    "PatternIdentity",
    "PatternPromotionRequest",
    "build_lineage_graph",
    "build_promotion_requests",
    "find_lineage_siblings",
    "get_promotion_history",
    "graduate_from_transcripts",
    "scan_transcripts",
]
