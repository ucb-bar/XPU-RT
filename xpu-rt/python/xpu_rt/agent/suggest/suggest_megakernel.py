"""Suggester for ``propose_megakernel_synthesis``.

Strategy: cluster recipe regions whose roles indicate one of the two
canonical mega-blocks in transformer-style models:

  * Attention block: q_proj/k_proj/v_proj (matmuls) + softmax +
    o_proj (matmul). Detected by the presence of >=2 matmul regions
    surrounding a softmax in declaration order.
  * MLP block: gate / up (matmuls) + silu/gelu + mul + down (matmul).
    Detected by a matmul-activation-mul-matmul pattern.

One candidate per cluster found, plus a fallback "all matmul regions"
cluster when the heuristics miss.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.suggest._candidate import ProposalCandidate
from compgen.agent.suggest._dispatch import register_suggester
from compgen.agent.suggest._recipe_index import build_recipe_index


_MATMUL_ROLES = frozenset({
    "matmul", "mm", "addmm", "linear", "bmm", "batch_matmul",
    "_weight_int4pack_mm", "_weight_int8pack_mm",
})
_ACTIVATION_ROLES = frozenset({
    "silu", "gelu", "sigmoid", "tanh", "relu",
})

# Roles considered "semantic" for megakernel membership. Structural
# noise (yield / empty / view / transpose / permute / cat / clone /
# unsqueeze / expand / fact ops) is excluded so the megakernel
# fused_region_refs list stays tight and meaningful — bundling 29
# regions including yield/empty makes the candidate look bloated and
# lowers structural-gate pass odds.
_SEMANTIC_ROLES: frozenset[str] = (
    _MATMUL_ROLES
    | _ACTIVATION_ROLES
    | frozenset({
        "softmax", "_softmax", "rmsnorm", "layer_norm",
        "native_layer_norm", "rsqrt",
        "mul", "add", "sub", "div", "neg",
    })
)


def _filter_semantic(idx, syms: list[str]) -> list[str]:
    """Drop structural-noise regions; keep only roles that carry compute."""
    out: list[str] = []
    for s in syms:
        role = idx.role_by_region.get(s, "")
        if role in _SEMANTIC_ROLES:
            out.append(s)
    return out


def _attention_window(idx) -> list[str] | None:
    """Find the first matmul...softmax...matmul triplet; return ONLY
    the semantic-role members of that window (not the structural
    intermediates the seed walks past)."""
    found_matmul = -1
    softmax_at = -1
    after_matmul = -1
    for i, sym in enumerate(idx.regions):
        role = idx.role_by_region.get(sym, "")
        if role in _MATMUL_ROLES and found_matmul < 0:
            found_matmul = i
        elif role.startswith("softmax") and found_matmul >= 0:
            softmax_at = i
        elif role in _MATMUL_ROLES and softmax_at > 0:
            after_matmul = i
            break
    if found_matmul >= 0 and softmax_at > 0 and after_matmul > 0:
        window = idx.regions[found_matmul : after_matmul + 1]
        return _filter_semantic(idx, window)
    return None


def _mlp_window(idx) -> list[str] | None:
    """Find a matmul → activation → mul → matmul window; return ONLY
    the semantic-role members."""
    n = len(idx.regions)
    for i in range(n - 3):
        slice_ = idx.regions[i : i + 4]
        roles = [idx.role_by_region.get(s, "") for s in slice_]
        if (roles[0] in _MATMUL_ROLES
                and roles[1] in _ACTIVATION_ROLES
                and roles[2] == "mul"
                and roles[3] in _MATMUL_ROLES):
            # The tight 4-op slice is already semantic — but defensively
            # filter so future slack windows still emit clean lists.
            return _filter_semantic(idx, list(slice_))
    return None


@register_suggester("propose_megakernel_synthesis")
def suggest_megakernel(
    *,
    recipe: ModuleOp,
    dossier: Any,
    target: Any,
    k: int = 5,
) -> list[ProposalCandidate]:
    idx = build_recipe_index(recipe)
    out: list[ProposalCandidate] = []

    target_name = getattr(target, "name", "agent_target")

    attn = _attention_window(idx)
    if attn:
        out.append(ProposalCandidate(
            chosen={
                "megakernel_name": f"{target_name}_attention_block",
                "fused_region_refs": attn,
                "event_tensor_decls": [
                    {"name": "scores_done", "wait_count": len(attn) - 1,
                     "scope": "block"},
                ],
            },
            rationale=(
                f"Attention-block megakernel: matmul→softmax→matmul "
                f"({len(attn)} regions: {attn})"
            ),
            expected_impact=0.85,
            target_feature_justification=(
                "persistent_kernels + semaphore_atomics on the attention "
                "block keep Q·K·V intermediates resident in scratchpad."
            ),
            metadata={"window": "attention"},
        ))

    mlp = _mlp_window(idx)
    if mlp:
        out.append(ProposalCandidate(
            chosen={
                "megakernel_name": f"{target_name}_mlp_block",
                "fused_region_refs": mlp,
                "event_tensor_decls": [
                    {"name": "gate_done", "wait_count": 1, "scope": "block"},
                    {"name": "up_done", "wait_count": 1, "scope": "block"},
                ],
            },
            rationale=(
                f"MLP-block megakernel: matmul→{idx.role_by_region.get(mlp[1])}→mul→matmul "
                f"({len(mlp)} regions: {mlp})"
            ),
            expected_impact=0.8,
            target_feature_justification=(
                "SwiGLU-style MLP collapses into one persistent kernel; "
                "intermediates never leave VTCM/SMEM."
            ),
            metadata={"window": "mlp"},
        ))

    # Fallback: cluster all matmuls (when neither attention nor MLP
    # window matched but matmuls exist).
    matmuls = [s for s in idx.regions
               if idx.role_by_region.get(s, "") in _MATMUL_ROLES]
    if not out and len(matmuls) >= 2:
        out.append(ProposalCandidate(
            chosen={
                "megakernel_name": f"{target_name}_all_matmuls",
                "fused_region_refs": matmuls[:8],   # cap so the recipe stays sane
            },
            rationale=(
                f"All-matmul cluster (no attention/MLP pattern found); "
                f"{len(matmuls)} regions"
            ),
            expected_impact=0.5,
            target_feature_justification="matmul-heavy region grouping",
            metadata={"window": "all_matmuls"},
        ))

    return out[:k]


__all__ = ["suggest_megakernel"]
