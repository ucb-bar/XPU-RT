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
    llm_client: Any | None = None,
    target_name: str = "",
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

        # Fall back to LLM-guided layout planning (Unit 7)
        if pack_spec_str is None and llm_client is not None:
            try:
                from compgen.agent.prompts.layout_plan import LAYOUT_PLAN_SCHEMA, LayoutPlanContext
                from compgen.agent.prompts.layout_plan import format_prompt as fmt_lp
                from compgen.agent.prompts.layout_plan import parse_response as parse_lp
                from compgen.llm.base import GenerationRequest, LLMConfig

                tile_hint_str = ""
                if tile_hint and hasattr(tile_hint, "data"):
                    tile_hint_str = tile_hint.data

                ctx = LayoutPlanContext(
                    op_name=type(op).__name__,
                    encoding_str=encoding_str,
                    target_name=target_name,
                    capabilities_summary=str(capabilities) if capabilities else "unknown",
                    tile_family_hint=tile_hint_str,
                )
                prompt = fmt_lp(ctx)
                request = GenerationRequest(
                    prompt_template=prompt,
                    config=LLMConfig(temperature=0.1, max_tokens=600),
                )
                response = llm_client.generate_structured(request, LAYOUT_PLAN_SCHEMA)
                result = parse_lp(response.raw_text)
                if result and "inner_tiles" in result:
                    pack_spec_str = str(result["inner_tiles"])
            except Exception:
                pass

        if pack_spec_str:
            op.attributes[PACK_SPEC_ATTR] = StringAttr(pack_spec_str)

        op.attributes[SPECIALIZED_ATTR] = StringAttr(encoding_str)
        specialized += 1

    log.debug("layout.specialize_layouts", specialized=specialized)
    return module


__all__ = ["specialize_layouts"]
