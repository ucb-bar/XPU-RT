"""Vendor-dialect exploration and scaffolding agent.

This is the agent side of third-party MLIR integration. The module
:mod:`xpu_rt.extensions.vendor_dialect` holds the generic data types
and side-effects (scanner, descriptor, scaffold engine); this module
holds the LLM-driven orchestration that turns a repo path into a
reviewable :class:`VendorDialectDescriptor`.

Design contract:

* No MCP-specific glue lives here — those wrappers are in
  :mod:`xpu_rt.mcp.tools.vendor_dialect`.
* No filesystem work beyond reading prompt templates.
* All LLM calls go through ``autocomp.common.llm_utils.LLMClient``;
  a ``MockLLMClient`` shim is used by tests.
"""

from __future__ import annotations

from xpu_rt.agent.vendor_integration.explore import ExploreResult, explore_vendor_repo
from xpu_rt.agent.vendor_integration.propose_adapter import (
    ProposedAdapter,
    propose_adapter_layout,
)

__all__ = [
    "ExploreResult",
    "ProposedAdapter",
    "explore_vendor_repo",
    "propose_adapter_layout",
]
