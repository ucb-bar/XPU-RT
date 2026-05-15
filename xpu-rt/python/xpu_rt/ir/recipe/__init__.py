"""Recipe IR -- the LLM-facing control IR layer.

Layer 2 of the three-layer IR stack. Recipe IR encodes the compiler's
decision problem over the program: regions, facts, alternatives,
candidate actions, costs, proof obligations, and provenance.

The LLM edits Recipe IR, NOT raw Payload IR. This keeps the LLM's
outputs bounded, declarative, checkable, and searchable.

Recipe IR lowers into five concrete outputs:
    - Transform Dialect scripts (for Payload IR rewrites)
    - Kernel search jobs (for Autocomp/Triton)
    - Execution plan fragments (for the solver/planner)
    - Verification obligations (for the semantic layer)
    - EqSat job specifications (for the equality saturation pipeline)

The ``Recipe`` dialect object is the xDSL dialect registration
containing all 44 operations and 5 custom attributes.
"""

from __future__ import annotations

from compgen.ir.recipe.dialect import Recipe

__all__ = ["Recipe"]
