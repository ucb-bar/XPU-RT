"""Per-layer LLM-leverage primitives (P3.1 → P3.7).

Each module in this package implements one named typed primitive
behind the :func:`compgen.llm.call_site.llm_call_site` decorator.
The seven primitives, in order:

* P3.1 ``recognize_python_pattern``
* P3.2 ``name_cluster``
* P3.3 ``rank_candidates``  (highest-leverage; landed first)
* P3.4 ``revise_kernel``
* P3.5 ``pick_dispatch``
* P3.6 ``explain_counterexample``
* P3.7 ``compare_recipes``

Every primitive has:

1. A *deterministic fallback* registered via ``@register_fallback`` so
   the test suite runs without an LLM (``COMPGEN_DISABLE_LLM=1`` is
   the CI default).
2. A *primary* that may call an LLM. The contract is identical to
   the fallback's; the decorator validates the output schema.
3. A test that asserts the primitive honors its declared *forbidden*
   actions.

The module-level import side effect (registering fallback + site)
fires when the primitive is imported. Tests + the bridge
introspect the registry through :mod:`compgen.llm.call_site`.
"""

from __future__ import annotations

# Import every shipped primitive so the registry is populated as soon
# as ``compgen.agent.primitives`` is imported. Order matches the
# P3 milestone numbering.
from compgen.agent.primitives import compare_recipes as _compare_recipes  # noqa: F401
from compgen.agent.primitives import explain_counterexample as _explain_cex  # noqa: F401
from compgen.agent.primitives import name_cluster as _name_cluster  # noqa: F401
from compgen.agent.primitives import pick_dispatch as _pick_dispatch  # noqa: F401
from compgen.agent.primitives import rank_candidates as _rank_candidates  # noqa: F401
from compgen.agent.primitives import recognize_python_pattern as _recognize  # noqa: F401
from compgen.agent.primitives import revise_kernel as _revise_kernel  # noqa: F401

__all__ = []
