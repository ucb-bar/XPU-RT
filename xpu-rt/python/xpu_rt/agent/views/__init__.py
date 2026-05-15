"""Token-economical state views over the agent compilation session (P2.1).

Three pure-function projections that shrink the LLM's reasoning
surface to the smallest payload that still answers the question:

* :func:`canonical_view.canonical_view` — ≤1 KB per region; open
  decision sites, last verdicts, current plan rung.
* :func:`focus_chunk.focus_chunk` — lazy-expand one region's full
  dossier on demand.
* :func:`diff_since.diff_since` — delta between two driver
  checkpoints; what changed since the last plan?

The views accept ``dict``-shaped session state (mirroring
:class:`compgen.agent.llm_driver.DriverCheckpoint`'s
``to_dict`` output) so they are runnable + testable without
materialising a full LLMDrivenCompiler. The wire-in to the live
driver lands in P2.4 + P2.5.

Hard rules:

* Views are *pure* — same input → same output. No I/O, no global
  state, no time-dependent fields in the output.
* Views are bounded — :func:`canonical_view` enforces an output
  size cap (in bytes of canonical JSON) so a runaway session state
  can't blow up an LLM context.
"""

from __future__ import annotations

from compgen.agent.views.canonical_view import (
    CANONICAL_VIEW_BYTE_BUDGET,
    CanonicalView,
    CanonicalViewBudgetError,
    canonical_view,
)
from compgen.agent.views.diff_since import DiffEntry, diff_since
from compgen.agent.views.focus_chunk import FocusChunk, focus_chunk

__all__ = [
    "CANONICAL_VIEW_BYTE_BUDGET",
    "CanonicalView",
    "CanonicalViewBudgetError",
    "DiffEntry",
    "FocusChunk",
    "canonical_view",
    "diff_since",
    "focus_chunk",
]
