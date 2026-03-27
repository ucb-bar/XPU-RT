"""Kernel gap analysis and generation subpackage (Stages 2 and 4).

Handles:
- Stage 2: Kernel gap analysis -- determine which ops need custom kernels
  vs native lowering, library calls, or fallback.
- Stage 4: Kernel generation via Autocomp -- run LLM-driven search loops
  to produce missing kernels.

The critical integration point with autocomp is at ``autocomp_adapter.py``,
which translates CompGen's kernel contracts into autocomp's problem format
and runs the beam search.
"""

from __future__ import annotations

__all__: list[str] = []
