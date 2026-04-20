"""Invent-slot baselines + gate wiring.

Each invent-slot gets:

1. A deterministic ``baseline_seed`` callable that produces a default
   proposal the LLM can refine or replace.
2. A composite gate (default: structural + differential; SMT-on-demand
   per the slot's metadata) bound as :attr:`InventSlot.gate_impl`.

Importing this package auto-registers the core invent-slots into
:func:`compgen.llm.registry.get_registry` so the LLM drive loop can
consume them directly.
"""

from __future__ import annotations

from compgen.agent.invent_slots.registrar import register_invent_slots

# Auto-register on import.
register_invent_slots()


__all__ = ["register_invent_slots"]
