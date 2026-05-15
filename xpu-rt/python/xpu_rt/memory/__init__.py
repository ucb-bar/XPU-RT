"""Unified Compiler Memory System.

Three-layer persistence for all CompGen-generated artifacts:

- **Layer A**: Content-addressed blob store (immutable artifacts)
- **Layer B**: Relational metadata (SQLite, upgradeable to Postgres)
- **Layer C**: Analytics export (Parquet + DuckDB)

Four memory levels:

- **L0 Episode**: Per-run state (current candidates, traces)
- **L1 Replay**: Per-task trajectories (search history)
- **L2 Knowledge**: Cross-task reusable items (tactics, rules, templates)
- **L3 Promoted**: Verified, immutable artifacts (production recipes)
"""

from __future__ import annotations

__all__: list[str] = []
