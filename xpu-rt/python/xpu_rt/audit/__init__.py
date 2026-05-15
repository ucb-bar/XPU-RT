"""Trust, realness, and fresh-agent audit layer.

This package implements the permanent audit gate every milestone after
Section 19 must pass. It is the layer that turns "no stubs, real examples"
from a habit into a checkable contract.

Key components:
- :mod:`compgen.audit.contracts` — per-feature realness contracts
  (machine-readable claim records)
- :mod:`compgen.audit.caveat_ledger` — machine-readable caveats
- :mod:`compgen.audit.errors` — typed audit errors
- :mod:`compgen.audit.realness_scan` — source-level no-stub scan
- :mod:`compgen.audit.import_provenance` — runtime import scan
- :mod:`compgen.audit.trace_replay` — deterministic replay of agent
  decisions from artifact hashes alone
- :mod:`compgen.audit.fresh_agent` — task-pack builder for
  fresh-Claude-Code reproducibility
- :mod:`compgen.audit.perturbations` — holdout-style perturbations to
  catch hardcoded behavior
- :mod:`compgen.audit.negative_controls` — fault-injection tests that
  prove gates fire
- :mod:`compgen.audit.trust_report` — single-page aggregator
"""

from __future__ import annotations
