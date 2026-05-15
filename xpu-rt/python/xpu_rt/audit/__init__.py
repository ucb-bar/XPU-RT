"""Trust, realness, and fresh-agent audit layer.

This package implements the permanent audit gate every milestone after
Section 19 must pass. It is the layer that turns "no stubs, real examples"
from a habit into a checkable contract.

Key components:
- :mod:`xpu_rt.audit.contracts` — per-feature realness contracts
  (machine-readable claim records)
- :mod:`xpu_rt.audit.caveat_ledger` — machine-readable caveats
- :mod:`xpu_rt.audit.errors` — typed audit errors
- :mod:`xpu_rt.audit.realness_scan` — source-level no-stub scan
- :mod:`xpu_rt.audit.import_provenance` — runtime import scan
- :mod:`xpu_rt.audit.trace_replay` — deterministic replay of agent
  decisions from artifact hashes alone
- :mod:`xpu_rt.audit.fresh_agent` — task-pack builder for
  fresh-Claude-Code reproducibility
- :mod:`xpu_rt.audit.perturbations` — holdout-style perturbations to
  catch hardcoded behavior
- :mod:`xpu_rt.audit.negative_controls` — fault-injection tests that
  prove gates fire
- :mod:`xpu_rt.audit.trust_report` — single-page aggregator
"""

from __future__ import annotations
