"""Flow C — ModelBlaster's front end, a QNN back end.

PyTorch models are ingested exactly the way the RISC-V flow ingests them
(modelblaster's `extract_graph` → `graph.json` IR), scheduled by the same
`xpu-rt` scheduler, and ingested back through modelblaster's own
`ingest_xpurt_schedule`.  Only the last stage differs: instead of emitting
Zephyr C that calls generated kernels on harts, we emit Linux C++ that
calls `QnnGraph_execute` on backend lanes.

See README.md for the stage-by-stage flow and what is reused verbatim.
"""
