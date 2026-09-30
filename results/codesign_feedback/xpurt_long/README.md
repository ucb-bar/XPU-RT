# xpurt_long/

One file set per XPU-RT board run (`<kind>_<label>_other_run<n>`). Tracked: `cpu_*.csv`,
`manifest_*.json`, `hart_*`.

The dispatch traces (`trace_*.csv`) and the runtime's stdout (`stdout_*.txt`), 0.35 GB, are not
tracked. They are in `archive_v3/board_traces_xpurt_long_2026-09-29.tar.gz` and the two older
`archive_v3/board_traces_*.tar`; `ARCHIVED.sha256` lists the sha256 of every member of the first.
Fetching, checking and unpacking: `docs/Artifact/external_data.md`, "Raw traces".
