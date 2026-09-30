# ros_traced/

One directory per ROS 2 board run (`<camera Hz>_<arm>_r<replicate>`). The small per-run evidence is
tracked: `manifest.json`, `summary.json`, `ctrl_gaps.csv`, `cpu.csv`, `chain.csv`, `consumed.csv`,
`released.csv`, `goals.csv`, `kernel_sha256.txt` and the node logs.

The dispatch traces (`trace.csv` and `<node>/trace.csv`, 4.28 GB) are not tracked. They are in
`archive_v3/board_traces_ros_traced_2026-09-29.tar.gz` and the two older `archive_v3/board_traces_*.tar`;
`ARCHIVED.sha256` lists the sha256 of every member of the first. Fetching, checking and unpacking:
`docs/Artifact/external_data.md`, "Raw traces".
