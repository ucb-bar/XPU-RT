# External data the repository does not carry

Two inputs are supplied by the reader. Nothing in the no-hardware gate or the audited figure set reads
them; they are needed only to retrain the navigation models and to run the ViNT forest demo.

| path in a working tree | what it holds | used by |
|---|---|---|
| `datasets/` | navigation training data: `idsia`, `udacity`, `pulp_dronet_himax`, `sim_forest`, `isaaclab_forest_renders` | `docs/Demo/replicate_forest_trail_demo.md`, `docs/Artifact/environment.md` |
| `sims/external/visualnav-transformer/` | a checkout of the ViNT / visualnav-transformer code | `docs/Demo/replicate_forest_trail_demo.md` |

Point each path at your own copy, e.g. `ln -s /path/to/datasets datasets` and
`ln -s /path/to/visualnav-transformer sims/external/visualnav-transformer`. Both paths are ignored by git.

The raw board traces are the third external input. The gate does read them; they are fetched as
archives, described in [Raw traces](#raw-traces-google-drive).

## Raw traces (Google Drive)

The raw board traces and long run logs are not in git. They are stored as three gzip tars, split by
top-level directory, beside the display-dump tars in `results/codesign_feedback/archive_v3/`
(gitignored). The tars' sha256 and sizes are in the tracked `archive_v3/MANIFEST.sha256`, and the
sha256 of every member is in a tracked `ARCHIVED*.sha256` list next to the data.

Google Drive link: TBD

| archive | size | sha256 | holds | needed by |
|---|---|---|---|---|
| `board_traces_ros_traced_2026-09-29.tar.gz` | 696,665,198 B (4.28 GB unpacked) | `fd0b913b6b9f012be62abb8638da6c612f2fbfb23529cad4d42ed35da8561fb4` | every `results/codesign_feedback/ros_traced/**/trace.csv` (1579 files: run-level and per-node ROS 2 dispatch traces). Member list: `ros_traced/ARCHIVED.sha256` | `measured_timing.py --verify` (ROS constants), `make_measured_gantt_pair.py` (the ROS row of panel I in every showdown figure, through the ROS arms' `measured_gantt_*` sidecars), `verify_showdown_figure.py`, `verify_ros_model_fidelity.py` (`ros_model_fidelity_*`), `verify_board_traces.py`, `verify_archived_dumps.py` |
| `board_traces_xpurt_long_2026-09-29.tar.gz` | 73,384,944 B (0.35 GB unpacked) | `e9f59ba9f277f244051abe5884a2fcba3d311bf41c28601998a470edf2fc395b` | every `results/codesign_feedback/xpurt_long/trace_*.csv` and `stdout_*.txt` (771 files; the 18 `*er0*` symlinks stored as files). Member list: `xpurt_long/ARCHIVED.sha256` | `measured_timing.py --verify` (XPU-RT constants), `make_measured_gantt_pair.py` (the XPU-RT row of panel I, through the XPU-RT arms' `measured_gantt_*` sidecars), `ros_ladder_figure.py` / `verify_ros_effort_ladder.py` (XPU-RT hart count), `verify_modelblaster_inputs.py` (the `w2pg36{base,ime}` medians), `verify_board_traces.py` |
| `raw_dumps_misc_2026-09-29.tar.gz` | 4,169,491 B (40 MB unpacked) | `768e63957c7bef9febd5b79581fc9b2716aabdd855201eac6347c8cee9fa1790` | `results/freshness_probe_v{2,3}/per_invocation.csv`, the `trace_*`/`stdout_*` dumps of `results/codesign_feedback/hil_feedback/misaligned_emitter/`, and the Isaac process logs under `results/codesign_feedback/*/tmp/isaaclab/logs/` (93 files). Member list: `results/ARCHIVED_raw_dumps_misc_2026-09-29.sha256` | `benchmarks/freshness_eval/plot.py` only; no figure sidecar and no gate check reads these |

The per-run evidence beside the traces stays tracked: `manifest.json`, `summary.json`, `ctrl_gaps.csv`,
`cpu.csv`, `chain.csv` and the other small CSVs of `ros_traced/<run>/`, and `cpu_*`, `manifest_*` and
`hart_*` of `xpurt_long/`. The older trace archives `board_traces_2026-09-17.tar` and
`board_traces_2026-09-24.tar` in the same directory hold the traces that were never tracked; a clean
clone needs them as well.

### Fetch, check, unpack

From the repository root, after placing the downloaded tars in `results/codesign_feedback/archive_v3/`:

```bash
AR=results/codesign_feedback/archive_v3
( cd "$AR" && grep -E '2026-09-(17|24|29)' MANIFEST.sha256 | awk '{print $1"  "$3}' | sha256sum -c )
for t in "$AR"/board_traces_2026-09-17.tar "$AR"/board_traces_2026-09-24.tar; do tar -xf "$t" -C .; done
for t in "$AR"/*_2026-09-29.tar.gz; do tar -xzf "$t" -C .; done
sha256sum -c --quiet results/codesign_feedback/ros_traced/ARCHIVED.sha256 \
                     results/codesign_feedback/xpurt_long/ARCHIVED.sha256 \
                     results/ARCHIVED_raw_dumps_misc_2026-09-29.sha256
```

The older two and the `2026-09-29` tars share 243 members, byte for byte, so the order of unpacking
does not matter. Members are stored at repo-relative paths and as regular files, never as
symlinks, so unpacking over a checkout replaces nothing git tracks.

### Run the gate against the archives

```bash
ARCHIVES="$PWD/results/codesign_feedback/archive_v3" bash artifact/verify_no_hardware.sh
```

`verify_board_traces.py`, `verify_archived_dumps.py` and `verify_modelblaster_inputs.py` read the tars
themselves (plain `.tar` and `.tar.gz`, through `scripts/archive_members.py`). With the traces on
disk they compare each file's sha256 with its archived copy; without them they count the archived
copies and print how many are not unpacked. `measured_timing.py --verify`,
`verify_ros_effort_ladder.py`, `verify_showdown_figure.py` and the Gantt producers open the traces
themselves and fail (missing trace, or DRIFT) until the tars are unpacked.

`verify_archived_dumps.py` reports a sidecar input whose archived copy matches the file on disk but
not the hash the render recorded as STALE, not FAIL: the render predates its input, which holds with
or without the archives. On 2026-09-29 this covers nine `ros_traced/*/trace.csv` read by four renders
outside the audited set (`latency_waterfall`, `warehouse_showdown_atlas`, `warehouse_showdown_final`,
`warehouse_showdown_paper10`).

`scripts/repro_clean_clone.sh` unpacks the same five tars into its clone.
