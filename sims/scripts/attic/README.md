# sims/scripts/attic — previews, probes and retired simulator tools

These scripts were run once for the study and are kept for attribution; the documented
reproduction path is in `docs/Artifact/reproduction_full.md` (§5 for the flights). Nothing here is
called by a tracked script, test or document. They are kept as they ran under the Isaac
interpreter (`ISAAC_PY`, see `scripts/env.sh`); several import a sibling from the old
`sims/scripts/` location or write to `sims/out/`, so re-running one may need its `sys.path`
or working directory adjusted. Files from `sims/scripts/pilot/` were flattened into this
directory.

| Script | What it did | Wrote | Documented |
|---|---|---|---|
| `pilot_steering_with_camera.py` | Manually pilot the trained steering policy with an FPV camera preview (forest env). | `sims/out/` previews | — |
| `play_dronet_with_mlp_policy.py` | DroNet vision → command generator → trained MLP steering policy, closed loop in the forest. | `sims/out/` videos | — |
| `compose_hil_figure.py` | Early HIL warehouse composite: isometric drone views, sensor bank with YOLO, large K1 Gantt, from `record_sensor_demo --dump_figure_data`. | `results/codesign_feedback/refined/` (superseded by `showdown_gatecourse.py` and the v3 figures) | `docs/Evaluation/figure_runbook.md` §3–4 (history) |
| `debug_forest_sensors.py` | Verification harness for the forest onboard sensor rig (greyscale frame, ToF heatmaps, cross composite). | scratchpad PNGs | — |
| `fpv_flight.py` | Physics-driven warehouse flight with onboard FPV capture, geometric pursuit controller. | `sims/out/` mp4 | — |
| `fpv_policy_flight.py` | Physics drone flown by the trained locomotion policy, onboard FPV recorded. | `sims/out/` mp4 | — |
| `hw_cycle_model.py` | Per-op cycle model of the target SoC for the model DSE (replaced by board-measured profiles). | stdout / DSE tables | — |
| `inspect_meshes_offline.py` | Offline gallery renderer of the reference drone-RL repos' geometry. | PNG gallery | — |
| `inspect_scene.py` | Canonical inspection views of any Isaac USD scene to PNG. | PNGs | — |
| `people_rich_ira.py` | Showcase: ~20 animated walking humans (IRA) plus ~500 dense props in the warehouse. | `sims/out/` mp4 | — |
| `play_warehouse_course.py` | Render a trained warehouse-course policy over the 4-gate course with a chase camera. | `sims/out/` mp4 | — |
| `preview_course_video.py` | Fast scripted fly-through of the aisle course to iterate on the look of the scene. | `sims/out/` mp4 | — |
| `preview_physics_flight.py` | Waypoint flight with the real geometric velocity controller, no policy. | `sims/out/` mp4 | — |
| `preview_warehouse.py` | Viewpoint renders of the stock Isaac warehouse as a candidate environment. | PNGs | — |
| `probe_aisle_corridor.py` | Dump every prim whose AABB intrudes into the fused-nav flight corridor. | JSON / stdout | — |
| `probe_warehouse_aisles.py` | Extract the real rack/aisle geometry of `full_warehouse` via `UsdGeom.BBoxCache`. | JSON + `out/warehouse_aisles__*.png` | — |
| `smoke_warehouse_nav.py` | Smoke test of the warehouse navigation task: instantiate, step, verify each subsystem. | stdout | — |
| `test_fused_perception.py` | End-to-end test of the forest perception stack (sensors → Madgwick → fused-model input). | stdout | — |
| `test_velocity_controller.py` | Isolation test of the geometric velocity controller on a lone Crazyflie. | stdout | — |
| `export_showdown_subimages.py` | Export every sensor sub-image of the showdown figure as its own PNG (reuses `showdown_gatecourse.py`). | PNG + PDF tiles under a caller-chosen `--out-dir` | — |

## Left in place because a tracked file cites them

`sims/scripts/utils/render_vint_calibration.py` (`ModelBlaster/models/vint.py`,
`ModelBlaster/mb_datasets/isaaclab_forest_render.py`) and
`sims/scripts/pilot/pilot_forest_with_vint.py` (`ModelBlaster/notes/vint_zephyr_plan.md`).
