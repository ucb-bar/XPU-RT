# Warehouse sensor-fusion nav — shipped model weights

These are the trained checkpoints for the warehouse drone sensor-fusion navigation experiment, shipped
in-repo so the demo/eval run from a fresh clone **without** any external `train_out/` directory. See
`docs/warehouse_sensorfusion_reproduce.md` for how they're used, and how to retrain them from scratch.

| file | model | what it is |
|---|---|---|
| `nav_fused_v12_cnn.pt` | FusedSensorNet (CNN vision + cross-ToF + state → yaw_rate, forward_speed), 3×128 LSTM | the **guidance** net — behavior-cloned on the warehouse gate course. crowded-collidable ~25–42%, prop-free ~100%. |
| `rl_controller_velctrl_dr4.pt` | PPO actor MLP `16→256→128→64→4` ELU (obs-norm off), DirectThrustMoment | the **low-level controller** — velocity-tracking, domain-randomized (DR4). warehouse gate-nav ~50% at `moment_scale 0.006`. |
| `yolov8n_gate_person_128x192.pt` | YOLOv8n, nc=2 {gate, person}, rect 128×192 | **detector** (deployment default). test mAP50: gate 0.975 / person 0.79. K1 rvv_x60 int8 ≈ 184 ms / 5.4 Hz. |
| `yolov8n_gate_person_64x96.pt` | YOLOv8n, nc=2 {gate, person}, rect 64×96 | cheaper detector variant. gate 0.852 / person 0.578. K1 rvv_x60 int8 **46 ms / 21.6 Hz, bit-exact** (4.9× the old 160² build). |

The script defaults in `sims/scripts/record_sensor_demo.py` (and `--weights`/`--rl_checkpoint`/`--yolo` in the
eval scripts) point here, so `record_sensor_demo.py` runs with no arguments beyond the env flags.

Provenance (for retraining): nav via the fused-BC pipeline; controller via
`sims/scripts/train_steering_tracking.py --task Isaac-Track-VelocityCtrl-DR-Crazyflie-v0`; YOLO via
`sims/scripts/gen_yolo_dataset.py` (nc=2, 60×90 deployment-geometry frames) + `sims/scripts/train_yolo.py`
(`--rect`, `--imgsz 96` → 64×96, or `--imgsz 192` → 128×192).
