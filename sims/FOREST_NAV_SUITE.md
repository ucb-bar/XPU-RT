# Forest Nav + Sensor-Fusion + Model-DSE Suite

Reproducibility catalog for the drone autonomy stack: onboard-sensor-driven navigation
(trail-following → goal-conditioned gate navigation with obstacle avoidance) and the
model→hardware co-design (quality-vs-cost DSE). All commands use the Isaac Lab python:

```
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
```
Run training/collection from a writable CWD (`train_out/`); Isaac `close()` hangs → scripts
`os._exit(0)`; `--headless` is required for the onboard cameras to render.

---

## 1. Environments (registered gym-ids)

Forest trail, corridor half-width 1.5 m, drone cruises z≈1.0. `*_WithSensors` variants attach the
full onboard rig (front HM01B0 greyscale cam + 4× VL53L5CX cross ToF + downward ToF/optical-flow/baro).

| gym-id | what |
|---|---|
| `Isaac-Forest-Trail-Vision-Crazyflie-Play-v0` (+ `-WithHumans`, `-Curved-`) | straight/curved trail |
| `Isaac-Forest-Trail-...-Play-WithSensors-v0` | + full onboard sensor rig |
| `Isaac-Forest-Trail-Slalom-Vision-Crazyflie-Play-WithSensors-v0` | in-corridor slalom obstacles (avoidance) |
| `Isaac-Forest-Gates-Vision-Crazyflie-Play-WithSensors-v0` | **4-gate goal-conditioned course** |
| `Isaac-Track-BodyRate-Crazyflie-{v0,DR-v0}` | CTBR body-rate control task (+ domain-randomized) |

Scene/sensor code: `isaaclab_tasks/forest_trail/{forest_scene,sensors,gates,expert,state_estimator}.py`;
env cfgs + registration under `forest_trail/config/crazyflie/`.

## 2. Models

| model | file | role | result |
|---|---|---|---|
| analytic body-rate controller | `eval_bodyrate_tracking.py` (fixed) | inner control loop | RMS 0.36 rad/s, beats RL |
| DroNet greyscale (cls/reg) | `qnn_models/dronet.py` | vision-only trail follow | trail 100%, offset 0.38–1.0 m |
| TrailViT (vitfly backbone) | `vitfly/models/trail_vit.py` | vision-only trail follow | trail 100%, offset 0.32 m |
| **FusedSensorNet** | `vitfly/models/fused_model.py` | goal-conditioned gate nav | **gate 6/6, offset 0.264 m** |

FusedSensorNet: greyscale→ViT ⊕ 4×8×8 ToF→conv ⊕ low-dim state→LSTM → (yaw_rate, forward_speed);
every sensor a maskable/zero-skip slot (cam-off = 0.88M MACs, 32× cheaper).

## 3. Pipelines

**Control (RL/analytic):** `train_steering_tracking.py --task Isaac-Track-BodyRate-...`;
eval `sims/scripts/eval_bodyrate_tracking.py`; robustness `sims/scripts/eval_robustness.py`.

**Greyscale nav:** train `sims/training/train_dronet.py` / `train_trail_vit.py`;
closed-loop `sims/scripts/eval_forest_nav.py --nav_arch {dronet,vit} --trail {straight,curved}`.

**Fused goal-conditioned gate nav (headline):**
```
# collect (expert flies the gates, dumps sensor+goal seqs; DART recovery noise)
$PY sims/training/collect_fused_data.py --headless --trail gate --episodes 20 --noise_std 0.15 --out <d>/fused_gate.pt
# train Stage-1 (mapped goal)   and   Stage-2 (vision goal, --mask_desired_vel → gate from camera, no YOLO)
$PY sims/training/train_fused.py --data <d>/fused_gate.pt --epochs 45 --out_dir <o>
$PY sims/training/train_fused.py --data <d>/fused_gate.pt --epochs 45 --mask_desired_vel --out_dir <o_vision>
# fly it
$PY sims/scripts/eval_forest_nav_fused.py --headless --weights <o>/best.pt --trail gate --episodes 6
$PY sims/scripts/eval_forest_nav_fused.py --headless --weights <o_vision>/best.pt --trail gate --mask_off desired_vel
```
KEY finding: trail-following BC from a centred expert fails (covariate shift); **goal-conditioning
fixes it** — Stage-1 (mapped goal) AND Stage-2 (vision goal, no YOLO) both **100% / 6-gate flight**.

## 4. Model DSE (co-design)

```
$PY sims/scripts/dse_pareto.py                    # quality-vs-cost Pareto (MACs/latency/energy/int8) + png
$PY vitfly/models/ablate_fused_compute.py         # per-sensor-subset compute (zero-skip)
$PY sims/scripts/eval_fused_ablation.py --weights <fused>/best.pt --trail gate   # per-subset FLIGHT quality
$PY sims/training/qat_dronet.py --checkpoint <dronet>/best.pt --head classifier  # real int8 (measured lossless)
```
Pareto frontier: dronet-cls (11M MACs) / vit-reg (28M) / fused-gate-mapped (28.6M, best 0.264 m).
int8 measured LOSSLESS on DroNet (fp32 0.543 → int8 0.544). Energy + int8-latency are labeled
op-count ESTIMATES; measured SoC latency needs the ModelBlaster/FireSim pipeline (profiles absent).

## 5. Key results at a glance

- **Control:** analytic ≫ RL (RMS 0.36 vs 0.83); DR flattens OOD degradation but costs accuracy.
- **Vision-only trail nav:** ~0.60 accuracy ceiling (feature-limited, not architecture); flies 100%,
  but offline accuracy did NOT predict flight (classifier's smooth softmax-yaw tracks tighter).
- **Sensor-fusion gate nav:** goal-conditioned FusedSensorNet flies the 4-gate course **100%**, mapped
  goal AND vision goal (no YOLO) both 0.26–0.27 m; drives at zero-extra-network gate perception.
- **DSE:** clean Pareto; int8 is free on DroNet; camera ViT dominates compute (zero-skip = 32× lever).
