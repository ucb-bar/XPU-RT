# Warehouse Navigation — faithful build-spec (contract)

Re-audit of aerial_gym `navigation_task` (line-cited) mapped onto verified Isaac Lab 2.3.2
constructs. This is the contract for a **faithful** implementation — the previous
`warehouse_nav` was a skeleton (goal reward + collision only; no obstacle field, no
curriculum, no vision, wrong action). Every number below is cited to aerial_gym source.

## Design decisions (deviations from aerial_gym, deliberate)

1. **Obstacle content = warehouse props, not aerial_gym's abstract panels/meshes.**
   Keep aerial_gym's *mechanism* verbatim (a spawned pool, a curriculum-controlled active
   count, teleport-unused-to-z=-1000, 15% half-density), but the obstacles are pallets /
   boxes / crates / cones (SimReady) so the scene is realistic and YOLO-detectable — which
   is the entire reason we moved to the warehouse. The abstract 11x8x6 box + 6 walls is
   replaced by the warehouse's own volume; "walls" = the warehouse shell (always present).
2. **Vision = raw depth into OUR encoder, not aerial_gym's frozen VAE.**
   aerial_gym feeds a 64-D latent from a pretrained VAE. We instead expose the depth image
   and let the policy's own CNN encode it (rsl_rl 5.0.1 native `RslRlCNNModelCfg`/`CNNModel`).
   Rationale: the project's point is that OUR models (vitfly zoo, DroNet) + QAT do the
   perception for the DSE; a borrowed frozen VAE is aerial_gym baggage. (A `--vae` option can
   load their `.pth` later for a strict A/B, but it is not the default.)
3. **Observation = multi-modal, asymmetric (user requirement).** Three groups:
   - `vision` (actor): TiledCamera depth `(H,W,1)` — `concatenate_terms=False`.
   - `proprio` (actor+critic): lin/ang vel, gravity, height, goal vec+dist, last action.
   - `privileged` (critic only): nearest-obstacle rel-pos/vel, exact goal error, obstacle count.
   rsl_rl `obs_groups = {"actor": ["vision","proprio"], "critic": ["proprio","privileged"]}`.
   Toggle `vision_only=True` → actor uses only `vision`; `state_only=True` → drop the camera.
4. **Upstream bugs: FIX, keep a `strict_reference` flag.** aerial_gym ships three real bugs
   (below). Default = fixed; `strict_reference=True` reproduces them for exact comparison.

## The three aerial_gym bugs (decide per `strict_reference`)

- **Positive-biased goal-vector noise**: `rand_like(vec - 0.5)` discards the `-0.5`, so obs
  noise is `+U[0,0.2)` not centered (`navigation_task.py:374`). FIX: center it.
- **Dead depth-proximity penalty**: masked by `terminations < 0` on a non-negative counter →
  never fires (`navigation_task.py:355`). FIX: mask on `terminations == 0` (non-crashed).
- **Half-density removes walls**: temporary floor `9//2=4` lets `k//2` dump wall bodies
  (`env_manager.py:287-295`). FIX: never teleport the warehouse shell.

## Spec (cited)

| Piece | aerial_gym value (cite) | Isaac construct |
|---|---|---|
| Rates | dt=0.01, 10 substeps → 10 Hz, 100 steps=10 s (`env_with_obstacles.py:29-30`, `nav_cfg.py:19`) | `SimulationCfg(dt=0.01)`, `decimation=10`, `episode_length_s=10.0` |
| Obstacle pool | 3 panels + 35 objects + 6 walls = 44; keep-floor 9 (`env_object_config.py:66,274`; `asset_loader.py:148-185`) | `RigidObjectCollection` (props); walls = warehouse shell |
| Active count | teleport idx≥k to −1000 (`asset_manager.py:51-71`) | `write_object_pose_to_sim` z=−1000 for unused |
| 15% half-density | Bernoulli(0.15)→k//2 (`env_manager.py:283-295`) | in the reset `EventTerm` |
| Curriculum | 15→50, +2/−1 @ succ>0.7/<0.6, every 2048 eps (`nav_cfg.py:62-69`, `nav.py:234-273`) | `CurriculumTermCfg`, manual 2048-episode accumulator |
| Progress frac | (level−15)/35 → reward ×(1+2f), abs-penalty ×f (`nav.py:258,446,486-500`) | shared buffer read by reward func |
| Goal | ratio x∈[.90,.94] y∈[.10,.90] z∈[.10,.90], far +x (`nav_cfg.py:26-27`) | `GoalPositionCommand` (already built; retune ranges) |
| Action | polar velocity: ch0+=1; vx=ch0·cos(π/4·ch1); vy=0; vz=ch0·sin(π/4·ch1); yawrate=ch2·π/3; max_speed=2 (`nav_cfg.py:87-117`) | custom `ActionTerm` + velocity controller (NEW; replaces thrust/moment) |
| Depth cam | 240×135, HFOV 87°, near .2 far 10, /10 (`base_depth_camera_config.py:15-16,46-52`) | `TiledCameraCfg` depth |
| Obs (81-D) | goal_unit(3)+dist(1)+roll/pitch(2)+0(1)+linvel(3)+angvel(3)+lastact(4)+latent(64) (`nav.py:369-390`) | our multi-group version (raw depth instead of latent) |
| Reward | M·(5·e^{−d²/3.5} + 5·e^{−2d²} + gc[10/20 asym] + (20−d)/20) + diff_pen + f·abs_pen; collision −100 (`nav.py:435-521`, `nav_cfg.py:29-48`) | single custom reward func (has all coeffs) |
| Obstacle proximity | **use analytic distance from collection poses** (RayCaster is warp-static-only → YELLOW, won't hit moving pool) | `min ‖drone − obs_i‖` over active obstacles |
| Termination | crash contact>0.05N; timeout 100 steps; NO goal term (`env_with_obstacles.py:36`, `nav.py:312-316`) | `ContactSensor` + `time_out` |
| Success | arrive-and-loiter: at truncation, dist<1.0 & not crashed (`nav.py:318-322`) | custom metric term |

## Build order

1. Velocity action term + low-level velocity controller (replaces DirectThrustMoment).
2. Obstacle `RigidObjectCollection` (warehouse props) + reset event (ratio placement,
   teleport-out, 15% half-density).
3. Curriculum term (2048-episode accumulator → active count + progress_fraction buffer).
4. Reward func (all coeffs above) + analytic obstacle-proximity + arrive-and-loiter success.
5. TiledCamera depth + multi-group obs + rsl_rl CNN actor / MLP critic.
6. Train (state-only first to validate task+curriculum fast, then vision) → prove it learns.

## Verified Isaac contract (from API audit)

RigidObjectCollection GREEN (regex in parent Xform; kinematic obstacles; env_ids=long tensor).
CurriculumTermCfg GREEN (fires per-reset; mutate buffers via the live env).
TiledCamera GREEN (bbox raises → use `instance_segmentation_fast`, colorize=False → int32 ids).
Multi-group obs + native CNN GREEN (rsl_rl 5.0.1 `RslRlCNNModelCfg`; image group `concatenate_terms=False`).
ContactSensor per-object attribution GREEN (`filter_prim_paths_expr` → `force_matrix_w (N,B,M,3)`).
RayCaster YELLOW (warp static mesh only — do NOT use for the moving obstacle pool; analytic distance instead).
