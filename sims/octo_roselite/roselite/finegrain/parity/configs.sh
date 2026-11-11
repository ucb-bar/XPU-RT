# The five arm-controller / actuation combinations under test. Gripper is held
# fixed at the registered planner variant throughout, so the ARM is the only
# variable. Combined mode name is "<arm>_<gripper>" (google_robot/defaults.py).
GRIP=gripper_pd_joint_target_delta_pos_interpolate_by_planner
CM_STOCK="arm_pd_ee_delta_pose_align_interpolate_by_planner_${GRIP}"       # pose-relative + planner (registered)
CM_TGTPLAN="arm_pd_ee_target_delta_pose_align_interpolate_by_planner_${GRIP}"  # target-accumulating + planner
CM_TGT="arm_pd_ee_target_delta_pose_align_${GRIP}"                          # target-accumulating, no planner
# name | actuation | tick-hz | control-mode
CONFIGS=(
  "A_stock_native|native|3|$CM_STOCK"
  "B_tgtplan_native|native|3|$CM_TGTPLAN"
  "C_tgt_native|native|3|$CM_TGT"
  "D_tgtplan_fine27|fine|27|$CM_TGTPLAN"
  "E_tgt_fine27|fine|27|$CM_TGT"
)
# Non-planner gripper: the last untested lever for coke at 27 Hz. E kept the registered
# planner gripper, which suffers the same path truncation as the arm did.
GRIP_NP=gripper_pd_joint_target_delta_pos
CM_TGT_NPG="arm_pd_ee_target_delta_pose_align_${GRIP_NP}"
