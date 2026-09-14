"""Probe 6: quantify the reconstruction error against the equation of motion.

balance_passive_force=True means qf = g(q)+C(q,qd)qd EXACTLY cancels the passive
terms, so in free space the articulation obeys  M(q) qddot = tau_drive.  Compare
the pre-step and post-step reconstructions against M @ (dv/dt).
"""
import os, sys, numpy as np
os.environ.setdefault("VK_ICD_FILENAMES","/etc/vulkan/icd.d/nvidia_icd.json"); os.environ["DISPLAY"]=""
np.set_printoptions(precision=4,suppress=True,linewidth=220)
import simpler_env
TASK=sys.argv[1]; env=simpler_env.make(TASK); base=env.unwrapped
obs,_=env.reset(options={"obj_init_options":{"episode_id":0}})
robot=base.agent.robot; aj=robot.get_active_joints()
K=np.array([j.stiffness for j in aj]);D=np.array([j.damping for j in aj]);FL=np.array([j.force_limit for j in aj])
DT=base._scene.get_timestep(); RIDS={l.get_id() for l in robot.get_links()}
ARM=slice(0,6) if robot.dof==8 else slice(0,7)
R=[]; pre={}
ob=base.agent.before_simulation_step
def bss():
    ob(); pre.update(q=robot.get_qpos().astype(float),v=robot.get_qvel().astype(float),
                     qt=robot.get_drive_target().astype(float),vt=robot.get_drive_velocity_target().astype(float),
                     M=robot.compute_manipulator_inertia_matrix().astype(float))
base.agent.before_simulation_step=bss
oa=base._after_simulation_step
def ass():
    oa(); q=robot.get_qpos().astype(float); v=robot.get_qvel().astype(float)
    ncon=0
    for c in base._scene.get_contacts():
        i0=c.actor0.get_id() in RIDS; i1=c.actor1.get_id() in RIDS
        if i0==i1: continue
        if any(np.any(p.impulse) for p in c.points): ncon+=1
    tpre=np.clip(K*(pre['qt']-pre['q'])+D*(pre['vt']-pre['v']),-FL,FL)
    tpost=np.clip(K*(pre['qt']-q)+D*(pre['vt']-v),-FL,FL)
    R.append((pre['M']@((v-pre['v'])/DT), tpre, tpost, ncon))
base._after_simulation_step=ass
rng=np.random.default_rng(0)
for i in range(14):
    a=np.zeros(7); a[:3]=rng.normal(0,0.02,3); a[3:6]=rng.normal(0,0.05,3); a[6]=1.0
    base.step_action(a)
Ma=np.array([r[0] for r in R]); Tp=np.array([r[1] for r in R]); Tq=np.array([r[2] for r in R]); NC=np.array([r[3] for r in R])
free=NC==0
print(f"{TASK}: {len(R)} substeps, {free.sum()} with NO external contact on the robot")
for nm,T in (("pre-step  K(qt-q_pre )+D(vt-v_pre )",Tp),("post-step K(qt-q_post)+D(vt-v_post)",Tq)):
    e=np.linalg.norm((Ma-T)[free][:,ARM],axis=1); s=np.linalg.norm(Ma[free][:,ARM],axis=1)
    rel=e/np.maximum(s,1e-9)
    print(f"  {nm}: median |M.qddot - tau|/|M.qddot| = {np.median(rel)*100:7.3f}%   "
          f"p90 {np.percentile(rel,90)*100:8.3f}%   max abs resid {e.max():.4g} N.m")
print(f"  scale: median |M.qddot| over arm joints = {np.median(np.linalg.norm(Ma[free][:,ARM],axis=1)):.4g} N.m,"
      f" max {np.linalg.norm(Ma[free][:,ARM],axis=1).max():.4g}")
