"""STEP 2 check (iv): closing the gripper ON the coke can vs on nothing.

The can is teleported between the fingers and pinned there (pose+velocity reset
every substep), so the only difference between the two branches is whether the
fingers meet an object.  Identical command, identical reset, identical arm.
"""
import os, sys, numpy as np
os.environ.setdefault("VK_ICD_FILENAMES","/etc/vulkan/icd.d/nvidia_icd.json"); os.environ["DISPLAY"]=""
np.set_printoptions(precision=5, suppress=True, linewidth=230)
import simpler_env, sapien.core as sapien
env=simpler_env.make("google_robot_pick_coke_can"); base=env.unwrapped
GSIGN=float(sys.argv[1]) if len(sys.argv)>1 else -1.0
NSTEP=6

def run(with_can):
    obs,_=env.reset(options={"obj_init_options":{"episode_id":0}})
    robot=base.agent.robot; aj=robot.get_active_joints()
    K=np.array([j.stiffness for j in aj]);D=np.array([j.damping for j in aj]);FL=np.array([j.force_limit for j in aj])
    DT=base._scene.get_timestep(); RIDS={l.get_id() for l in robot.get_links()}
    FIDS={l.get_id() for l in robot.get_links() if 'finger' in l.name}
    can = base.obj
    tcp=np.asarray(base.tcp.pose.p)
    tgt = sapien.Pose(tcp if with_can else np.array([10.,10.,10.]), can.pose.q)
    can.set_pose(tgt)
    L=[]; pre={}
    ob=base.agent.before_simulation_step
    def bss():
        can.set_pose(tgt); can.set_velocity(np.zeros(3)); can.set_angular_velocity(np.zeros(3))
        ob(); pre.update(qt=robot.get_drive_target().copy(), vt=robot.get_drive_velocity_target().copy(),
                         qf=robot.get_qf().copy())
    base.agent.before_simulation_step=bss
    oa=base._after_simulation_step
    def ass():
        oa()
        q=robot.get_qpos().copy(); v=robot.get_qvel().copy()
        tau=np.clip(K*(pre['qt']-q)+D*(pre['vt']-v),-FL,FL)
        cm=0.0; cg=0.0
        for c in base._scene.get_contacts():
            i0=c.actor0.get_id() in RIDS; i1=c.actor1.get_id() in RIDS
            if i0==i1: continue
            imp=np.zeros(3)
            for p in c.points: imp=imp+np.asarray(p.impulse)
            m=np.linalg.norm(imp); cm+=m
            if (c.actor0.get_id() in FIDS) or (c.actor1.get_id() in FIDS): cg+=m
        L.append((q,v,tau,pre['qf'].copy(),cm/DT,cg/DT))
    base._after_simulation_step=ass
    a=np.zeros(7); a[6]=GSIGN
    for _ in range(NSTEP): base.step_action(a)
    base.agent.before_simulation_step=ob; base._after_simulation_step=oa
    return L,[j.name for j in aj]

A,names=run(True); B,_=run(False)
FI=[i for i,n in enumerate(names) if 'finger' in n]
print("joints",names,"finger idx",FI, "gripper action sign", GSIGN)
n=min(len(A),len(B))
print(f"{'sub':>5} {'qfingA':>9} {'qfingB':>9} {'|vfingA|':>9} {'tau_fingA':>10} {'tau_fingB':>10} {'ratio':>8} {'cf_gripA':>9} {'cf_gripB':>9} {'qfA(N)':>9}")
for i in range(0,n,50):
    qa,va,ta,fa,cma,cga=A[i]; qb,vb,tb,fb,cmb,cgb=B[i]
    fA=np.abs(ta[FI]).sum(); fB=np.abs(tb[FI]).sum()
    print(f"{i:5d} {qa[FI].mean():9.5f} {qb[FI].mean():9.5f} {np.abs(va[FI]).max():9.5f} {fA:10.4f} {fB:10.4f} "
          f"{fA/max(fB,1e-9):8.2f} {cga:9.3f} {cgb:9.3f} {np.abs(fa[FI]).sum():9.5f}")

import numpy as _np
_np.save('grip_A.npy', _np.array([_np.concatenate([x[0],x[1],x[2],x[3],[x[4]],[x[5]]]) for x in A]))
_np.save('grip_B.npy', _np.array([_np.concatenate([x[0],x[1],x[2],x[3],[x[4]],[x[5]]]) for x in B]))
w=slice(int(0.5*n),n)
tA=np.array([x[2] for x in A[w]]); tB=np.array([x[2] for x in B[w]])
qfA=np.array([x[3] for x in A[w]]); qfB=np.array([x[3] for x in B[w]])
print(f"\nsecond half of the close ({tA.shape[0]} substeps):")
print(f"  mean |tau_finger| ON can = {np.abs(tA[:,FI]).sum(1).mean():.4f} N   on nothing = {np.abs(tB[:,FI]).sum(1).mean():.4f} N   ratio {np.abs(tA[:,FI]).sum(1).mean()/max(np.abs(tB[:,FI]).sum(1).mean(),1e-12):.2f}")
print(f"  mean |qf_finger|  ON can = {np.abs(qfA[:,FI]).sum(1).mean():.6f} N   on nothing = {np.abs(qfB[:,FI]).sum(1).mean():.6f} N")
print(f"  mean gripper contact force ON can = {np.mean([x[5] for x in A[w]]):.3f} N   on nothing = {np.mean([x[5] for x in B[w]]):.3f} N")
print(f"  finger qpos end: on can {A[-1][0][FI]}  on nothing {B[-1][0][FI]}")
