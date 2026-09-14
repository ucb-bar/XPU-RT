"""STEP 2 check (ii): contact vs free at matched (qpos,qvel).

Two runs of the SAME deterministic action sequence from the SAME reset:
  A: sink present   -> the gripper presses into it
  B: sink collision disabled -> the identical command sweeps through free space
They are bit-identical until the first contact substep, so at contact onset the
two branches sit at (nearly) identical qpos/qvel by construction.
"""
import os, sys, numpy as np
os.environ.setdefault("VK_ICD_FILENAMES","/etc/vulkan/icd.d/nvidia_icd.json"); os.environ["DISPLAY"]=""
np.set_printoptions(precision=5, suppress=True, linewidth=230)
import simpler_env
TASK="widowx_put_eggplant_in_basket"
NSTEP=10
ACT=np.array([0,0,-0.03,0,0,0,1.0])

env=simpler_env.make(TASK); base=env.unwrapped

def run(kill_sink):
    obs,_=env.reset(options={"obj_init_options":{"episode_id":0}})
    robot=base.agent.robot
    aj=robot.get_active_joints()
    K=np.array([j.stiffness for j in aj]);D=np.array([j.damping for j in aj]);FL=np.array([j.force_limit for j in aj])
    DT=base._scene.get_timestep()
    RIDS={l.get_id() for l in robot.get_links()}
    if kill_sink:
        for a in base._scene.get_all_actors():
            if a.name in ("sink","dummy_sink_target_plane"):
                for s in a.get_collision_shapes(): s.set_collision_groups(0,0,0,0)
    L=[]; pre={}
    ob=base.agent.before_simulation_step
    def bss():
        ob(); pre.update(qt=robot.get_drive_target().copy(), vt=robot.get_drive_velocity_target().copy(),
                         qf=robot.get_qf().copy())
    base.agent.before_simulation_step=bss
    oa=base._after_simulation_step
    def ass():
        oa()
        q=robot.get_qpos().copy(); v=robot.get_qvel().copy()
        tau=np.clip(K*(pre['qt']-q)+D*(pre['vt']-v),-FL,FL)
        cf=np.zeros(3); cm=0.0
        for c in base._scene.get_contacts():
            i0=c.actor0.get_id() in RIDS; i1=c.actor1.get_id() in RIDS
            if i0==i1: continue                      # skip robot-internal AND scene-only
            imp=np.zeros(3)
            for p in c.points: imp=imp+np.asarray(p.impulse)
            s=1.0 if i1 else -1.0
            cf=cf+s*imp; cm+=np.linalg.norm(imp)
        L.append((q,v,tau,pre['qf'].copy(),cf/DT,cm/DT,np.asarray(base.tcp.pose.p)))
    base._after_simulation_step=ass
    for _ in range(NSTEP): base.step_action(ACT)
    base.agent.before_simulation_step=ob; base._after_simulation_step=oa
    return L, [j.name for j in aj]

A,names=run(False)
B,_=run(True)
n=min(len(A),len(B))
print(f"{len(A)} substeps each, {len(names)} joints: {names}")
ARM=slice(0,6)
first=None
for i in range(n):
    if A[i][5]>1.0: first=i; break
print("first substep with external contact force > 1 N on robot (branch A):", first)
hdr=f"{'sub':>5} {'|dq|inf':>9} {'|dv|inf':>9} {'|dqf|inf':>9} {'S(tau^2)A':>11} {'S(tau^2)B':>11} {'ratio':>8} {'cfA(N)':>9} {'cfB(N)':>9} {'|qd|A':>8}"
print(hdr)
for i in list(range(max(first-2,0), min(first+40,n),2))+[first+60,first+120,first+250]:
    if i>=n: continue
    qa,va,ta,fa,_,cma,_=A[i]; qb,vb,tb,fb,_,cmb,_=B[i]
    print(f"{i:5d} {np.abs(qa-qb).max():9.2e} {np.abs(va-vb).max():9.2e} {np.abs(fa-fb).max():9.2e} "
          f"{(ta[ARM]**2).sum():11.4f} {(tb[ARM]**2).sum():11.4f} {(ta[ARM]**2).sum()/max((tb[ARM]**2).sum(),1e-12):8.2f} "
          f"{cma:9.3f} {cmb:9.3f} {np.abs(va[ARM]).max():8.4f}")
# summary over the sustained-press window
w=slice(first, min(first+400,n))
ta=np.array([x[2] for x in A[w]]); tb=np.array([x[2] for x in B[w]])
fa=np.array([x[3] for x in A[w]]); fb=np.array([x[3] for x in B[w]])
qa=np.array([x[0] for x in A[w]]); qb=np.array([x[0] for x in B[w]])
va=np.array([x[1] for x in A[w]]); vb=np.array([x[1] for x in B[w]])
print(f"\nover the {ta.shape[0]}-substep press window after contact onset:")
print(f"  mean Sum tau_drive^2  contact={np.mean((ta[:,ARM]**2).sum(1)):10.4f}   free={np.mean((tb[:,ARM]**2).sum(1)):10.4f}   ratio={np.mean((ta[:,ARM]**2).sum(1))/np.mean((tb[:,ARM]**2).sum(1)):8.2f}")
print(f"  mean Sum qf^2         contact={np.mean((fa[:,ARM]**2).sum(1)):10.4f}   free={np.mean((fb[:,ARM]**2).sum(1)):10.4f}   ratio={np.mean((fa[:,ARM]**2).sum(1))/np.mean((fb[:,ARM]**2).sum(1)):8.2f}")
print(f"  max |dq|inf {np.abs(qa-qb).max():.3e}   max |dv|inf {np.abs(va-vb).max():.3e}")
np.save('./press_A.npy', np.array([np.concatenate([x[0],x[1],x[2],x[3],[x[5]]]) for x in A]))
np.save('./press_B.npy', np.array([np.concatenate([x[0],x[1],x[2],x[3],[x[5]]]) for x in B]))
