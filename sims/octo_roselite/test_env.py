import os
os.environ.setdefault("VK_ICD_FILENAMES","/etc/vulkan/icd.d/nvidia_icd.json")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np, simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
task="widowx_put_eggplant_in_basket"
env = simpler_env.make(task)
obs, reset_info = env.reset()
instr = env.get_language_instruction()
print("TASK:", task)
print("INSTRUCTION:", repr(instr))
img = get_image_from_maniskill2_obs_dict(env, obs)
print("image:", img.shape, img.dtype, "mean", img.mean().round(1))
print("action_space:", env.action_space)
import imageio; imageio.imwrite("env_smoke.png", img)
# random step
for i in range(3):
    a = env.action_space.sample()*0
    obs, rew, done, trunc, info = env.step(a)
print("step ok. info keys:", sorted(info.keys()))
print("success:", info.get("success"))
print("max steps (trunc horizon):", env.spec.max_episode_steps if env.spec else "?")
