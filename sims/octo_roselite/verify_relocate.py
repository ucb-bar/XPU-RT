import os, json, pickle
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import jax, numpy as np
from octo.model.octo_model import OctoModel
OUT="/scratch2/dima/misc_sw/octo_work"
print("octo module:", __import__("octo").__file__)
assert "/tmp/" not in __import__("octo").__file__, "STILL ON /tmp!"
m = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
eps = pickle.load(open(f"{OUT}/bridge_episodes.pkl","rb")); ep=eps[0]
img0, INSTR = ep['images'][0], ep['instr']
obs = {"image_primary": img0[None,None,...], "timestep_pad_mask": np.array([[True]])}
task = m.create_tasks(texts=[INSTR])
a = np.asarray(m.sample_actions(obs, task,
      unnormalization_statistics=m.dataset_statistics["bridge_dataset"]["action"],
      rng=jax.random.PRNGKey(0)))
ref = np.array(json.load(open(f"{OUT}/results_GPU.json"))["single_action"])
print("instr:", INSTR)
print("shape:", a.shape, "ref:", ref.shape)
print("new t+0:", np.round(a[0,0],6))
print("ref t+0:", np.round(ref[0,0],6))
d = np.abs(a-ref).max()
print(f"MAX ABS DIFF vs recorded reference: {d:.3e}")
print("VERDICT:", "REPRODUCED" if d < 1e-4 else "MISMATCH")
