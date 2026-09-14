source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
cd /home/ubuntu/simpler/sim_eval/roselite

# --- traps on this DLAMI -----------------------------------------------------
# SAPIEN otherwise selects the llvmpipe CPU device -> ErrorExtensionNotPresent.
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# JAX otherwise preallocates ~75% of the A10G.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The DLAMI exports LD_LIBRARY_PATH=/usr/local/cuda/lib64:... (CUDA 12.8), whose
# libcusolver.so.11 (11.7.3.90) SHADOWS the nvidia-cusolver-cu12 11.7.5.82 wheel
# that jaxlib 0.4.20+cuda12.cudnn89 needs. The symptom is silent and expensive:
# jax prints "CUDA backend failed to initialize: Unable to load cuSOLVER" and
# falls back to CpuDevice instead of erroring.
# But LD_LIBRARY_PATH must not simply be unset either: conda's libicui18n.so.78
# then binds against the Ubuntu 22.04 /lib/x86_64-linux-gnu/libstdc++.so.6,
# which lacks CXXABI_1.3.15, and `import sqlite3` dies. So point it at the
# conda env's own lib and nothing else.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/ubuntu/hf_cache
export DISPLAY=""
