source ../miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
cd ../octo_work/sim_eval/roselite
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
