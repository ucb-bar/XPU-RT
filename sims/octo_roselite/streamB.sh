source /scratch2/dima/miniforge3/etc/profile.d/conda.sh; conda activate octo_sim
cd /scratch2/dima/misc_sw/octo_work/sim_eval
set -x
python run_eval.py --task widowx_spoon_on_towel      --n 24 --init-rng 0 --save-video-every 8
python run_eval.py --task widowx_carrot_on_plate     --n 24 --init-rng 0 --save-video-every 8
python run_eval.py --task widowx_stack_cube          --n 24 --init-rng 0 --save-video-every 8
