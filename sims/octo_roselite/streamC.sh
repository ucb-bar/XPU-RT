source /scratch2/dima/miniforge3/etc/profile.d/conda.sh; conda activate octo_sim
cd /scratch2/dima/misc_sw/octo_work/sim_eval
set -x
for r in 0 2 4; do
  python run_eval.py --ckpt hf://rail-berkeley/octo-small --task widowx_spoon_on_towel --n 24 --init-rng $r --save-video-every 8
done
for r in 0 2 4; do
  python run_eval.py --ckpt hf://rail-berkeley/octo-small --task widowx_put_eggplant_in_basket --n 24 --init-rng $r --save-video-every 8
done
