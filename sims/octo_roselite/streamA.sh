source /scratch2/dima/miniforge3/etc/profile.d/conda.sh; conda activate octo_sim
cd /scratch2/dima/misc_sw/octo_work/sim_eval
set -x
python run_eval.py --task widowx_put_eggplant_in_basket --n 24 --init-rng 0
python run_eval.py --task widowx_put_eggplant_in_basket --n 24 --init-rng 0 --legacy-unnorm --save-video-every 8
python run_eval.py --task widowx_put_eggplant_in_basket --n 24 --init-rng 2 --save-video-every 8
python run_eval.py --task widowx_put_eggplant_in_basket --n 24 --init-rng 4 --save-video-every 8
